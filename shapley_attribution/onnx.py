"""
ONNX serialization helpers for attribution models.

Two levels of serialization are supported:

1. **GBM-backed models** (``MonteCarloShapleyAttribution``,
   ``PathShapleyAttribution``): the learned ``value_model_``
   (GradientBoostingClassifier) is converted to ONNX via *skl2onnx* and
   embedded as the model graph.  Channel metadata, attribution scores, and
   model hyperparameters are stored in ``metadata_props`` so the full model
   state can be restored without sklearn at load time.

2. **Non-GBM models** (``SimplifiedShapleyAttribution``,
   ``OrderedShapleyAttribution``, heuristics): no ML subgraph is needed.
   A minimal ONNX model is created whose graph is a single Identity op on
   the attribution scores tensor.  All state is stored in ``metadata_props``.

In both cases ``load_onnx`` reconstructs the original Python object and
returns a fully-usable fitted model.

Requirements
------------
    pip install onnx skl2onnx onnxruntime

Optional runtime inference
--------------------------
For GBM-backed models you can also query the embedded value function
directly via ``onnxruntime``::

    import onnxruntime as rt
    import numpy as np

    sess = rt.InferenceSession("model.onnx")
    mask = np.array([[1, 0, 1, 0]], dtype=np.float32)  # (1, n_channels)
    prob = sess.run(None, {"input": mask})[1][0, 1]   # P(conversion)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shapley_attribution.base import BaseAttributionModel

# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def save_onnx(model: "BaseAttributionModel", path: str | Path) -> None:
    """Serialize a fitted attribution model to an ONNX file.

    Parameters
    ----------
    model : BaseAttributionModel
        A *fitted* attribution model (``is_fitted_`` must be True).
    path : str or Path
        Destination file path (e.g. ``"mc_shapley.onnx"``).

    Raises
    ------
    RuntimeError
        If the model has not been fitted.
    ImportError
        If ``onnx`` or ``skl2onnx`` are not installed.
    """
    model._check_is_fitted()
    _require_onnx()

    proto = _build_onnx_proto(model)
    Path(path).write_bytes(proto.SerializeToString())


def load_onnx(path: str | Path) -> "BaseAttributionModel":
    """Load a fitted attribution model from an ONNX file.

    Parameters
    ----------
    path : str or Path
        Path to an ONNX file previously created by :func:`save_onnx`.

    Returns
    -------
    model : BaseAttributionModel
        Fully usable fitted model (``is_fitted_`` is True).

    Raises
    ------
    ImportError
        If ``onnx`` is not installed.
    ValueError
        If the file does not contain a valid shapley-attribution ONNX model.
    """
    _require_onnx()
    import onnx as onnx_lib

    proto = onnx_lib.load(str(path))
    return _restore_from_proto(proto)


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------

def _require_onnx() -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ONNX serialization requires the 'onnx' package. "
            "Install with: pip install onnx skl2onnx onnxruntime"
        ) from e


def _meta(proto) -> dict:
    """Return metadata_props as a plain dict."""
    return {p.key: p.value for p in proto.metadata_props}


def _set_meta(proto, key: str, value: str) -> None:
    entry = proto.metadata_props.add()
    entry.key = key
    entry.value = value


def _build_onnx_proto(model: "BaseAttributionModel"):
    """Dispatch to the right serializer based on model type."""
    import onnx as onnx_lib

    has_gbm = getattr(model, "value_model_", None) is not None

    if has_gbm:
        proto = _gbm_to_onnx(model)
    else:
        proto = _metadata_only_onnx(model)

    # ---- Common metadata ----
    proto.doc_string = (
        f"shapley-attribution {type(model).__name__} — "
        "https://github.com/ianchute/shapley-attribution-model-zhao-naive"
    )
    _set_meta(proto, "sa_model_class", type(model).__name__)
    _set_meta(proto, "sa_model_module", type(model).__module__)
    _set_meta(proto, "sa_has_gbm", str(has_gbm))

    # channels_ as JSON list
    channels_list = model.channels_.tolist()
    _set_meta(proto, "sa_channels", json.dumps(channels_list))

    # attribution_ dict
    attribution_serializable = {
        str(k): float(v) for k, v in model.attribution_.items()
    }
    _set_meta(proto, "sa_attribution", json.dumps(attribution_serializable))

    # hyperparams via get_params()
    params = model.get_params()
    _set_meta(proto, "sa_params", json.dumps(params))

    # position_attribution_ (PathShapley / OrderedShapley)
    if hasattr(model, "position_attribution_"):
        pos_attr = {
            str(k): [float(x) for x in v]
            for k, v in model.position_attribution_.items()
        }
        _set_meta(proto, "sa_position_attribution", json.dumps(pos_attr))

    onnx_lib.checker.check_model(proto)
    return proto


def _gbm_to_onnx(model: "BaseAttributionModel"):
    """Convert the GBM value_model_ to an ONNX graph via skl2onnx."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as e:
        raise ImportError(
            "skl2onnx is required for saving GBM-backed models. "
            "Install with: pip install skl2onnx"
        ) from e

    n_channels = len(model.channels_)
    initial_type = [("input", FloatTensorType([None, n_channels]))]
    proto = convert_sklearn(
        model.value_model_,
        name=f"{type(model).__name__}_value_model",
        initial_types=initial_type,
        options={type(model.value_model_): {"zipmap": False}},
    )
    return proto


def _metadata_only_onnx(model: "BaseAttributionModel"):
    """Build a trivial ONNX graph (Identity) carrying only metadata."""
    import onnx as onnx_lib
    from onnx import helper, TensorProto

    n_channels = len(model.channels_)

    # Graph: input attribution_scores (1, n_channels) → Identity → output
    input_info = helper.make_tensor_value_info(
        "attribution_scores", TensorProto.FLOAT, [1, n_channels]
    )
    output_info = helper.make_tensor_value_info(
        "attribution_output", TensorProto.FLOAT, [1, n_channels]
    )
    identity_node = helper.make_node(
        "Identity", inputs=["attribution_scores"], outputs=["attribution_output"]
    )
    graph = helper.make_graph(
        [identity_node],
        f"{type(model).__name__}_graph",
        [input_info],
        [output_info],
    )
    proto = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    proto.ir_version = 8
    return proto


def _restore_from_proto(proto) -> "BaseAttributionModel":
    """Reconstruct a Python attribution model from an ONNX proto."""
    meta = _meta(proto)

    # ---- Validate ----
    if "sa_model_class" not in meta:
        raise ValueError(
            "This ONNX file does not appear to be a shapley-attribution model. "
            "'sa_model_class' metadata key is missing."
        )

    class_name = meta["sa_model_class"]
    has_gbm = meta.get("sa_has_gbm", "False") == "True"

    # ---- Reconstruct channels_ ----
    channels_raw = json.loads(meta["sa_channels"])
    channels = np.array(channels_raw)

    # ---- Reconstruct attribution_ ----
    attribution_raw = json.loads(meta["sa_attribution"])
    # Restore original key types (int if possible, else str)
    attribution = {}
    for k, v in attribution_raw.items():
        try:
            key = int(k)
        except (ValueError, TypeError):
            key = k
        attribution[key] = v

    # Keys must match channels_ dtype
    if channels.dtype.kind in ("i", "u"):  # integer channels
        attribution = {int(k): v for k, v in attribution.items()}

    # ---- Build model instance ----
    params = json.loads(meta.get("sa_params", "{}"))
    model = _instantiate_model(class_name, params)

    # ---- Populate fitted attributes ----
    model.channels_ = channels
    model.channel_to_idx_ = {ch: i for i, ch in enumerate(channels)}
    model.attribution_ = attribution
    model.conversions_ = np.array([1], dtype=int)  # sentinel — not used post-fit; keeps mean() safe
    model.is_fitted_ = True

    # ---- Restore GBM value_model_ ----
    if has_gbm:
        gbm = _onnx_to_gbm(proto)
        model.value_model_ = gbm
    else:
        model.value_model_ = None  # for models that expose the attribute

    # ---- Restore position_attribution_ ----
    if "sa_position_attribution" in meta:
        pos_raw = json.loads(meta["sa_position_attribution"])
        pos_attr = {}
        for k, v in pos_raw.items():
            try:
                key = int(k)
            except (ValueError, TypeError):
                key = k
            pos_attr[key] = [float(x) for x in v]
        model.position_attribution_ = pos_attr

    return model


def _onnx_to_gbm(proto):
    """Deserialize the embedded ONNX graph back into a sklearn GBM."""
    try:
        from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
        import onnx as onnx_lib
        from onnxconverter_common import OnnxConvertContext  # noqa: F401
    except ImportError:
        pass  # will fall through to onnxruntime wrapper

    # We use an onnxruntime wrapper so no sklearn reconstruction is needed;
    # the GBM is kept as an OrtValueModelWrapper that duck-types the
    # predict_proba interface sklearn expects.
    return _OrtGBMWrapper(proto.SerializeToString())


class _OrtGBMWrapper:
    """Duck-typed wrapper around an ONNX Runtime session for GBM inference.

    Provides a ``predict_proba(X)`` method matching the sklearn GBM interface
    so that ``PathShapleyAttribution._make_coalition_value_fn`` and
    ``MonteCarloShapleyAttribution`` can call it identically to the original
    sklearn model.
    """

    def __init__(self, onnx_bytes: bytes):
        import onnxruntime as rt
        opts = rt.SessionOptions()
        opts.log_severity_level = 3  # suppress INFO/WARNING spam
        self._session = rt.InferenceSession(onnx_bytes, sess_options=opts)
        # Detect output names: ONNX GBM has label + probabilities outputs
        output_names = [o.name for o in self._session.get_outputs()]
        # Probabilities are the second output for binary classifiers
        self._prob_output = output_names[1] if len(output_names) > 1 else output_names[0]
        self._input_name = self._session.get_inputs()[0].name

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_f32 = np.asarray(X, dtype=np.float32)
        result = self._session.run(
            [self._prob_output], {self._input_name: X_f32}
        )
        probs = result[0]
        # skl2onnx with zipmap=False returns (n_samples, n_classes) float array
        if probs.ndim == 1:
            # Binary case returned as 1-D — reshape to (n_samples, 2)
            probs = np.column_stack([1 - probs, probs])
        return probs


def _instantiate_model(class_name: str, params: dict) -> "BaseAttributionModel":
    """Import and instantiate the model class from its name."""
    from shapley_attribution import (
        SimplifiedShapleyAttribution,
        OrderedShapleyAttribution,
        MonteCarloShapleyAttribution,
        PathShapleyAttribution,
        FirstTouchAttribution,
        LastTouchAttribution,
        LinearAttribution,
        TimeDecayAttribution,
        PositionBasedAttribution,
    )

    registry = {
        "SimplifiedShapleyAttribution": SimplifiedShapleyAttribution,
        "OrderedShapleyAttribution": OrderedShapleyAttribution,
        "MonteCarloShapleyAttribution": MonteCarloShapleyAttribution,
        "PathShapleyAttribution": PathShapleyAttribution,
        "FirstTouchAttribution": FirstTouchAttribution,
        "LastTouchAttribution": LastTouchAttribution,
        "LinearAttribution": LinearAttribution,
        "TimeDecayAttribution": TimeDecayAttribution,
        "PositionBasedAttribution": PositionBasedAttribution,
    }

    cls = registry.get(class_name)
    if cls is None:
        raise ValueError(
            f"Unknown model class '{class_name}'. "
            f"Known classes: {list(registry.keys())}"
        )

    # Filter params to only those the constructor accepts
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in params.items() if k in valid_keys}
    return cls(**filtered)
