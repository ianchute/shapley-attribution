"""Tests for attribution models."""

import os
import tempfile

import numpy as np
import pytest

from shapley_attribution import (
    SimplifiedShapleyAttribution,
    MonteCarloShapleyAttribution,
    PathShapleyAttribution,
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    TimeDecayAttribution,
    PositionBasedAttribution,
    save_onnx,
    load_onnx,
)
from shapley_attribution.base import BaseAttributionModel
from shapley_attribution.datasets import make_attribution_problem


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def simple_journeys():
    return [[1, 2, 3], [1, 2], [2, 3], [1], [3, 1, 2]]


@pytest.fixture
def simple_conversions():
    return np.array([1, 1, 0, 1, 0])


@pytest.fixture
def synthetic_data():
    return make_attribution_problem(
        n_channels=5, n_journeys=500, random_state=42
    )


ALL_MODELS = [
    SimplifiedShapleyAttribution,
    MonteCarloShapleyAttribution,
    PathShapleyAttribution,
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    TimeDecayAttribution,
    PositionBasedAttribution,
]


# ---------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------

class TestSklearnAPI:
    """Verify scikit-learn estimator contract."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_fit_returns_self(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass()
        result = model.fit(simple_journeys, y=simple_conversions)
        assert result is model

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_fit_without_y(self, ModelClass, simple_journeys):
        """All models should work without y (legacy mode)."""
        model = ModelClass()
        result = model.fit(simple_journeys)
        assert result is model

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_get_params(self, ModelClass):
        model = ModelClass()
        params = model.get_params()
        assert isinstance(params, dict)
        assert "normalize" in params

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_set_params(self, ModelClass):
        model = ModelClass()
        model.set_params(normalize=False)
        assert model.normalize is False

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_clone(self, ModelClass):
        from sklearn.base import clone
        model = ModelClass()
        cloned = clone(model)
        assert type(cloned) is type(model)
        assert cloned.get_params() == model.get_params()

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_not_fitted_raises(self, ModelClass):
        model = ModelClass()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_attribution()

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_transform_shape(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass()
        model.fit(simple_journeys, y=simple_conversions)
        result = model.transform(simple_journeys)
        assert result.shape == (len(simple_journeys), len(model.channels_))

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_fit_transform(self, ModelClass, simple_journeys):
        model = ModelClass()
        result = model.fit_transform(simple_journeys)
        assert result.shape[0] == len(simple_journeys)


# ---------------------------------------------------------------
# Attribution correctness
# ---------------------------------------------------------------

class TestAttributionCorrectness:
    """Test that attribution scores are sensible."""

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_scores_non_negative(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass()
        model.fit(simple_journeys, y=simple_conversions)
        scores = model.get_attribution_array()
        assert np.all(scores >= 0)

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_normalized_sums_to_one(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass(normalize=True)
        model.fit(simple_journeys, y=simple_conversions)
        scores = model.get_attribution_array()
        np.testing.assert_almost_equal(scores.sum(), 1.0, decimal=5)

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_all_channels_discovered(self, ModelClass, simple_journeys, simple_conversions):
        model = ModelClass()
        model.fit(simple_journeys, y=simple_conversions)
        channels = {ch for j in simple_journeys for ch in j}
        assert set(model.channels_) == channels

    def test_first_touch_assigns_to_first(self):
        journeys = [[1, 2, 3], [2, 3, 1]]
        y = [1, 1]
        model = FirstTouchAttribution(normalize=False)
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        assert attr[1] == 1.0  # first in journey 1
        assert attr[2] == 1.0  # first in journey 2

    def test_last_touch_assigns_to_last(self):
        journeys = [[1, 2, 3], [2, 3, 1]]
        y = [1, 1]
        model = LastTouchAttribution(normalize=False)
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        assert attr[3] == 1.0  # last in journey 1
        assert attr[1] == 1.0  # last in journey 2

    def test_linear_equal_split(self):
        journeys = [[1, 2]]
        y = [1]
        model = LinearAttribution(normalize=False)
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        np.testing.assert_almost_equal(attr[1], 0.5)
        np.testing.assert_almost_equal(attr[2], 0.5)

    def test_non_converting_gets_no_credit(self):
        """Non-converting journeys should not receive attribution credit."""
        journeys = [[1, 2], [3, 4]]
        y = [1, 0]  # Only first journey converts
        model = LinearAttribution(normalize=False)
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        # Channels 3 and 4 only appear in non-converting journey
        assert attr.get(3, 0.0) == 0.0
        assert attr.get(4, 0.0) == 0.0

    def test_simplified_shapley_single_channel(self):
        journeys = [[1], [1], [1]]
        model = SimplifiedShapleyAttribution()
        model.fit(journeys)
        attr = model.get_attribution()
        assert 1 in attr
        assert attr[1] > 0


# ---------------------------------------------------------------
# Monte Carlo convergence
# ---------------------------------------------------------------

class TestMonteCarlo:
    def test_reproducibility(self, simple_journeys, simple_conversions):
        m1 = MonteCarloShapleyAttribution(n_iter=200, random_state=42)
        m2 = MonteCarloShapleyAttribution(n_iter=200, random_state=42)
        m1.fit(simple_journeys, y=simple_conversions)
        m2.fit(simple_journeys, y=simple_conversions)
        np.testing.assert_array_almost_equal(
            m1.get_attribution_array(),
            m2.get_attribution_array(),
        )

    def test_uses_conversion_labels(self):
        """MC Shapley should give different results with different labels."""
        journeys = [[1, 2], [2, 3], [1, 3], [1, 2, 3]] * 50
        y1 = ([1, 0, 0, 1]) * 50  # ch 1+2 together convert
        y2 = ([0, 1, 0, 1]) * 50  # ch 2+3 together convert

        m1 = MonteCarloShapleyAttribution(n_iter=500, random_state=42)
        m2 = MonteCarloShapleyAttribution(n_iter=500, random_state=42)
        m1.fit(journeys, y=y1)
        m2.fit(journeys, y=y2)

        s1 = m1.get_attribution_array()
        s2 = m2.get_attribution_array()
        # The two should differ meaningfully
        assert not np.allclose(s1, s2, atol=0.05)

    def test_mc_beats_heuristics_on_synthetic(self):
        """MC Shapley should recover true importance better than heuristics."""
        journeys, conv, gt, _ = make_attribution_problem(
            n_channels=8, n_journeys=5000, random_state=42
        )

        mc = MonteCarloShapleyAttribution(n_iter=2000, random_state=42)
        mc.fit(journeys, y=conv)

        linear = LinearAttribution()
        linear.fit(journeys, y=conv)

        first = FirstTouchAttribution()
        first.fit(journeys, y=conv)

        from shapley_attribution.metrics import normalized_mean_absolute_error

        mc_nmae = normalized_mean_absolute_error(gt, mc.get_attribution_array())
        lin_nmae = normalized_mean_absolute_error(gt, linear.get_attribution_array())
        first_nmae = normalized_mean_absolute_error(gt, first.get_attribution_array())

        # MC Shapley should have lower NMAE (closer to ground truth)
        # than both heuristics
        assert mc_nmae < lin_nmae, (
            f"MC NMAE {mc_nmae:.4f} should be < Linear NMAE {lin_nmae:.4f}"
        )
        assert mc_nmae < first_nmae, (
            f"MC NMAE {mc_nmae:.4f} should be < First Touch NMAE {first_nmae:.4f}"
        )


# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_empty_journeys_raises(self, ModelClass):
        model = ModelClass()
        with pytest.raises(ValueError):
            model.fit([])

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_empty_single_journey_raises(self, ModelClass):
        model = ModelClass()
        with pytest.raises(ValueError):
            model.fit([[]])

    @pytest.mark.parametrize("ModelClass", ALL_MODELS)
    def test_string_channels(self, ModelClass):
        journeys = [["email", "social"], ["social", "search"], ["email"]]
        y = [1, 0, 1]
        model = ModelClass()
        model.fit(journeys, y=y)
        attr = model.get_attribution()
        assert "email" in attr or "social" in attr


# ---------------------------------------------------------------
# Dataset generator
# ---------------------------------------------------------------

class TestDatasetGenerator:
    def test_output_shape(self):
        journeys, conv, gt, channels = make_attribution_problem(
            n_channels=6, n_journeys=100, random_state=42
        )
        assert len(journeys) == 100
        assert len(conv) == 100
        assert len(gt) == 6
        assert len(channels) == 6

    def test_has_both_classes(self):
        _, conv, _, _ = make_attribution_problem(
            n_channels=5, n_journeys=1000, random_state=42
        )
        assert conv.sum() > 0
        assert (1 - conv).sum() > 0

    def test_ground_truth_sums_to_one(self):
        _, _, gt, _ = make_attribution_problem(n_channels=5, random_state=42)
        np.testing.assert_almost_equal(gt.sum(), 1.0, decimal=5)

    def test_reproducibility(self):
        j1, c1, gt1, _ = make_attribution_problem(random_state=42)
        j2, c2, gt2, _ = make_attribution_problem(random_state=42)
        np.testing.assert_array_equal(gt1, gt2)
        np.testing.assert_array_equal(c1, c2)
        assert j1 == j2

    def test_directed_interactions_return_ordered_gt(self):
        result = make_attribution_problem(
            n_channels=5, n_journeys=500,
            directed_interaction_strength=0.5,
            random_state=42,
            return_ordered_ground_truth=True,
        )
        assert len(result) == 5
        journeys, conv, gt, channels, ordered_gt = result
        assert len(ordered_gt) == 5
        np.testing.assert_almost_equal(ordered_gt.sum(), 1.0, decimal=5)

    def test_no_directed_interactions_gt_matches(self):
        """Without directed effects the 4-tuple API is unchanged."""
        result = make_attribution_problem(
            n_channels=5, n_journeys=500, random_state=42
        )
        assert len(result) == 4


# ---------------------------------------------------------------
# PathShapleyAttribution specific tests
# ---------------------------------------------------------------

class TestPathShapley:

    @pytest.fixture
    def path_data(self):
        """Synthetic data with both classes and mixed-length journeys."""
        rng = np.random.RandomState(7)
        journeys = [list(rng.choice([0, 1, 2, 3], size=rng.randint(2, 5)))
                    for _ in range(200)]
        conversions = np.array([rng.randint(0, 2) for _ in range(200)])
        # Ensure at least one of each class
        conversions[0] = 1
        conversions[1] = 0
        return journeys, conversions

    def test_position_attribution_populated(self, path_data):
        journeys, conv = path_data
        model = PathShapleyAttribution(random_state=0).fit(journeys, y=conv)
        assert hasattr(model, "position_attribution_")
        assert len(model.position_attribution_) == len(model.channels_)
        # Each channel should have a list with at least one entry
        for ch, pos_scores in model.position_attribution_.items():
            assert isinstance(pos_scores, list)

    def test_ordering_sensitivity(self):
        """Reversed journey order should give different per-journey attribution."""
        # Enough data for GBM to have both classes
        journeys = [[0, 1, 2]] * 40 + [[2, 1, 0]] * 20 + [[0]] * 20 + [[2]] * 20
        conv     = [1] * 40 + [0] * 20 + [1] * 20 + [0] * 20

        model = PathShapleyAttribution(random_state=42).fit(journeys, y=conv)

        fwd = model._attribute_single([0, 1, 2])  # ch0 first
        rev = model._attribute_single([2, 1, 0])  # ch2 first

        # Channel 0 credit differs between orderings
        assert not np.isclose(fwd.get(0, 0.0), rev.get(0, 0.0), atol=1e-6), (
            "PathShapley should assign different credit to ch0 depending on "
            "whether it appears first ([0,1,2]) or last ([2,1,0])"
        )

    def test_path_differs_from_mc(self):
        """PathShapley and MCShapley should produce different attributions."""
        journeys, conv, _, _ = make_attribution_problem(
            n_channels=6, n_journeys=500, random_state=99
        )
        path = PathShapleyAttribution(random_state=0).fit(journeys, y=conv)
        mc   = MonteCarloShapleyAttribution(n_iter=300, random_state=0).fit(journeys, y=conv)

        arr_path = path.get_attribution_array()
        arr_mc   = mc.get_attribution_array()
        # They should differ (different sampling strategy)
        assert not np.allclose(arr_path, arr_mc, atol=1e-3)

    def test_path_beats_heuristics_on_directed_data(self):
        """PathShapley should outperform Linear on ordered-ground-truth
        when directed interactions are present."""
        journeys, conv, _, _, ordered_gt = make_attribution_problem(
            n_channels=6, n_journeys=3000,
            directed_interaction_strength=0.6,
            random_state=42,
            return_ordered_ground_truth=True,
        )

        path   = PathShapleyAttribution(random_state=42).fit(journeys, y=conv)
        linear = LinearAttribution().fit(journeys, y=conv)

        from shapley_attribution.metrics import normalized_mean_absolute_error
        path_nmae   = normalized_mean_absolute_error(ordered_gt, path.get_attribution_array())
        linear_nmae = normalized_mean_absolute_error(ordered_gt, linear.get_attribution_array())

        assert path_nmae < linear_nmae, (
            f"PathShapley NMAE {path_nmae:.4f} should beat Linear {linear_nmae:.4f} "
            "on directed data evaluated against ordered ground truth"
        )


# ---------------------------------------------------------------
# ONNX serialization
# ---------------------------------------------------------------

# All models that should survive an ONNX round-trip
ONNX_MODELS = [
    SimplifiedShapleyAttribution,
    MonteCarloShapleyAttribution,
    PathShapleyAttribution,
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    TimeDecayAttribution,
    PositionBasedAttribution,
]

onnx = pytest.importorskip("onnx", reason="onnx not installed")


class TestONNX:
    """ONNX save/load round-trip tests for all attribution models."""

    @pytest.fixture
    def onnx_journeys(self):
        return [[1, 2, 3], [1, 2], [2, 3], [1], [3, 1, 2]]

    @pytest.fixture
    def onnx_conversions(self):
        return np.array([1, 1, 0, 1, 0])

    @pytest.fixture
    def onnx_path(self):
        """Yield a temporary .onnx file path and clean up after the test."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    # ------------------------------------------------------------------
    # Round-trip: attribution scores must survive save → load
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ModelClass", ONNX_MODELS)
    def test_attribution_array_round_trip(
        self, ModelClass, onnx_journeys, onnx_conversions, onnx_path
    ):
        """save_onnx → load_onnx preserves get_attribution_array()."""
        model = ModelClass().fit(onnx_journeys, y=onnx_conversions)
        save_onnx(model, onnx_path)
        loaded = load_onnx(onnx_path)

        np.testing.assert_allclose(
            model.get_attribution_array(),
            loaded.get_attribution_array(),
            atol=1e-5,
            err_msg=f"{ModelClass.__name__}: attribution array mismatch after ONNX round-trip",
        )

    @pytest.mark.parametrize("ModelClass", ONNX_MODELS)
    def test_channels_round_trip(
        self, ModelClass, onnx_journeys, onnx_conversions, onnx_path
    ):
        """channels_ must be identical after round-trip."""
        model = ModelClass().fit(onnx_journeys, y=onnx_conversions)
        save_onnx(model, onnx_path)
        loaded = load_onnx(onnx_path)

        np.testing.assert_array_equal(
            model.channels_,
            loaded.channels_,
            err_msg=f"{ModelClass.__name__}: channels_ mismatch after ONNX round-trip",
        )

    @pytest.mark.parametrize("ModelClass", ONNX_MODELS)
    def test_is_fitted_after_load(
        self, ModelClass, onnx_journeys, onnx_conversions, onnx_path
    ):
        """Loaded model must be marked as fitted."""
        model = ModelClass().fit(onnx_journeys, y=onnx_conversions)
        save_onnx(model, onnx_path)
        loaded = load_onnx(onnx_path)
        assert loaded.is_fitted_

    @pytest.mark.parametrize("ModelClass", ONNX_MODELS)
    def test_get_attribution_usable_after_load(
        self, ModelClass, onnx_journeys, onnx_conversions, onnx_path
    ):
        """get_attribution() returns a dict with the correct keys."""
        model = ModelClass().fit(onnx_journeys, y=onnx_conversions)
        save_onnx(model, onnx_path)
        loaded = load_onnx(onnx_path)

        attr = loaded.get_attribution()
        assert isinstance(attr, dict)
        assert set(attr.keys()) == set(model.get_attribution().keys())

    # ------------------------------------------------------------------
    # Instance method API (save_onnx / load_onnx on the model)
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ModelClass", ONNX_MODELS)
    def test_instance_method_api(
        self, ModelClass, onnx_journeys, onnx_conversions, onnx_path
    ):
        """model.save_onnx() / BaseAttributionModel.load_onnx() API works."""
        model = ModelClass().fit(onnx_journeys, y=onnx_conversions)
        model.save_onnx(onnx_path)
        loaded = BaseAttributionModel.load_onnx(onnx_path)

        np.testing.assert_allclose(
            model.get_attribution_array(),
            loaded.get_attribution_array(),
            atol=1e-5,
        )

    # ------------------------------------------------------------------
    # String channels
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ModelClass", ONNX_MODELS)
    def test_string_channels_round_trip(self, ModelClass, onnx_path):
        """String channel identifiers are preserved through ONNX."""
        journeys = [["email", "social"], ["social", "search"], ["email"]]
        y = [1, 0, 1]
        model = ModelClass().fit(journeys, y=y)
        save_onnx(model, onnx_path)
        loaded = load_onnx(onnx_path)

        assert set(loaded.channels_) == set(model.channels_)
        np.testing.assert_allclose(
            model.get_attribution_array(),
            loaded.get_attribution_array(),
            atol=1e-5,
        )

    # ------------------------------------------------------------------
    # GBM models: onnxruntime direct inference
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("ModelClass", [MonteCarloShapleyAttribution, PathShapleyAttribution])
    def test_onnxruntime_inference(
        self, ModelClass, onnx_journeys, onnx_conversions, onnx_path
    ):
        """For GBM-backed models, the ONNX file can be run via onnxruntime."""
        ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

        model = ModelClass().fit(onnx_journeys, y=onnx_conversions)
        save_onnx(model, onnx_path)

        sess = ort.InferenceSession(onnx_path)
        n_channels = len(model.channels_)
        mask = np.ones((1, n_channels), dtype=np.float32)
        outputs = sess.run(None, {"input": mask})

        # Binary classifier: second output is probability array (n, 2)
        proba = outputs[1]
        assert proba.shape == (1, 2), f"Expected (1, 2) proba, got {proba.shape}"
        assert 0.0 <= proba[0, 1] <= 1.0, "P(conversion) must be in [0, 1]"

    @pytest.mark.parametrize("ModelClass", [MonteCarloShapleyAttribution, PathShapleyAttribution])
    def test_loaded_model_transform_works(
        self, ModelClass, onnx_journeys, onnx_conversions, onnx_path
    ):
        """transform() on the loaded GBM model uses the ORT wrapper correctly."""
        model = ModelClass().fit(onnx_journeys, y=onnx_conversions)
        save_onnx(model, onnx_path)
        loaded = load_onnx(onnx_path)

        result = loaded.transform(onnx_journeys)
        assert result.shape == (len(onnx_journeys), len(model.channels_))
        assert np.all(result >= 0)

    # ------------------------------------------------------------------
    # position_attribution_ preserved (PathShapley)
    # ------------------------------------------------------------------

    def test_position_attribution_round_trip(
        self, onnx_journeys, onnx_conversions, onnx_path
    ):
        """PathShapley position_attribution_ is preserved through ONNX."""
        model = PathShapleyAttribution().fit(onnx_journeys, y=onnx_conversions)
        save_onnx(model, onnx_path)
        loaded = load_onnx(onnx_path)

        assert hasattr(loaded, "position_attribution_")
        assert set(loaded.position_attribution_.keys()) == set(
            model.position_attribution_.keys()
        )
        for ch in model.channels_:
            np.testing.assert_allclose(
                model.position_attribution_[ch],
                loaded.position_attribution_[ch],
                atol=1e-5,
                err_msg=f"position_attribution_[{ch}] mismatch after ONNX round-trip",
            )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_load_invalid_onnx_raises(self, onnx_path):
        """Loading an ONNX file without sa_model_class metadata raises ValueError."""
        import onnx as onnx_lib
        from onnx import helper, TensorProto

        # Build a valid ONNX model but without shapley metadata
        input_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        output_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
        graph = helper.make_graph([node], "bare_graph", [input_info], [output_info])
        proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        proto.ir_version = 8
        onnx_lib.save(proto, onnx_path)

        with pytest.raises(ValueError, match="sa_model_class"):
            load_onnx(onnx_path)

    @pytest.mark.parametrize("ModelClass", ONNX_MODELS)
    def test_save_unfitted_raises(self, ModelClass, onnx_path):
        """Saving an unfitted model raises RuntimeError."""
        model = ModelClass()
        with pytest.raises(RuntimeError, match="not fitted"):
            save_onnx(model, onnx_path)
