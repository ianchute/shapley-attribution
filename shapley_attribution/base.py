"""
Base classes for attribution models with scikit-learn compatible API.
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseAttributionModel(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for all attribution models.

    Follows the scikit-learn estimator contract:
      - ``fit(X)`` learns from journey data.
      - ``transform(X)`` returns per-journey attribution matrices.
      - ``fit_transform(X)`` convenience composition.
      - ``get_attribution()`` returns aggregate channel-level scores after fit.

    Parameters
    ----------
    normalize : bool, default=True
        If True, attribution scores for each journey sum to 1.0.

    Attributes
    ----------
    channels_ : ndarray of shape (n_channels,)
        Unique channels discovered during ``fit``.
    attribution_ : dict[int | str, float]
        Aggregate attribution score per channel (populated after ``fit``).
    is_fitted_ : bool
        Whether the model has been fitted.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def fit(self, X, y=None):
        """Fit the attribution model on journey data.

        Parameters
        ----------
        X : list of list of int/str
            Customer journeys.  Each journey is an ordered list of
            channel identifiers (ints or strings).
        y : array-like of shape (n_journeys,) or None
            Binary conversion labels (1=converted, 0=not).
            Some models (e.g., MonteCarloShapleyAttribution) use this
            to learn a conversion model.  Heuristic baselines ignore it.
            If None, all journeys are assumed to be converting.

        Returns
        -------
        self
        """
        X = self._validate_journeys(X)
        self.channels_ = np.array(sorted({ch for journey in X for ch in journey}))
        self.channel_to_idx_ = {ch: i for i, ch in enumerate(self.channels_)}

        # Store conversion labels for subclasses that need them
        if y is not None:
            self.conversions_ = np.asarray(y, dtype=int)
        else:
            self.conversions_ = np.ones(len(X), dtype=int)

        self.attribution_ = self._compute_attribution(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Return per-journey attribution matrix.

        Parameters
        ----------
        X : list of list of int/str
            Customer journeys.

        Returns
        -------
        result : ndarray of shape (n_journeys, n_channels)
            Attribution weight for each channel in each journey.
        """
        self._check_is_fitted()
        X = self._validate_journeys(X)
        n_channels = len(self.channels_)
        result = np.zeros((len(X), n_channels))
        for i, journey in enumerate(X):
            scores = self._attribute_single(journey)
            for ch, score in scores.items():
                if ch in self.channel_to_idx_:
                    result[i, self.channel_to_idx_[ch]] = score
            if self.normalize and result[i].sum() > 0:
                result[i] /= result[i].sum()
        return result

    def get_attribution(self):
        """Return aggregate attribution scores as a dict.

        Returns
        -------
        attribution : dict
            Mapping from channel identifier to aggregate attribution score.
        """
        self._check_is_fitted()
        return dict(self.attribution_)

    def get_attribution_array(self):
        """Return aggregate attribution as a numpy array aligned with ``channels_``.

        Returns
        -------
        scores : ndarray of shape (n_channels,)
        """
        self._check_is_fitted()
        scores = np.array([self.attribution_.get(ch, 0.0) for ch in self.channels_])
        if self.normalize and scores.sum() > 0:
            scores = scores / scores.sum()
        return scores

    # ------------------------------------------------------------------
    # Abstract interface for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_attribution(self, X):
        """Compute aggregate attribution scores.

        Parameters
        ----------
        X : list of list
            Validated journey data.

        Returns
        -------
        attribution : dict
            Channel → aggregate score.
        """

    def _attribute_single(self, journey):
        """Compute attribution for a single journey.

        Default implementation distributes the aggregate scores proportional
        to channel presence.  Subclasses may override for journey-level
        attribution.
        """
        channels_in_journey = set(journey)
        present = {ch: self.attribution_.get(ch, 0.0) for ch in channels_in_journey}
        total = sum(present.values())
        if total == 0:
            n = len(channels_in_journey)
            return {ch: 1.0 / n for ch in channels_in_journey} if n > 0 else {}
        return {ch: v / total for ch, v in present.items()}

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_journeys(X):
        """Validate and coerce journey data."""
        if not hasattr(X, "__iter__"):
            raise ValueError("X must be an iterable of journeys.")
        journeys = []
        for i, journey in enumerate(X):
            if not hasattr(journey, "__iter__"):
                raise ValueError(f"Journey at index {i} is not iterable.")
            journey = list(journey)
            if len(journey) == 0:
                raise ValueError(f"Journey at index {i} is empty.")
            journeys.append(journey)
        if len(journeys) == 0:
            raise ValueError("X must contain at least one journey.")
        return journeys

    # ------------------------------------------------------------------
    # Visualization helpers (require matplotlib)
    # ------------------------------------------------------------------

    def plot_attribution(self, ax=None, title=None, top_k=None, ground_truth=None):
        """Horizontal bar chart of channel attribution scores.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Target axes; a new figure is created when None.
        title : str or None
            Plot title.  Defaults to the model class name.
        top_k : int or None
            Show only the top-k channels by score.
        ground_truth : ndarray or None
            If provided, overlays ground-truth markers (◆) on the bars.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from shapley_attribution.visualization import plot_attribution
        return plot_attribution(self, ax=ax, title=title,
                                top_k=top_k, ground_truth=ground_truth)

    def plot_journey(self, journey, converted=True, ax=None, title=None):
        """Sequence diagram for a single customer journey.

        Touchpoints are rendered as labelled boxes connected by arrows.
        Box colour encodes this model's attribution weight for each channel.

        Parameters
        ----------
        journey : list
            Ordered list of channel identifiers.
        converted : bool
            Whether the journey ended in a conversion.
        ax : matplotlib.axes.Axes or None
        title : str or None

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from shapley_attribution.visualization import plot_journey
        return plot_journey(journey, model=self, converted=converted,
                            ax=ax, title=title)

    def plot_journeys_heatmap(self, journeys, conversions=None, ax=None,
                              title=None, max_journeys=60):
        """Heatmap of the per-journey attribution matrix.

        Parameters
        ----------
        journeys : list of list
            Journey data to transform.
        conversions : array-like or None
            If provided, only converting journeys are shown.
        ax : matplotlib.axes.Axes or None
        title : str or None
        max_journeys : int
            Maximum rows to render.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        from shapley_attribution.visualization import plot_journeys_heatmap
        return plot_journeys_heatmap(self, journeys, conversions=conversions,
                                     ax=ax,
                                     title=title or f"{type(self).__name__} — Journey Heatmap",
                                     max_journeys=max_journeys)

    # ------------------------------------------------------------------
    # ONNX serialization
    # ------------------------------------------------------------------

    def save_onnx(self, path) -> None:
        """Serialize this fitted model to an ONNX file.

        GBM-backed models (``MonteCarloShapleyAttribution``,
        ``PathShapleyAttribution``) embed their learned value function as an
        ONNX graph so it can be queried directly via ``onnxruntime``.  All
        other models store their attribution state in ONNX ``metadata_props``.

        Parameters
        ----------
        path : str or Path
            Destination ``.onnx`` file path.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ImportError
            If ``onnx`` / ``skl2onnx`` are not installed.
            Install with: ``pip install onnx skl2onnx onnxruntime``

        Examples
        --------
        >>> model.save_onnx("my_model.onnx")
        >>> loaded = BaseAttributionModel.load_onnx("my_model.onnx")
        >>> loaded.get_attribution_array()
        """
        from shapley_attribution.onnx import save_onnx
        save_onnx(self, path)

    @staticmethod
    def load_onnx(path) -> "BaseAttributionModel":
        """Load a fitted attribution model from an ONNX file.

        Parameters
        ----------
        path : str or Path
            Path to an ``.onnx`` file created by :meth:`save_onnx`.

        Returns
        -------
        model : BaseAttributionModel
            Fully usable fitted model.

        Raises
        ------
        ImportError
            If ``onnx`` is not installed.
        ValueError
            If the file is not a valid shapley-attribution ONNX model.

        Examples
        --------
        >>> loaded = BaseAttributionModel.load_onnx("my_model.onnx")
        >>> scores = loaded.get_attribution()
        """
        from shapley_attribution.onnx import load_onnx
        return load_onnx(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet. "
                "Call 'fit' before using this method."
            )
