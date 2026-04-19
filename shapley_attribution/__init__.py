"""
Shapley Attribution: Multi-touch attribution modeling using Shapley values.

A scikit-learn compatible library implementing Shapley value methods for
attribution modeling in online advertising, based on Zhao et al. (2018).

Provides exact and approximate Shapley attribution models, heuristic
baselines, synthetic data generation, and evaluation metrics.
"""

from shapley_attribution.base import BaseAttributionModel
from shapley_attribution.models import (
    SimplifiedShapleyAttribution,
    OrderedShapleyAttribution,
    MonteCarloShapleyAttribution,
    PathShapleyAttribution,
)
from shapley_attribution.baselines import (
    FirstTouchAttribution,
    LastTouchAttribution,
    LinearAttribution,
    TimeDecayAttribution,
    PositionBasedAttribution,
)
from shapley_attribution.datasets import make_attribution_problem
from shapley_attribution.metrics import (
    normalized_mean_absolute_error,
    rank_correlation,
    top_k_overlap,
    attribution_summary,
)
from shapley_attribution.visualization import (
    plot_attribution,
    compare_models,
    plot_performance,
    plot_journey,
    plot_journeys_heatmap,
    plot_position_attribution,
)
from shapley_attribution.onnx import save_onnx, load_onnx

__version__ = "2.0.0"

__all__ = [
    # Models
    "BaseAttributionModel",
    "SimplifiedShapleyAttribution",
    "OrderedShapleyAttribution",
    "MonteCarloShapleyAttribution",
    # Baselines
    "FirstTouchAttribution",
    "LastTouchAttribution",
    "LinearAttribution",
    "TimeDecayAttribution",
    "PositionBasedAttribution",
    # Data
    "make_attribution_problem",
    # Metrics
    "normalized_mean_absolute_error",
    "rank_correlation",
    "top_k_overlap",
    "attribution_summary",
    # Visualization
    "plot_attribution",
    "compare_models",
    "plot_performance",
    "plot_journey",
    "plot_journeys_heatmap",
    "plot_position_attribution",
    # Models (continued)
    "PathShapleyAttribution",
    # ONNX
    "save_onnx",
    "load_onnx",
]
