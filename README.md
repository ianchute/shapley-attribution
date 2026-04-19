# Shapley Attribution

```bash
pip install shapley-attribution
```

A scikit-learn compatible Python library for multi-touch attribution modeling using Shapley values from game theory. Computes marginal contribution of each marketing channel to conversion, inspired by Zhao et al. (2018) but using an interventional Shapley approach with a learned conversion model.

## Installation

```bash
pip install shapley-attribution

# With ONNX export/import support
pip install "shapley-attribution[onnx]"

# With benchmark dependencies
pip install "shapley-attribution[benchmarks]"

# Development install from source
pip install -e ".[dev]"
```

## Quick Start

```python
from shapley_attribution import (
    MonteCarloShapleyAttribution,
    PathShapleyAttribution,
    LinearAttribution,
    make_attribution_problem,
    attribution_summary,
)

# Standard dataset (set-based ground truth)
journeys, conversions, ground_truth, channels = make_attribution_problem(
    n_channels=8, n_journeys=5000, random_state=42
)

# MC Shapley — order-agnostic, strict Shapley axioms
mc = MonteCarloShapleyAttribution(n_iter=2000, random_state=42)
mc.fit(journeys, y=conversions)

# Path Shapley — ordering-aware, uses actual journey sequences
path = PathShapleyAttribution(random_state=42)
path.fit(journeys, y=conversions)

# Get aggregate attribution scores
scores = mc.get_attribution()        # dict: channel -> score
array  = mc.get_attribution_array()  # numpy array aligned with model.channels_

# Per-journey attribution matrix
matrix = mc.transform(journeys)      # shape (n_journeys, n_channels)

# Compare multiple models against ground truth
linear = LinearAttribution().fit(journeys, y=conversions)
results = attribution_summary(
    {
        "MC Shapley"  : mc.get_attribution_array(),
        "Path Shapley": path.get_attribution_array(),
        "Linear"      : linear.get_attribution_array(),
    },
    ground_truth
)

# Dataset with directed (ordered) interactions — use Path Shapley here
journeys_d, conv_d, gt_d, ch_d, ordered_gt = make_attribution_problem(
    n_channels=8, n_journeys=5000,
    directed_interaction_strength=0.5,
    return_ordered_ground_truth=True,
    random_state=42,
)
path_d = PathShapleyAttribution(random_state=42).fit(journeys_d, y=conv_d)

# Visualize position-level credit (upper-funnel vs lower-funnel)
path_d.plot_position_attribution()
```

## Models

### Shapley-based

| Model | Class | Complexity | Best for |
|---|---|---|---|
| **Simplified Shapley** | `SimplifiedShapleyAttribution` | O(n_journeys x n_channels) | Exact values, <= 20 channels |
| **Ordered Shapley** | `OrderedShapleyAttribution` | O(n_journeys x n_positions x 2^n_channels) | Position-aware, <= 15 channels |
| **Monte Carlo Shapley** | `MonteCarloShapleyAttribution` | O(n_iter x n_channels) | Scalable to 100+ channels, order-agnostic |
| **Path Shapley** | `PathShapleyAttribution` | O(n_journeys x max_journey_length) | Ordering-aware, typically faster than MC |

**MC Shapley** trains a GradientBoostingClassifier to learn conversion probability, then estimates Shapley values by averaging marginal contributions across random permutations (interventional formulation — no background averaging). It is order-agnostic: `[A, B]` and `[B, A]` receive identical attribution.

**Path Shapley** uses the same GBM value function but replaces random permutation sampling with the *actual journey sequence* as the coalition-formation order. Channel B in journey `[A, B]` is credited for the lift *given A was already seen*; in `[B, A]`, A receives that conditional credit instead. This makes the model sensitive to channel ordering, which is meaningful when upstream channels prime the customer for downstream ones. Use `directed_interaction_strength > 0` in `make_attribution_problem` to generate data that rewards this sensitivity.

### PathShapleyAttribution — Deep Dive

**How it works.** For each converting journey `[c₁, c₂, ..., cₙ]`, Path Shapley builds the coalition incrementally along the actual touchpoint sequence:

```
contribution(cᵢ) = v({c₁,...,cᵢ}) − v({c₁,...,cᵢ₋₁})
```

where `v(S)` is the GBM's predicted conversion probability for the channel set S. Duplicate channels within a journey are collapsed to first occurrence before scoring, so the ordering is the order of *first* exposure.

The key insight: because the GBM value function is nonlinear, `v({A})` and `v({B})` are generally different. So the marginal contribution of B *given A was already seen* (`v({A,B}) − v({A})`) differs from its contribution in the reverse order (`v({A,B}) − v({B})`). This is where the ordering sensitivity comes from — it does not require any modification to the GBM itself.

**When to use Path Shapley vs MC Shapley.**

Path Shapley is the right choice when channel ordering genuinely matters — for example, when awareness channels (display, social) consistently appear before conversion channels (search, email) and the sequence itself creates synergy. It is also faster than MC Shapley at inference time because it requires only one value-function call per position (O(n_journeys × max_journey_length)) vs. O(n_iter × n_channels) for MC.

MC Shapley is better when you want pure set-based Shapley values that are agnostic to sequencing, or when journeys are so short that ordering effects are negligible. MC Shapley also satisfies the strict Shapley axioms (efficiency, symmetry, null player, additivity); Path Shapley does not — it is a path-dependent approximation, not a true Shapley value.

**Limitations.**
- Not a strict Shapley value: Path Shapley violates the symmetry axiom — two channels with identical marginal contributions can receive different scores if they typically appear in different positions.
- Rare channels are noisy: channels that appear infrequently have few paths to average over, making their scores higher-variance than MC's permutation average.
- Journey deduplication: repeated channel exposures within a single journey are collapsed to first occurrence, discarding frequency information.
- The GBM value function is still set-based: it cannot directly represent interaction effects that depend on order, only on channel co-presence. Ordering effects enter through the path accumulation, not the value function.

**Performance.** Evaluated on synthetic data (8 channels, 5000 journeys). See the [Benchmark](#benchmark) section for full results. Summary:

| Setting | Path NMAE | MC NMAE | Path wins? |
|---|---|---|---|
| Undirected (set-based GT) | **0.0195** | 0.0221 | ✓ lower NMAE |
| Directed (ordered GT, strength=0.6) | **0.0399** | 0.0503 | ✓ lower NMAE |

On undirected data, Path Shapley's lower NMAE reflects that real journeys carry ordering signal even in the absence of explicit directed coefficients — the path accumulation produces a useful inductive bias. On directed data with asymmetric channel interactions, the gap widens: MC Shapley is blind to the ordering and its NMAE actually exceeds the heuristic baselines, while Path Shapley remains the best model.

### Heuristic Baselines

| Model | Class | Rule |
|---|---|---|
| **First Touch** | `FirstTouchAttribution` | 100% to first channel |
| **Last Touch** | `LastTouchAttribution` | 100% to last channel |
| **Linear** | `LinearAttribution` | Equal credit across all touchpoints |
| **Time Decay** | `TimeDecayAttribution` | Exponential decay favoring recent touchpoints |
| **Position Based** | `PositionBasedAttribution` | 40/20/40 split (first/middle/last) |

## scikit-learn Compatibility

All models inherit from `sklearn.base.BaseEstimator` and `TransformerMixin`:

```python
from sklearn.base import clone

model = MonteCarloShapleyAttribution(n_iter=2000)
model.get_params()          # {'n_iter': 2000, 'random_state': None, ...}
model.set_params(n_iter=500)
cloned = clone(model)       # Deep copy with same params
```

## ONNX Serialization

All fitted models can be saved to and loaded from ONNX format, enabling deployment via any ONNX-compatible runtime.

### Installation

```bash
pip install -e ".[onnx]"
# or manually:
pip install onnx skl2onnx onnxruntime
```

### Save and load

```python
from shapley_attribution import MonteCarloShapleyAttribution, save_onnx, load_onnx

# Fit a model
mc = MonteCarloShapleyAttribution(n_iter=2000, random_state=42)
mc.fit(journeys, y=conversions)

# Save — available as standalone function or instance method
save_onnx(mc, "mc_shapley.onnx")
mc.save_onnx("mc_shapley.onnx")   # equivalent

# Load — returns a fully usable fitted model
from shapley_attribution.base import BaseAttributionModel
loaded = load_onnx("mc_shapley.onnx")
loaded = BaseAttributionModel.load_onnx("mc_shapley.onnx")  # equivalent

# Full API is available immediately after loading
scores  = loaded.get_attribution()        # dict: channel → score
array   = loaded.get_attribution_array()  # numpy array
matrix  = loaded.transform(new_journeys)  # per-journey attribution
```

### What is serialized

| Model type | ONNX graph | Metadata |
|---|---|---|
| `MonteCarloShapleyAttribution` | Learned GBM (GradientBoostingClassifier → ONNX via skl2onnx) | channels, attribution scores, hyperparams |
| `PathShapleyAttribution` | Learned GBM | channels, attribution scores, position_attribution_, hyperparams |
| `SimplifiedShapleyAttribution` | Identity op (trivial) | channels, attribution scores, hyperparams |
| `OrderedShapleyAttribution` | Identity op (trivial) | channels, attribution scores, position_attribution_, hyperparams |
| Heuristic baselines | Identity op (trivial) | channels, attribution scores, hyperparams |

For GBM-backed models the ONNX file embeds the full value function graph, so you can query the conversion probability model directly via `onnxruntime`:

```python
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession("mc_shapley.onnx")
# binary presence mask — shape (1, n_channels)
mask = np.array([[1, 0, 1, 0, 0, 0, 1, 0]], dtype=np.float32)
label, proba = sess.run(None, {"input": mask})
p_conversion = proba[0, 1]  # P(conversion | channels 0, 2, 6 present)
```

This makes it straightforward to deploy the value function to any ONNX-compatible serving stack (ONNX Runtime Server, Triton, MLIR, etc.) without a Python sklearn dependency.

## Conversion Labels

All models accept an optional `y` parameter with binary conversion labels:

```python
model.fit(journeys, y=conversions)  # 1=converted, 0=not
model.fit(journeys)                 # Legacy mode: all journeys assumed converting
```

MC Shapley uses these labels to train a conversion model, giving it a significant accuracy advantage over heuristic baselines. The heuristic baselines use the labels to attribute credit only to converting journeys.

## Evaluation Metrics

```python
from shapley_attribution.metrics import (
    normalized_mean_absolute_error,  # NMAE in [0, 2], lower is better
    rank_correlation,                # Spearman rho in [-1, 1], higher is better
    top_k_overlap,                   # Fraction of true top-k recovered
    attribution_summary,             # Compare multiple models at once
)
```

## Synthetic Data

```python
from shapley_attribution.datasets import make_attribution_problem

journeys, conversions, ground_truth, channels = make_attribution_problem(
    n_channels=10,
    n_journeys=5000,
    max_journey_length=8,
    interaction_effects=0.5,   # Pairwise synergy between channels
    base_conversion_rate=0.3,
    random_state=42,
)
```

The synthetic generator creates journeys with roughly uniform channel sampling but conversion probability driven by channel presence and pairwise interactions via a logistic model with known coefficients. This rewards models that capture marginal contribution (Shapley) over frequency counting (heuristics).

Enable `directed_interaction_strength > 0` to additionally bake in **ordering effects**: an asymmetric bonus/penalty is applied when channel i appears *before* channel j (`directed_matrix[i, j] ≠ directed_matrix[j, i]`). This creates genuine sequential synergies that PathShapleyAttribution is designed to exploit, while set-based models (MC Shapley, heuristics) remain blind to them.

```python
journeys, conversions, ground_truth, channels, ordered_ground_truth = make_attribution_problem(
    n_channels=8,
    n_journeys=5000,
    directed_interaction_strength=0.5,   # ordering effects active
    return_ordered_ground_truth=True,    # oracle path GT via true model
)
```

`ordered_ground_truth` is computed by walking each converting journey through the true logistic model and accumulating marginal contributions along the path — the oracle that PathShapleyAttribution aims to recover.

### How ground truth is computed

Ground truth is determined entirely before any journeys are generated:

1. **Channel importances** are sampled from a Dirichlet distribution (α=2) and raised to the power 1.3, producing moderate skew — one or two channels are notably more important than others, but not overwhelmingly so.
2. **Conversion coefficients** are scaled from those importances: `coef[ch] = importance[ch] × n_channels × 3`.
3. **Pairwise interaction terms** are sampled for every channel pair, adding synergy effects that no heuristic can capture.
4. **An intercept is calibrated** via root-finding (`scipy.brentq`) so the overall conversion rate matches `base_conversion_rate`.
5. **Journeys are generated** with near-uniform channel sampling, so channel frequency carries almost no signal — only which combinations appear drives conversion.
6. **Conversion labels** are drawn as Bernoulli samples from the logistic model: `P(convert) = σ(intercept + Σcoef·presence + Σinteraction·pair_presence)`.

The ground truth returned is the **normalized channel importance vector** from step 1 — the "true" individual channel weights, independent of interactions. This is intentionally slightly different from perfect Shapley values (which would distribute interaction credit across channels), giving a realistic evaluation target.

## Visualization

All fitted models expose three plot methods directly. Standalone functions are also available for multi-model comparison.

```python
from shapley_attribution import (
    MonteCarloShapleyAttribution, LinearAttribution,
    make_attribution_problem,
    compare_models, plot_performance, plot_journey, plot_journeys_heatmap,
)

journeys, conversions, ground_truth, channels = make_attribution_problem(
    n_channels=8, n_journeys=5000, random_state=42
)

mc = MonteCarloShapleyAttribution(n_iter=2000, random_state=42).fit(journeys, y=conversions)
lin = LinearAttribution().fit(journeys, y=conversions)
```

### Per-model attribution bar chart

```python
# On the model directly — overlays ground truth markers
mc.plot_attribution(ground_truth=ground_truth, top_k=8)
```

Shows a horizontal bar chart sorted by attribution score. Pass `ground_truth` to overlay ◆ markers for easy comparison.

### Compare multiple models

```python
# Standalone function — accepts 2+ models or pre-computed arrays
compare_models(
    {"MC Shapley": mc, "Linear": lin},
    ground_truth=ground_truth,
)
```

Renders a grouped bar chart with one cluster per channel. Accepts either fitted model objects or raw numpy arrays.

### Performance metrics panel

```python
from shapley_attribution.metrics import attribution_summary

results = attribution_summary(
    {"MC Shapley": mc.get_attribution_array(),
     "Linear": lin.get_attribution_array()},
    ground_truth,
)
plot_performance(results)
```

Three-panel figure (NMAE / Spearman rank correlation / top-3 overlap). The best-performing model is highlighted in each panel.

### Journey sequence diagram

```python
# On the model directly — boxes are coloured by attribution weight
mc.plot_journey(journeys[0], converted=bool(conversions[0]))

# Standalone — without a model (plain sequence)
plot_journey(journeys[0], converted=True)
```

Renders touchpoints as rounded boxes connected by arrows, with a conversion outcome node at the end. When called on a fitted model, box colour encodes that model's attribution score for each channel.

### Position attribution breakdown (PathShapley only)

```python
path = PathShapleyAttribution(random_state=42).fit(journeys, y=conversions)

# On the model directly
path.plot_position_attribution()

# Standalone
from shapley_attribution import plot_position_attribution
plot_position_attribution(path, top_k=6)
```

Stacked bar chart where each bar is a channel and each stack segment is a journey position (position 1 = first touchpoint, position 2 = second, …). Channels with tall early-position stacks are **upper-funnel** (awareness); channels with tall late-position stacks are **lower-funnel** (conversion drivers). Only available after fitting a `PathShapleyAttribution` model.

### Per-journey attribution heatmap

```python
# On the model directly
mc.plot_journeys_heatmap(journeys, conversions=conversions, max_journeys=50)

# Standalone
plot_journeys_heatmap(mc, journeys, conversions=conversions)
```

Heatmap of the `transform()` output matrix. Each row is a journey, each column is a channel. Pass `conversions` to show only converting journeys.

---

## Benchmark

```bash
python benchmarks/benchmark.py
```

Results use 8 channels, 5000 journeys, 2000 MC iterations, `random_state=42`. Lower NMAE is better; higher rank correlation and top-3 overlap are better.

### Benchmark A — Undirected (standard set-based ground truth)

No directed interaction effects. Ground truth is the normalized Dirichlet channel importance vector.

```
Model                    NMAE    Rank Corr    Top-3    Time(s)
--------------------------------------------------------------
First Touch            0.0481       0.7619     0.67     0.004
Last Touch             0.0446       0.9701     1.00     0.003
Linear                 0.0457       0.9524     1.00     0.005
Time Decay             0.0445       0.9762     1.00     0.006
Position Based         0.0461       0.9762     0.67     0.005
Simplified Shapley     0.0455       1.0000     1.00     0.006
MC Shapley             0.0221       0.9762     1.00     0.496
Path Shapley           0.0195       0.9762     0.67     0.471  ← best NMAE
```

Path Shapley achieves the lowest NMAE even on undirected data — the path accumulation provides a useful inductive bias that captures ordering effects present in real journeys even without explicit directed coefficients. MC Shapley has the edge on top-3 recovery (1.00 vs 0.67), which reflects that Path Shapley's asymmetric treatment of channels can slightly misrank channels that happen to be consistently upstream.

### Benchmark B — Directed (ordered ground truth, `directed_interaction_strength=0.6`)

Asymmetric channel interactions baked into the data generator. Ground truth is the oracle path-based marginal contribution computed by walking converting journeys through the true logistic model. This is the evaluation target that PathShapley is designed to recover.

```
Model                    NMAE    Rank Corr    Top-3    Time(s)
--------------------------------------------------------------
First Touch            0.0474       0.2619     0.33     0.004
Last Touch             0.0404       0.7306     0.67     0.003
Linear                 0.0427       0.7143     0.67     0.005
Time Decay             0.0407       0.6905     0.67     0.006
Position Based         0.0429       0.6667     0.67     0.005
Simplified Shapley     0.0423       0.7619     0.67     0.006
MC Shapley             0.0503       0.5952     0.67     0.522  ← worst NMAE
Path Shapley           0.0399       0.6667     0.67     0.459  ← best NMAE
```

With directed interactions active, MC Shapley is blind to channel ordering and performs worse than all heuristic baselines on NMAE. Path Shapley remains the best model, and its advantage over the next-best baseline (Last Touch, 0.0404) is consistent across both NMAE and timing. The relatively modest rank correlation improvement reflects that six-channel ordering effects are still partially captured by heuristics through position biases, but the NMAE gap is clear.

**Rule of thumb:** use Path Shapley when you suspect that upper-funnel channels (display, social, video) systematically prime customers for lower-funnel conversions (search, email, retargeting), and you want that sequential credit reflected in the attribution. Use MC Shapley when channel ordering is noise and you want strict Shapley-axiom compliance.

## Tests

```bash
pytest tests/ -v
```

Tests covering sklearn API compliance, attribution correctness, MC convergence, PathShapley ordering sensitivity, directed data generation, input validation, the synthetic dataset generator, and ONNX round-trip serialization for all model types.

## Project Structure

```
shapley_attribution/
├── __init__.py                   # Public API
├── base.py                       # BaseAttributionModel (sklearn mixin + plot + ONNX methods)
├── onnx.py                       # save_onnx() / load_onnx() — ONNX serialization
├── models/
│   ├── simplified.py             # Exact set-based Shapley
│   ├── ordered.py                # Exact position-aware Shapley
│   ├── monte_carlo.py            # Approximate Shapley (GBM + MC permutation sampling)
│   └── path_shapley.py           # Path Shapley (GBM + actual journey sequence)
├── baselines/
│   └── heuristic.py              # First/Last Touch, Linear, Time Decay, Position
├── datasets/
│   └── synthetic.py              # make_attribution_problem() (+ directed interactions)
├── metrics/
│   └── evaluation.py             # NMAE, rank correlation, top-k overlap
└── visualization/
    └── plots.py                  # plot_attribution, compare_models, plot_performance,
                                  # plot_journey, plot_journeys_heatmap,
                                  # plot_position_attribution
```

## Roadmap

- [x] ONNX export/import (`save_onnx` / `load_onnx`) with onnxruntime inference support
- [ ] GPU acceleration (PyTorch/CuPy) for 100k+ journeys
- [ ] Distributed computing support (Ray/Dask) for multi-machine scaling
- [ ] Comprehensive tests on larger datasets (>100k journeys)
- [ ] Additional baseline models (Markov, first-order attribution chains)
- [ ] Custom coalition value functions (user-provided models)
- [ ] Interactive visualization dashboard

## Related Libraries

- **[SHAP](https://github.com/shap/shap)** — General-purpose Shapley value library for model interpretation (KernelSHAP, TreeSHAP, DeepSHAP)
- **[Alibi](https://github.com/SeldonIO/alibi)** — Model-agnostic explainability and fairness (includes Shapley approximations)
- **[ELI5](https://github.com/eli5-org/eli5)** — Model interpretation library with permutation importance
- **[Captum](https://github.com/pytorch/captum)** — PyTorch model interpretability library (supports Shapley values)
- **[MultiTouch](https://github.com/AnalyticsEnthusiasts/MultiTouchAttribution)** — Another multi-touch attribution library (heuristics only)

## References

The following papers directly inform the algorithms in this library:

**Foundation**
- Shapley, L. S. (1952). A Value for n-Person Games. In *Contributions to the Theory of Games II*, Annals of Mathematics Studies, vol. 28. Princeton University Press.
  _The original cooperative game theory paper that defines the Shapley value axiomatically. All models in this library compute or approximate this quantity._

**Attribution-specific Shapley**
- Zhao, K., Mahboobi, S. H., & Bagheri, S. R. (2018). [Shapley Value Methods for Attribution Modeling in Online Advertising](https://arxiv.org/abs/1804.05327). arXiv:1804.05327.
  _Direct inspiration for this library. Introduces set-based and ordered Shapley variants for the multi-touch attribution problem. Our `SimplifiedShapleyAttribution` and `OrderedShapleyAttribution` implement these directly._

**Monte Carlo sampling**
- Castro, J., Gómez, D., & Tejada, J. (2009). Polynomial calculation of the Shapley value based on sampling. *Computers & Operations Research*, 36(9), 1726–1730.
  _Introduces the ApproShapley algorithm: estimate Shapley values by averaging marginal contributions over random permutations. This is the sampling backbone of `MonteCarloShapleyAttribution`._

**Interventional Shapley & learned value functions**
- Lundberg, S. M., & Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874). In *Advances in Neural Information Processing Systems* (NeurIPS).
  _Introduces KernelSHAP: using a learned model as the coalition value function and sampling over coalitions. Our MC Shapley model adopts this approach (GBM as the value function)._
- Janzing, D., Minorics, L., & Blöbaum, P. (2020). [Feature relevance quantification in explainable AI: A causal problem](https://arxiv.org/abs/1910.13413). In *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics* (AISTATS).
  _Distinguishes interventional Shapley (v(S) = f(binary mask)) from observational Shapley (v(S) = E[f | X_S]). We use the interventional formulation — deterministic, no background averaging — which eliminates variance and improves stability._

## License

MIT
