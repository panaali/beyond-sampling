# Structural-reliability (FORM / SORM / system) estimator for M_16,10

## Question
Can the structural-reliability toolkit give a deductive estimate of M_16,10's failure probability (about 4.7%)
that is as precise as random sampling at equal compute, on the original model and on noisy copies? The toolkit:
design points, FORM, Breitung/Hohenbichler SORM, Ditlevsen bounds and the Hohenbichler–Rackwitz multinormal
for series systems, plus design-point importance sampling as a hybrid.

## Method
**Homogeneity.** The net is positively homogeneous, so every failure set is a cone through the origin. In
x-space every mode's design point is the origin and β = 0, so FORM is degenerate. Fix: **condition on the
rank-order chamber and map to Rosenblatt (order-statistic) space** (`rosen.py`).
- Chamber = which position holds each rank.
- u ∈ R^10, u ~ N(0, I), maps exactly to the sorted values: u0 → 2nd largest (conditioned >0, or ≤0 for the
  "blind" region, prob 11/1024), u1 → max given 2nd, u2..u9 → ranks 3..10 top-down.
- u = 0 is the median ("canonical") input. Ties sit at u → ±∞, so each failure mode now has a finite design
  point. The map is non-linear, so homogeneity is broken in the right place.

**Pieces.**
- **Modes**: for chamber c and wrong rank k, g_k(u) = logit[pos of rank 2] − logit[pos of rank k]. The input is
  correct iff g_k > 0 for all k (a series system).
- **Chambers**: the 720 ordered (p1, p2, p3) triples, each with one fixed (seeded, model-independent)
  arrangement of ranks 4–10. Both sign regions: 1,440 chambers × 9 modes = 12,960 limit states.
- **Design points** (`form.py: design_points`): batched search on the exact piecewise-linear gradients
  (autograd). An HL-RF step proposes a direction, an exact ray search (grid + bisection) puts the point on
  g = 0, and the shortest point is kept (6 rounds). Plain HL-RF oscillates between gate patterns and does not
  converge (`diag1.py`).
- **SORM** (`sorm.py`): in x-space g is exactly linear inside the gate pattern at u*. So the u-space Hessian
  there is Σ_i (∂g/∂v_i) ∇²v_i(u), and it comes from the model-free Rosenblatt map. This is Hohenbichler's
  SORM. Curvature from gate kinks is ignored.
- **System** (`system.py`): SORM-equivalent β per mode, then the midpoint of the Ditlevsen bounds of the union.
  Pairwise terms are bivariate normals from ρ = α_i·α_j (vectorised Owen's T). The HR-MVN (Genz) gives the
  same answer to 1e-4.
- **Estimator** (`estimator.py`, `DEDUCTIVE = True`): `est(model, budget, rng)` picks the most accurate
  configuration in a cost ladder that fits the budget.
- **Hybrid** (`hybrid.py`, sampling-based): a mixture importance-sampling proposal centred at the original
  model's rank-1 and rank-3 design points, plus region stratification.

## Status
Done. **Negative result**: no configuration comes near sampling at equal compute.

## Results
Truth: `results/leaderboard_truths.json` (n = 4M). For the original model the 20M truth 0.953179 is also used.

| config (`sweep.py <cfg> all`) | cost (fwd-equiv) | bias, original | RMSE σ=0.0025 | RMSE σ=0.005 | sampling RMSE at same cost |
|---|---|---|---|---|---|
| L0 (3 modes, 1 round, 180/90 chambers) | 35k | +1.90 pp | 2.43 pp | 3.17 pp | ~0.13 pp |
| L1 | 111k | +1.56 pp | 1.98 | 2.73 | ~0.07 |
| L2 | 337k | +1.06 pp | 1.41 | 2.11 | ~0.04 |
| M3 (all modes, 3 rounds) | 1.22M | +0.56 pp | 0.80 | 1.17 | ~0.02 |
| FORM only (no SORM), full search | 2.17M | −0.36 pp | 0.31 | 1.19 | ~0.017 |
| **full (SORM + Ditlevsen)** | **2.48M** | **+0.15 pp** (vs 20M truth; +0.17 vs 4M) | **0.30** | **0.80** | ~0.016 |
| multi-start, 4 starts (`multistart.py 4 ...`, 4 models) | 9.9M | −1.42 pp | (−1.78 on s0.0025_1) | (−1.65, −1.09) | ~0.008 |

**Full config, error per copy** (pp; `sweep_full_all.json`):
- σ = 0.0025: +0.13, +0.49, +0.42, +0.08, +0.29, +0.17.
- σ = 0.005: +0.92, +1.60, +0.56, +0.15, +0.12, +0.22.

The errors are all positive: missed failure mass. They grow as accuracy falls. There are no blow-ups (v4
probes had −9 pp on one copy).

**Hybrid, design-point IS** (`eval_hybrid.py 16000 24`), RMSE at 16k evals:

| model | sampling | region stratification only | DP-IS |
|---|---|---|---|
| original | 0.135 pp | 0.144 pp | 0.139 pp |
| σ = 0.0025 | 0.175 pp | 0.175 pp | 0.169 pp |
| σ = 0.005 | 0.234 pp | 0.210 pp | 0.221 pp |

This is no gain beyond noise. Failure (5–18%) is not rare, and much of its mass is far from the two design
points.

**Diagnostics** (random inputs, diagnosis only):
- `analyze2.py`: positive region, original model, same chambers. Per-chamber failure from SORM + system is
  0.0367; chamber MC gives 0.0379 (corr 0.84). FORM gives 0.0418.
  - Rank-3 mode: FORM 0.0337, SORM 0.0290, MC 0.0279.
  - Blind region: failure 0.83–0.86 against MC 0.78.
- `diag_split.py`: the one-arrangement chamber design itself costs only −0.04 to +0.14 pp. Nearly all of the
  error is the FORM/SORM approximation.
- `diag_modes.py s0.005_1`: the "picks the max" mode is the main miss (SORM 0.069 vs MC 0.100).
  `diag_multi.py`: the best of 21 starts reaches 0.072, and a union of the distinct design points (about 2.5
  per chamber per mode) reaches 0.093–0.097. So **there are multiple design points per mode**.

## Compute accounting
- Each g evaluation counts as 1 forward pass × 1.3; the extra 0.3 is the Rosenblatt map, about 2k FLOPs. A
  gradient counts as 3 × 1.3.
- SORM counts 20 forward-equivalents per limit state: a finite-difference Hessian of w·v(u), about 120k FLOPs.
  The 10×10 eigendecompositions and the Owen's-T / bivariate-normal work are not counted. They are model-free
  in form but run per model, at about 5% extra.
- Full config: 12,960 limit states × (≈137 evals × 1.3 + 20) ≈ 2.48M.
- Nothing is precomputed, except the hybrid's design-point proposal. That comes from the original model and is
  reused unchanged for every copy (declared).

## Claims
- [measured] Mapping to Rosenblatt order-statistic space removes the homogeneity degeneracy. Design points are
  finite, and per-chamber FORM/SORM mode probabilities correlate 0.73–0.94 with chamber MC.
- [measured] Rosenblatt-curvature SORM fixes most of FORM's error on the dominant 2nd-vs-3rd mode (0.0337 →
  0.0290, MC 0.0279).
- [measured] Full SORM + Ditlevsen: bias +0.15 pp on the original. RMSE 0.30 / 0.80 pp on the noisy copies at
  2.48M forward-equivalents. Sampling at that compute: about 0.016 pp, so this is **20–50× worse in RMSE**.
  At cheaper budgets the search is under-converged and the bias reaches +1–3 pp.
- [measured] The design-point IS hybrid gives no gain over plain sampling at 16k.
- [inferred] The leftover bias comes from mode explosion. Each (chamber, wrong class) has several design points
  (piecewise-linear gates make the limit state non-smooth with several local minima), and the single-start
  search misses some. Adding them as half-spaces over-counts (−1.4 pp), because the true region near each one
  is a polyhedral wedge, not a half-space.
- [speculative] An exact polyhedral treatment (cone volumes per gate pattern) would fix the over-counting, but
  the number of patterns is the same explosion.

## Alternatives not ruled out
- **Main claim: residual error = multiple design points.**
  - Alt 1: non-convergence of the search. Partly tested: more rounds help (M3 → full). A 21-start best-of
    still misses 30% of the rank-1 mass, which points to multiple design points.
  - Alt 2: the chamber design (one arrangement of ranks 4–10). Tested: it explains ≤0.14 pp.
  - Alt 3: curvature from gate kinks, which SORM ignores. Not separated.
- The Ditlevsen-midpoint combination is arbitrary inside narrow bounds. Bound width is small in the positive
  region and up to 0.04 in the blind region (×11/1024).
- Untested: more rest-arrangements; LP-based exact distance to each gate cone instead of ray search;
  importance sampling with many design points or with an adaptive (cross-entropy) proposal.

## Failure modes
- Cost is dominated by the design-point search: about 140 passes per limit state, against 12,960 limit states.
  Even a perfect deductive answer at this size costs more than sampling needs for 0.05 pp.
- Non-rare modes (β ≈ 1) and the blind region (failure about 0.8): FORM/SORM's asymptotics do not apply.
  System bounds still land within about 0.06, which is harmless only because the region weighs 1%.
- Lower-accuracy copies have more design points per mode, so the bias grows with noise (+1.6 pp at 82%
  accuracy).
- Multi-start unions over-count. The answer swings from +0.2 to −1.4 pp depending on deduplication, so the
  method is not robust.
