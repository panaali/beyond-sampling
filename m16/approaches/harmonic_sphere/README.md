# harmonic_sphere: ARC's harmonic deduction-projection, adapted to the bias-free RNN (T38)

↑ [Approaches index](../README.md) · [INDEX](../../../INDEX.md) · [TASKS](../../../TASKS.md) · [lit/web_arc](../../../lit/web_arc/README.md) (S10, S11)

## Question
The 2nd-argmax RNN has no biases, so its correctness depends only on x/|x|, and its accuracy is the uniform measure on
S^{T-1} of a union of polyhedral cones. ARC's deductive estimator for random halfspace intersections (Hilton 2025, S11)
propagates a degree-<=D harmonic density through halfspace cuts (Funk-Hecke projection) and matches Monte Carlo up to
logs on random instances. Does that method, applied step by step to the RNN, give a mechanistic estimate competitive
with sampling (or within ARC's own "roughly match" bar, MSE ratio ~4) at T = 3, 4, 5, 6?

## Method
**Formulation (the cheapest faithful one I found).** The state at step t is a list of branches. Each branch has
an *exact* polyhedral cone K_b (double description), a harmonic density g_b of degree <= D on S^{T-1}, and the linear map
A_b with h_t = A_b x on K_b.
- *Why x-space and not "the joint law of the needed functionals":* the future depends on x_{1..t} through h_t (16
  dims) and the top-2 order statistics. Both are at least as large as x_{1..t} itself for T <= 16. So S^{T-1} (dimension
  T-1) is already the smallest state. No reduction is possible this way. [inferred]
- *Order constraints are halfspaces:* the root is the T(T-1) order cells {x_i > x_p > others} (argmax i, target p).
  Their densities are model-free (precomputed, not charged).
- *Gate step:* the preactivation forms are W = Wh A_b + Wi e_t^T. For each neuron, the exact cone decides whether
  {W_j x >= 0} contains, misses or splits K_b. Only a split cuts the density: g+ = proj_D(g 1[W_j x >= 0]) and
  g- = g - g+, so mass is conserved exactly. When a gate is not split, it is decided by the cone. This extrapolates
  the leaf's linear map onto its Gibbs leakage, which is benign because the network is continuous across gate
  boundaries.
- *Readout:* the logit comparisons L_p - L_q are classified the same way, and split ones are cut. The last one is
  integrated by Funk-Hecke in O(m).
- *Cut (exact):* proj_D(g 1_H) = proj_D(g s_v^{(2D)}), where s_v^{(2D)} is the degree-2D Gegenbauer series of the
  halfspace indicator (exact, because g phi has degree <= 2D). It is computed on a product Gauss cubature that is exact
  to degree 4D. No model is ever evaluated at a node, and there are no random numbers.

Files: [`sphere.py`](sphere.py) (cubature, orthonormal harmonic basis via Dai-Xu recursion, Funk-Hecke coefficients,
cut), [`cones.py`](cones.py) (double description, checked against LPs), [`hprop.py`](hprop.py) (the algorithm, plus
variants `cells="global"|"tree"|"wedge"`), [`estimator.py`](estimator.py) (v3 interface), [`eval_all.py`](eval_all.py)
(all models, one deterministic run each), [`check_hilton.py`](check_hilton.py) (ARC's own setting),
[`diag_branches.py`](diag_branches.py), [`diag_leaves.py`](diag_leaves.py), [`diag_bias.py`](diag_bias.py) (diagnostics that
use random inputs only to measure true cone masses), [`truths_small.py`](truths_small.py) (T=3 truths, same protocol as
leaderboard.py, written here).

## Status
**Done. Decisive negative.** At every T and every D tried, the method loses to sampling by 2 to 7 orders of
magnitude in MSE at equal compute. The gap *grows* with D. T=6 and T=10 were not run, because the scaling makes them
pointless (see below).

## Results
All commands are run from this folder with `source ../../../.venv/bin/activate; OMP_NUM_THREADS=1`.

**0. Implementation check on ARC's setting** (`python check_hilton.py 4 6 40`; random v_i, n=4, k=6, mean p=0.012).
RMSE is 5.5e-3, 1.3e-3, 4.2e-4, 1.6e-4 and 1.0e-4 at D = 2, 4, 8, 12, 16 (m = 14 ... 1785). At D=16 this is worth
about 1e6 Monte Carlo samples, roughly 600 samples per coefficient (the MC truth noise, 5e-5, limits it). So the code
reproduces ARC's claim that the method beats MC per unit of state on random instances. [measured]

**1. Bias versus D on the RNN** (cells-first; `conv_T3.log`, `conv_T4.log`; `eval_all.py 3 2 4 8 16`,
`eval_all.py 4 2 4`). Aggregates are over all 29 leaderboard models of M_16,T (seed0-4 plus 24 noisy copies). Cost
is in FPE (rule 1). The ratio is MSE / sampling-MSE at the same cost (v3 definition).

| T | D | m | bias pp | RMSE pp | cost as implemented | ratio | cost, Hilton O(mD^2)/cut accounting | ratio |
|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 9 | -22.3 | 22.7 | 1.9e3 | 2.2e3 | 2.9e2 | 3.3e2 |
| 3 | 4 | 25 | -10.7 | 11.0 | 1.2e4 | 3.3e3 | 4.9e2 | 1.3e2 |
| 3 | 8 | 81 | -6.2 | 6.4 | 1.2e5 | 1.1e4 | 3.2e3 | 3.0e2 |
| 3 | 16 | 289 | -2.6 | 2.8 | 1.6e6 | 2.8e4 | 4.3e4 | 7.7e2 |
| 4 | 2 | 14 | -31.6 | 31.8 | 3.0e5 | 5.9e5 | 2.4e4 | 4.8e4 |
| 4 | 4 | 55 | -21.0 | 21.1 | 5.4e6 | 4.7e6 | 3.7e4 | 3.2e4 |

By group, the seeds (accuracy 0.9994 at T=3, 0.998 at T=4) are far worse: the T=3, D=16 seed ratio is 3.9e6 as
implemented. The bias is **negative on all 58 model x D cells**. [measured]

- The bias is a clean ~1/D law. On seed0 at T=3, err x D = 58, 62, 65, 62, 66, 67 pp at D = 10, 12, 20, 24, 32, 40,
  and at D=40 the error is still -1.67 pp against a true error rate of 0.06%. On s0.02_0 the constant is about 13 pp.
  At T=4 (seed0, D = 2..8), err x D = 58, 83, 100, 110 and is still growing. [measured]
- **Structural scaling:** MSE ~ D^-2, while cost ~ (#cuts) m D^2 ~ D^{T+1}. So the ratio to sampling grows like
  D^{T-1}. Raising D can never close the gap, and the smallest D is already 20-30 pp off. [inferred from the measured rates]
- **Exact cone bookkeeping alone already exceeds the budget at T=4.** The T=4 tree has 32-38k leaves (sampling
  2e5 inputs sees only ~4.4k of them, `diag_branches.py`). Double description costs 2.35e4 FPE, more than the 16k
  budget, before any harmonic work. Sampled leaf counts are ~5e2 (T=3), 4e3 (T=4), 9-21e3 (T=5) and 2-3e4 (T=6), with
  the exact count about 8x the sampled count at T=4. [measured]

**2. Where the bias comes from** (`python diag_bias.py 3 16 seed0 2000000`; `diag_bias_T3.log`). On seed0 (T=3, D=16)
the -4.0 pp splits into -0.11 pp of misallocation (propagated leaf mass versus true leaf mass) and **-3.91 pp at the
readout**. On s0.02_0 it is -0.45 / -0.50. The model's decision plane L_p = L_q nearly coincides with the order-cell
wall x_p = x_q, because a good model's decision boundary is the tie. The degree-D density of a cell leaks a band of width
~1/D across that wall, and that leakage lies on the wrong side of the nearly coincident decision plane, so it is scored
wrong. The better the model, the more of its boundary sits on ties, and the larger the bias: this is why the seeds are
worst. Mass also flows into sliver cones (`diag_leaves.py`, global mode: leaves with true mass < 1e-3 get 0.103 of
propagated mass against 0.045 true). [measured; the mechanism is inferred]

**3. Variants, all negative** (`conv_T3_tree.log`, `conv_T3_wedge.log`)
- `global` (no root cells; score each leaf with its own logits and the true target over the whole sphere) is worse:
  seed0 at T=3, D=16 gives -6.6 pp. `tree` (the same, organised as a mass-conserving binary decision tree) gives -9.8,
  -5.4, -3.4 pp at D = 10, 20, 30, against -5.8 (D=10), -3.2 (D=20), -2.1 (D=32) for cells-first. More cuts mean more smearing.
- `wedge` (wrong mass = the wedge between the cell wall and the decision plane, so that leakage beyond the wall is not
  scored) halves the seed0 bias (-1.75 pp at D=16, -0.91 at D=32). But it is unstable on noisy models (s0.02_0:
  -1.71 / -2.19 pp; s0.005_0: -0.63 / +0.23) and is still ~1/D.
- Richardson extrapolation in D (2 est(2D) - est(D), from the table above) leaves -0.1 to +0.2 pp at D=20/40 on T=3.
  That is still about 50x sampling's RMSE at the same Hilton-accounting cost (1.5e6 FPE gives 0.002 pp on seed0).

**4. Leaderboard (official, v3)**: `python m16/leaderboard.py run harmonic_sphere --H 16 --T 4 --budgets 16000 --groups seed`
(from the repo root; log in `leaderboard_T4.log`). At every leaderboard budget the estimator falls back to D=2, at a
cost of ~3e5 FPE per model. Result: see the log (seeds: bias about -31.6 pp, ratio ~1e7). Only the seed group was run,
because the method is deterministic and `eval_all.py` already covers all 29 models with one run each.

**ARC's bar.** ARC calls an MSE ratio of ~4 "roughly match". The best cell here is 1.3e2 (T=3, D=4, Hilton accounting,
all models, carried by the noisy copies). On the seeds the best is 1.1e4. That misses the bar by 1.5 to 6 orders of
magnitude.

## Compute accounting
Charged FLOPs: every harmonic cut (4 N m + N(2n + 6D) per cut, as implemented on the cubature), Funk-Hecke readouts
(13 m), double-description updates and tests (2n per ray test, plus adjacency checks), and logit maps (2T^2 H per
leaf). The "Hilton accounting" column replaces the implemented cut cost by m D^2 per cut (ARC's stated optimal
O(m D^2) via Givens rotations and Hermite tricks; not implemented here). That makes it a lower bound, given as the most
favourable comparison. Model-free precompute (cubature, basis, Funk-Hecke coefficients, root order-cell densities):
under 1e10 FLOPs per (T, D), not charged. Diagnostics used about 2e7 random inputs in total (truths for T=3 used
29 x 4M). CPU: about 1.5 core-hours in total.

## Claims
- [measured] The reimplementation reproduces ARC's random-halfspace behaviour (n=4, k=6: about 600 MC samples per
  coefficient at D=16).
- [measured] On M_16,3 and M_16,4 (29 models each) the estimate is biased low on every model, with bias ~ c/D.
  c ≈ 60-70 pp for well-trained seeds and ~13 pp for s0.02 copies at T=3; it grows with T.
- [measured] MSE ratio to sampling at equal cost: 1.3e2 to 4.7e6 (all models), 1.1e4 to 1.5e7 (seeds), worse as D grows.
- [inferred] The ratio scales like D^{T-1}, so no D and no budget makes this formulation competitive. At T>=4 the exact
  cone bookkeeping alone exceeds the 16k-FPE budget.
- [inferred] The dominant error is readout leakage across order-cell walls that nearly coincide with the model's
  decision planes. Hilton's sketching guarantee assumes rotation-invariant "future functions", and here they are
  concentrated on the tie hyperplanes. The isotropic degree-D projection is the wrong sketch for this ensemble.
- [speculative] The right sketch is anisotropic: high resolution normal to the tie faces and low resolution along them.
  That is essentially E25's tie-slicing (exact in the tie-normal direction), now with a deductive face rule instead of
  Sobol. See the Backlog row.

## Alternatives not ruled out
1. *A bug, not the method.* Against this: `check_hilton.py` passes, the cones agree with LPs on 200 random cones,
   leaf masses sum to 1, and the bias shrinks like 1/D, as Gibbs smearing predicts. Not done: an independent
   implementation, e.g. ARC's `deduct-est` code (private).
2. *The pre-asymptotic regime.* The 1/D rate might steepen at much larger D. Against this: err x D is flat from D=10 to
   40 at T=3. Even an MSE ~ D^-4 rate would leave the ratio growing as D^{T-3}.
3. *A better cut/readout ordering* (for example, logit cuts before gate cuts) might cut the readout leakage. I tried
   three orderings (first / global / tree) and one leakage-aware readout (wedge). None changes the 1/D law.
4. *A data-adapted sketch* (top eigenfunctions of E_f[f f^T] over DEV-model future functions). Not tried. It would
   need exact cuts in a non-polynomial basis, or a subspace of a high-degree P_D', where the cut costs as much as
   degree D'. So it saves state but not FLOPs.

## Failure modes
- Thin error sets: errors live in wedges of angle << 1/D next to ties, and a degree-D density cannot resolve them.
- Branching: one density per exact linear region. 3.5e4 leaves at T=4, and far more at T>=5 (H-M7). Python
  double description is also slow: the CPU time is far above sampling's.
- Signed densities: the projected g can be negative. This is harmless for linear estimates, but makes per-leaf
  diagnostics noisy.
