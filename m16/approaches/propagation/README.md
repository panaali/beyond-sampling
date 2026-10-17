# Distribution propagation for the accuracy of M_16,10

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [STATUS](../../../STATUS.md)

## Question
Can we estimate the accuracy of M_16,10 deductively, without running the model on random inputs, by
propagating an approximate distribution of the state step by step? The state is the hidden state plus
discrete bookkeeping of the running top-2, the style of assumed-density filtering, Gaussian-sum filters and
cumulant propagation. To compete with sampling, the method must match sampling's RMSE at equal compute.

## Status
**Done. Negative result.** None of the three propagators gets within 10 pp of the truth (0.9532). The
diagnostics below indicate that the gap comes mainly from the method family, not from tuning: even exact
low-order moments, conditioned on fine mechanism states, leave biases of 0.35 to 3.5 pp at the final
read-out. The ARC bar is about 0.1 pp. A shared-machine slowdown cut short the RMSE run on noisy copies
(see Results 4).

## Method (key files)
All three propagators are deterministic and use only the weights. FLOPs are counted analytically inside the code.

1. `prop.py`: **input-space branches with exact cones.** There are no biases, so within a fixed gate history
   h_t = A_t x is exactly linear in x, and every event is a homogeneous linear constraint on x. These events
   are a gate being on, x_t beating the running top-2, and argmax = p2. A branch is (p1, p2, A_t) plus
   N(mu, S) approximating x given the branch's cone, using sequential truncated-Gaussian moment matching (ADF).
   Branches split on the rank of x_t and on ambiguous gates. Branches with mass below eps are pruned.
2. `gsf.py`: **Gaussian-sum ADF in state space.** Each discrete state (p1, p2) carries a Gaussian over
   z = (h, m1, m2) in R^18. Each step appends x_t, splits 3 ways on the rank of x_t (truncation), and
   pushes z through the ReLU with *exact* mean and covariance of a ReLU'd Gaussian. Those use
   bivariate-normal formulas in `relu_moments.py` (checked against MC in `test_relu_moments.py`), plus Stein's
   lemma for the cross-covariances. Components are then merged to K per (p1, p2). At the last step it splits on
   ambiguous gates, and P(argmax = p2) is a 9-margin Gaussian orthant probability (sequential moment matching).
3. `gsf2.py`: **the same, keyed by mechanism state (p1, p2, current gate pattern).** Gate patterns are
   enumerated. The probability of each gate uses a tree approximation (exact bivariate normal with the most
   correlated decided gate), which makes collinear gates such as step 0's exactly implied. Within a pattern
   h' = diag(g) u is exactly linear, so no ReLU closure is needed. A gate side with mass below eps is absorbed.
   The masses of (p1, p2) are renormalised to their exact model-free values 1/(t(t+1)). The K heaviest
   patterns per (p1, p2) are kept, and the rest are pooled.
4. `estimator.py`: leaderboard wrapper around `gsf2` (K=1; budget sets eps). `DEDUCTIVE = True`.
   Cost = FLOPs / 6e3.

The `diag_*.py` files are **diagnosis only**. They use random inputs and are never called by an estimator.

## Results (original model, truth 0.95318)
Commands are run from this folder with the venv active.

**1. Bias vs compute.** "passes" = counted FLOPs / 5,920, the FLOPs of one 10-step forward pass.

| method (command) | setting | estimate | bias | passes |
|---|---|---|---|---|
| prop.py (`python run0.py`) | eps=1e-3 / 1e-4 / 1e-5 | 0.858 / 0.831 / 0.739 | -10 / -12 / -21 pp | 2.0k / 20k / 189k |
| (same) mass dropped by pruning | | 99% / 95% / 88% | | |
| gsf.py (`python run1.py`) | K=1, no splits / K=4, splits 2-3 | 0.294 / 0.274 | -66 / -68 pp | 6k / 443k |
| gsf2.py (`python run2.py`) | K=1, eps 1e-3 / 1e-4 | 0.320 / 0.261 | -63 / -69 pp | 22k / 117k |
| gsf2.py | K=4, eps 1e-4 / K=16, eps 1e-5 | 0.316 / 0.388 | -64 / -57 pp | 132k / 633k |
| estimator.py (budgets 1k..256k) | eps from budget | 0.42 / 0.44 / 0.35 / 0.28 / 0.26 | -52..-69 pp | 10k / 13k / 22k / 49k / 116k |

The bias does **not** shrink with compute; it is non-monotone. Sampling at the same compute has s.d.
0.2 pp at 10k passes and 0.06 pp at 116k. **The method does not beat sampling at any compute level**,
being off by about 300-1000x in RMSE.

**2. Where it breaks (diagnostics, random inputs, diagnosis only).**

- `python diag_cones.py` [measured]: 1M inputs fall into **534k distinct gate histories**, and singletons
  hold 43% of the mass. Exact cone enumeration (prop.py without pruning) is hopeless, so states must be merged.
- `python diag_patterns.py` [measured]: the (p1, p2, current gate pattern) keys reach ~2,600 by step 9;
  10 patterns per (p1, p2) cover 95-99% of the mass.
- `python diag_gsf2.py 1 1e-3` [measured]: the error in E[h | p1, p2] grows from 0.005 (step 0) to 1.6 (step 8),
  against a per-neuron s.d. of about 5. With 1,000 patterns per state the error is still 0.86 at step 6.
- `python diag_final.py` / `diag_final3.py` [measured]: feed the last step with Gaussians whose moments are
  fitted exactly from 1M MC samples, keyed by (p1, p2), (p1, p2, g_9) or (p1, p2, g_9, g_8), over
  (h_9, m1, m2) or (h_9, all x). The final-step machinery still predicts **0.718 / 0.740 / 0.757**, against
  0.953. **Even perfect moment propagation would fail with this read-out.**
- `python diag_skew.py` [measured]: a cleaner ceiling that skips my final-step machinery. Take the exact
  margin (logit of p2 minus the best wrong logit) and predict P(margin > 0) from its exact cumulants per
  cell:

| cells | Gaussian (2 cumulants) | Edgeworth (4 cumulants) |
|---|---|---|
| 90 (p1, p2) | 0.918 (-3.6 pp) | 0.936 (-1.7 pp) |
| 1,310 ((p1, p2) x 3x3 gap bins, blind flag) | 0.932 | 0.947 |
| 10,291 (10x10 gap bins) | 0.939 (-1.4 pp) | 0.950 (-0.35 pp) |

  The margin is right-skewed (mean skewness 0.64). There are no biases and the ReLUs are positively homogeneous, so margins scale with
  order-statistic gaps and the error is the far left tail. Low-order closures miss that tail.

**3. ReLU moment formulas.** `python test_relu_moments.py` [measured]: these agree with 4M-sample MC to
within MC error (max abs err 0.003 on covariances of order 1).

**4. ARC protocol on noisy copies.** The command is `python eval_arc.py` (true_acc n=4M).

| sigma | setting | passes | RMSE | sampling RMSE at equal passes |
|---|---|---|---|---|
| 0.0025 | K=1, eps 1e-3 | 21.9k | **62.3 pp** | 0.165 pp |
| 0.0025 | K=1, eps 1e-4 | 117k | **67.4 pp** | 0.071 pp |

[measured] The truths are 0.915-0.946, and the estimates are 0.26-0.32 on all six copies. The estimator
barely responds to the weights. **The sigma=0.005 run is incomplete.** The process was killed at about 21:30 (the DRI reports another
agent's process cleanup; it also coincided with the 1,500 s timeout under a load average of about 48), so no
`results_arc.json` was written. Per the DRI it was not restarted, because the sigma=0.0025 result is already
decisive. The sigma=0.0025 rows above come from the run's log. The estimator's insensitivity on the original
model and at sigma=0.0025 implies RMSE of roughly 55-65 pp there too [inferred].

## Compute accounting
FLOPs are counted in code (`Counter`):
- each rank-1 truncation update: 4d^2+6d+40;
- moment transforms: matrix products counted as 2mnk;
- each bivariate-normal CDF (Owen's T): 80;
- each pairwise ReLU moment: 120.

One forward pass is 5,920 FLOPs (10 x (2*16*16 + 3*16) + 2*10*16). The leaderboard wrapper divides by 6e3.
There is no model-free precomputation, no input generation and no LPs.

## Claims
- [measured] A Gaussian (or Gaussian-sum, K up to 16) closure of (h, top-2 values) conditioned on (p1, p2), or
  on (p1, p2, gate pattern), estimates accuracy with -52 to -69 pp bias at 6k-630k forward-pass
  equivalents. More compute does not help.
- [measured] Input-space exact-cone propagation with ADF loses 88-99% of the mass to pruning and is biased
  by 10-21 pp. Gate histories are too diverse (534k per 1M inputs) to enumerate.
- [measured] Even with exact cumulants of the final margin per mechanism cell (up to 10k cells), Gaussian /
  Edgeworth read-outs are biased by -1.4 / -0.35 pp.
- [inferred] Therefore moment or cumulant propagation, at any order up to 4 and at this granularity of
  discrete state, cannot reach the ~0.1 pp needed to match sampling at 64k passes. The error lives in a
  4.7% left tail of a skewed, scale-mixed margin.
- [speculative] A closure that treats order-statistic spacings with their exact (near-exponential)
  laws, rather than Gaussians, might fix the read-out. That is a different method, closer to the probe and
  stratified approaches.

## Alternatives not ruled out
For the main claim ("Gaussian closure fails because of the method, not my implementation"):
1. *Implementation bugs in the final step.* My final step (gate enumeration with the tree approximation,
   then sequential ADF over 9 margins) gives 0.72-0.76 even with exact moments. The clean margin-cumulant
   ceiling is 0.918, so about 20 pp of the gsf2 bias is due to my final-step approximations or a bug, not
   to Gaussianity alone. **Partly tested**: the pieces were tested (ReLU formulas vs MC; rank masses 1/(t(t+1))
   match MC to 1e-3; step-0 means match MC to 0.005). The full final step was not validated
   against MC per component.
2. *A better family would work.* Skew-normal, scale-mixture or Edgeworth closures. **Tested only at the
   read-out**: with exact 4 cumulants, the bias is still -0.35 pp at 10k cells. A properly propagated
   higher-order closure would be no better than that ceiling.
3. *Different state variables* (spacings of the top order statistics, or sequential ranks to depth 3) could
   make within-cell distributions nearly Gaussian. **Not tested.**
4. EP (multi-pass) instead of one-pass ADF for cone probabilities in `prop.py`. **Not tested**; it
   would not address the 534k-cone explosion.

## Failure modes
- Truncated-Gaussian moment matching leaks mass across ReLU hinges. On redundant constraints (e.g. 16 gates
  all ∝ x_0 at step 0) it creates spurious branches. `gsf2` handles exact collinearity but not
  near-redundancy.
- Merging gate patterns produces multimodal-to-Gaussian collapse. Errors in E[h] then grow over steps.
- The read-out probability is a tail event (4.7%), so relative errors of a few percent in the tail are fatal.
- Cost is at least about 10k forward-pass equivalents even at the coarsest setting, so there is no
  low-budget regime.
