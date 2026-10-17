# sci_laplace: can the tie-band integral be reduced to computable "Laplace" pieces?

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [STATUS](../../../STATUS.md)

## Question
The error of M_16,10 is mostly mass in thin bands next to rank ties (tube, boundary, observer). Can the band
integral be evaluated *deductively*, by an asymptotic or generating-function reduction (a few computable
saddle points, linear or additive laws whose distribution is a convolution, or a time-causal recursion),
precisely enough (about 3% relative on the R23 term, about 0.1 pp overall) to beat sampling at 16k-64k
forward passes, on the original, seeds 0-4 and noisy copies?

## Method
Plain words. Each 2-3 tie face point y (top-3 positions a<b, c; tie value u; max xc; 7 "rest" inputs below u)
has a signed network tie offset delta(y): the distance along s = x_a - x_b at which the network's a-vs-b
margin crosses zero. Band mass is about W(0,|delta|). The face is 9-D per position triple. I tested, with
random inputs **for diagnosis only**, whether delta (or the band mass F) has structure that a Laplace-style
calculation could integrate exactly:
1. a few dominant face points (saddle-point / Laplace concentration);
2. linear in the face coordinates (Gaussian law, closed form);
3. additive in the 7 rest inputs given the top-3 (law = convolution of 7 one-dimensional laws, i.e. a
   generating-function calculation), including an oracle ceiling test;
4. causal: decided when the later tie element b arrives (so only the past matters, enabling a recursion in time);
5. few gate flips inside the band (delta = a min/max of a few linear-fractional forms).
All of these failed, so I also built **LTQ**, a deterministic quadrature in tube's class. It uses one
low-discrepancy point set per position triple (instead of tube's single 10-D Sobol whose 10th coordinate picks
the triple), and no model-fitted importance tables.

Files: `lp.py` (helpers; exact walks reused read-only from `../tube/tube.py`; `delta_ab` = exact signed tie offset),
`diag_causal.py`, `diag_linear.py`, `diag_dist.py`, `diag_additive.py`, `diag_additive_m0.py`, `diag_sobol.py`,
`diag_ceiling.py`, `diag_flips.py`, `diag_qmc_rest.py`, `diag_qmc9.py` (diagnostics, random inputs),
`lqt.py` + `estimator.py` (LTQ, `DEDUCTIVE = True` in the sense of fixed nodes), `tune.py`, `eval_all.py`.

## Status
Done. **Negative for the deductive (analytic) route.** LTQ is a deterministic quadrature. It does not
consistently beat sampling (it ties or loses on seeds and sigma 0.005), so it is not a solution.

## Results
All commands are run in this folder with the venv active and `OMP_NUM_THREADS=1`. Original model unless stated.

| test | result | command |
|---|---|---|
| Laplace concentration of band mass over faces | none: top 1% / 10% / 50% of faces carry 10% / 42% / 88% of F | `python diag_dist.py` + inline analysis of `data/orig_faces.npz` |
| tail of \|delta\|/\|x\| | quantiles 50/90/99%: 0.0046 / 0.0185 / 0.059 (heavy tail, ratio-like) | `python diag_dist.py` |
| delta linear in (u, xc, rest) within a triple | median R^2 **0.16** (40 triples) | `python diag_linear.py` |
| delta additive in rest given top-3 (backfit, 6 bins) | median R^2 0.63 (overfit null 0.12); for F 0.40 | `python diag_additive.py` |
| tie margin D0 additive / effective slope additive | R^2 0.69 / 0.28 (null 0.11) | `python diag_additive_m0.py` |
| Sobol indices of delta over rest inputs | sum first-order 0.83, sum total 1.34; every rest position contributes 0.09-0.16 (x9 0.04) | `python diag_sobol.py` |
| **oracle ceiling**: replace delta by its best additive fit (800 samples per config) | E W(0,\|delta\|) off **-4.0%** in aggregate, 5.5% median per config (budget about 3%) | `python diag_ceiling.py` |
| W(0,\|delta\|) vs exact F | ratio 1.14, per-config corr 0.69 (third classes, blind cut, multiple intervals) | `python diag_ceiling.py` |
| causal (variance of F from resampling rest inputs before b / after b) | 39% / 15% of Var F | `python diag_causal.py` |
| gate flips inside the band [0, delta] | 0.1-2.1 per point; 15-68 distinct flip sets per triple; often one dominant (t=c, n0) | `python diag_flips.py` |
| QMC vs MC on the 7-D rest integral, fixed top-3, n=256 | variance ratio median 3.4 (1.7-18) | `python diag_qmc_rest.py` |
| QMC vs MC per triple, 9-D face, m=16 / 64 | 2.1x / 2.2x (24 reps, about +-40%) | `python diag_qmc9.py` |

**LTQ vs sampling** (`python -W ignore eval_all.py 16000 8`, `... 64000 6`; truths `results/leaderboard_truths.json`;
the RMSE is over point-set randomisations; sampling RMSE is analytic at the same measured cost; ratio = MSE/sampling MSE, pooled per group):

| budget | orig | seeds 0-4 | sigma 0.0025 (6) | sigma 0.005 (6) |
|---|---|---|---|---|
| 16k, randomised RMSE (ratio) | 0.089 pp (0.28) | 0.275 pp (1.26) | 0.146 pp (0.56) | 0.239 pp (1.00) |
| 16k, fixed seed-0 rule (ratio) | 0.075 pp (0.20) | 0.339 pp (1.90) | 0.112 pp (0.33) | 0.151 pp (0.40) |
| 64k, randomised RMSE (ratio) | 0.027 pp (0.11) | 0.113 pp (0.85) | 0.056 pp (0.33) | 0.115 pp (0.92) |
| 64k, fixed seed-0 rule (ratio) | 0.056 pp (0.45) | 0.084 pp (0.47) | 0.081 pp (0.70) | 0.177 pp (2.19) |

The mean error across randomisations is within -0.05..+0.05 pp in every group, so there is no detected bias.
Per model, it loses at 64k on seed1 (1.59), s0.005_0 (2.46) and ties seed3 (1.07). At 16k it loses on seed1,
seed4, s0.005_0 and s0.005_1. Per-term spread on s0.005_0 and seed1 is dominated by the R23 face term (sd 0.28 pp at 16k).
Raising the exact-walk cut-off k0 from 0.04 to 0.16 does not help.
Per-model lines: `eval_16000.log`, `eval_64000.log`, `eval_*.json`.

## Compute accounting
- Unit = one RNN forward pass on one sequence. Face walks use tube's accounting: a linear cell = value+tangent
  recomputed from the flipped step, 2(T-t)/T; the shared prefix is counted once; the tail node costs (T-t0)/T.
  Blind points cost 1. Measured cost is 7-8 passes per face. Reported costs are the measured totals (below budget on
  seeds, whose faces are cheaper).
- Model-free and free: face masses Z23, Z12 and 11/1024, the inverse CDFs, point sets.
- Fitted on the original model (declared): the budget shares (0.66/0.27/0.05), the per-face cost used to size
  the rules, and k0 = 0.04. There are no per-triple importance tables.
- Diagnostics used about 1e6-3e6 pass-equivalents of random inputs in total (diagnosis only).

## Claims
- [measured] The band offset is not linear (R^2 0.16), not concentrated (top 10% of faces carry 42%), and not
  causal (inputs after b carry 15% of the variance) in any form that a closed-form or saddle-point calculation could use.
- [measured] Even an **oracle** additive model in the rest inputs, whose law a generating-function/convolution
  calculation could integrate exactly, misses E W(0,|delta|) by 4% in aggregate and 5.5% per configuration.
  A deductive (non-oracle) version would be worse. So the generating-function route cannot reach the about 3% needed.
- [measured] Inside the misclassified band only 0-2 gates flip (often one dominant flip), but the flip set
  varies (15-68 sets per triple), and the effective slope is not additive (R^2 0.28).
- [measured] QMC *does* help on the face integral once each position triple gets its own point set (2x at
  m=16, 3.4x at n=256 on the rest dims). This contradicts the inference in tube's README that QMC cannot help.
  The likely reason tube saw none is that it used the triple as a QMC coordinate [inferred].
- [measured] LTQ beats sampling on the original model and on sigma 0.0025 (MSE ratio 0.1-0.6), but ties or
  loses on seeds 0-4 and sigma 0.005. A single fixed rule (the deterministic entry) is erratic: 0.2-2.2x across groups.
- [inferred] The gain from tie-based variance reduction shrinks on lower-accuracy models, because the R23 face term
  grows (7.3% on s0.005_0 vs 2.9%) and its per-face CV stays high.

## Alternatives not ruled out
- A better coordinate system might make delta additive or low-rank (e.g. record-process coordinates, or
  ratios of rest inputs to u). I tried only raw rest values, D0 and log slope. Cheapest test: rerun
  `diag_ceiling.py` with the backfit done on the record-time/running-max coordinates.
- Second-order (pairwise) ANOVA terms would shrink the 4% oracle gap, but the cost (21 pairs x grid per triple
  x 360 triples) exceeds 64k before any accuracy gain. Not tested.
- The ceiling test uses W(0,|delta|), not F. Its 14% gap to F (third classes, blind cut, multiple intervals) is an
  extra term that any analytic route must also model.
- The randomised-RMSE ratios come from 6-8 randomisations per model (about +-30-50% per model). Group values are firmer.
  This has not been run on the DRI's leaderboard.

## Failure modes
- LTQ: low-accuracy models (seed1, seed4, s0.005_0/1), where face variance dominates. The fixed rule
  sometimes lands 2-3 RMSEs off (s0.005_0: +0.35 pp at 64k). Budgets under about 2.7k cannot give each triple one face.
- The analytic route fails because of interactions between the rest inputs (20-40% of variance), which
  are mediated by gate patterns. This is the same obstacle as E21/E23, now with an oracle bound on the best additive law.
