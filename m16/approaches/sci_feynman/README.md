# sci_feynman: exactly solvable core (tie segments) + structure of the corrections

## Question
Feynman-style attempt at Q4: find the exactly solvable part of the accuracy integral, expand around it, and check each
step with a cheap calculation. Can that give a deductive estimate of M_16,10's accuracy that beats sampling at equal
compute on the original, seeds 0-4 and noisy copies?

## Method
1. **Exact core (model-free).** Input space splits exactly into the blind region B = {2nd largest <= 0} (mass 11/1024),
   R23 (2-3 near-tie) and R12 (1-2 near-tie) segments: (x_a, x_b) -> (u, s) has Jacobian 1 and density
   phi(u)^2 exp(-s^2/4), so P(err) = 360 Z23 E_face[G] + 45 Z12 E_face[G] + P(err & B). This is the tube construction
   (E15), re-derived and re-implemented (`core.py`; R23 segment |s| < min(2(o1-u)/3, 2(u-o2)), B cut out via
   |s| > -2u; R12 |s| < min(2(w-o3)/3, 2w)). G is computed **exactly** by walking the linear cells along the segment
   (`core.walk_mass`, incremental cost accounting).
2. **Corrections, tested before use** (diagnosis scripts, random inputs allowed):
   order the face coordinates into "top" (positions a,b,p1; tie value u; max o1) and "bath" (7 lower inputs).
   `diag_bath*.py` (Sobol indices), `diag_top.py` (bath-averaged mass on a (u,o1) grid), `diag_linavg.py`
   (linear response of the band edge), `diag_qmc.py` (Sobol vs iid per configuration).
3. **Estimator** (`feyn.py`, `estimator.py`): every position configuration enumerated exactly (360 R23, 45 R12, 11
   blind cases), a fixed unscrambled Sobol net per configuration with a model-independent Cranley-Patterson shift
   (crc32 of the configuration name), budget shares 70/26/4 %. No tables fitted to any model.

## Status
Done (negative for the challenge). The estimator is deterministic, unbiased at large N, and general, but it is
mechanism-guided **quadrature**, not a compact deductive estimate. It beats sampling by ~1-3x on high-accuracy models
and only ties it on the low-accuracy ones.

## Results
Partition check (`python check_bias.py <model> 360000`, 4M MC inputs vs iid faces):
| model | MC R23 | faces R23 | MC R12 | faces R12 | MC B | blind rule |
|---|---|---|---|---|---|---|
| orig | 2.914+-0.009 % | 2.925+-0.008 % | 0.916+-0.005 % | 0.920+-0.002 % | 0.846 % | - |
| seed4 | 6.486+-0.013 % | 6.500+-0.007 % | 3.805+-0.010 % | 3.832+-0.008 % | 0.724 % | 0.716 % |

Structure of the integrand (orig, single runs):
| finding | number | script |
|---|---|---|
| share of Var(segment mass) from top vs bath | 0.56-0.63 / 0.42-0.44 | `diag_bath.py`, `diag_bath2.py orig 16000` |
| bath: sum of single-input first-order indices / bath variance | 0.975 (N=16k; at N=4k the estimate was noise: -0.02) | `diag_bath2.py` |
| bath: closed index of largest lower input alone / 4 largest | 0.03 / 0.35 (time position matters, not rank) | `diag_bath2.py` |
| bath-averaged mass E[F | config, u, o1] | smooth 2-D surface, ridge as o1 -> u | `diag_top.py orig 2 5 3`, `... 5 7 1` |
| delta vs the max o1 / vs one lower input (1-D scans) | near-linear, kinks at x_k ~ 0 and x_k ~ u | scratch scans (in this README only) |
| linear response of band edge averaged over the face law (adjacent mass ratio vs exact) | 3.71 (h=0.05, all coords), 1.90 (h=0.3), 0.95 (max only, h=0.3) | `diag_linavg.py orig 3000 h [max|bath|all]` |
| share of mass carried by the top 5% / 1% of faces | 35% / 15% (orig), 37% / 14% (seed1) | bandwidth data |
| Sobol vs iid per configuration (rel. s.d. at N=256) | ~1.5x lower s.d.; at whole-estimator level no gain | `diag_qmc.py`, `diag_rule.py` |

Estimator error over 12-32 random shifts of the rule (expected performance of the rule class; truths = leaderboard 4M):
`python m16/approaches/sci_feynman/diag_shift.py B reps names` and `python diag_rule.py B 32 iid sobol` (orig).
| model | 16k MSE/sampling | 64k MSE/sampling | 64k bias (pp) |
|---|---|---|---|
| orig | 0.36 (iid points 0.52) | 0.42-0.57 | +0.024..0.028 (+-0.009..0.017)* |
| s0.0025_3 | 0.49 | 0.73 | +0.038+-0.020 |
| seed4 | 0.86 | 0.45 | +0.082+-0.021** |
| s0.005_1 | 1.11 | 0.79 | +0.027+-0.038 |
| seed1 | 1.66 | 0.98 | +0.062+-0.040 |

\* The leaderboard truth for orig (0.95293) is ~0.025 pp below the 20M truth in `m16/truth.txt` (0.95318) and below
my 4M check (0.95324); against the 20M truth the bias is ~0.
\** The large-N face integrals for seed4 agree with MC (table above), so this is not a bias of the integral: the
segment mass is heavy-tailed, so a typical rule misses rare large faces and reads high in accuracy; the mean over
shifts converges slowly. A single fixed rule inherits this skew.

The fixed rule on all 18 leaderboard models (`python m16/approaches/sci_feynman/eval_all.py 16000|64000`, one run
each, `eval_*.json`): all-18 MSE/sampling 0.86 (16k), 0.62 (64k); per group 16k: orig 1.81, seeds 0.94, sigma 0.0025
0.98, sigma 0.005 0.63; 64k: 0.04 / 0.27 / 0.14 / 1.34 (s0.005_1 off by +0.30 pp). Mean error +0.07 / +0.04 pp
(same sign on most models: shared nodes, correlated errors). These single-draw group ratios are very noisy;
the shift table is the better guide.

## Compute accounting
Unit = one RNN forward pass on one sequence. A linear cell on a segment costs 2(10-t)/10 (value + tangent, recomputed
from the first step whose gate changed); the prefix before min(a,b) is charged once per segment (1/10 per step).
Blind points cost 1. Output-layer solves are counted as free. Model-free precomputation (face densities, Sobol nets,
shifts) is free and reused. Actual cost is returned (it is below budget on seeds 0, 1, 4: fewer cells per segment).

## Claims
- [measured] The partition plus exact walks reproduces MC truth for every part (orig, seed4), so the estimator has no
  detectable bias at large N.
- [measured] Low inputs act on the segment mass almost additively (first-order indices sum to 0.975 of their variance,
  one run), and the bath-averaged mass is a smooth function of (u, o1). Even so, Sobol nets barely beat iid points,
  because the per-face variance comes from rare large faces (top 5% of faces carry 35% of the mass) that no low-order
  structure captures.
- [measured] Linear response of the band edge (finite-difference slopes extrapolated over the face law) is useless:
  depending on the step it gives 0.95x to 3.7x the true adjacent mass. This agrees with H-E9 (the margin is a small
  residual of cancelling terms) and H-E10 (typical-value replacements are Jensen-biased).
- [measured] At equal compute the rule gains ~2-3x MSE on models above ~93% accuracy, and about nothing (0.8-1.7x
  worse to slightly better) on seed1 / s0.005_1 (82-88% accuracy).
- [inferred] Information bound for any face method: per-face relative variance ~CV^2 = 2-3.3 at ~7 passes per face,
  against ~(1-p)/p per pass for sampling. So faces alone give at most ~1.5-2x. The rest comes from exact
  stratification of positions and families, which is model-free, and so are most of the gains in E9, E12, E15 and E18.
- [speculative] A deductive estimator that beats sampling needs a description of the rare large bands (multi-cell bands
  at near-degenerate comparators), not of the typical band.

## Alternatives not ruled out
- The orig and s0.0025 gains could be mostly the model-free stratification (positions, families), as for stratified.py.
  Not separated beyond the iid-vs-Sobol comparison (iid within configurations already gives 0.52 at 16k on orig).
- Neyman shares from a deterministic pilot might help the low-accuracy models. Not tried.
- The skew argument for seed4's +0.08 pp is inferred from the agreement of the large-N integrals, not shown directly
  (e.g. with a reps-vs-bias curve).

## Failure modes
- Low-accuracy models: errors spread away from the tie (non-adjacent bands were 17-33% of segment mass in the
  bandwidth data), so per-face variance rises and the method falls to about sampling.
- A fixed node set gives errors that are correlated across similar models (same sign on most copies) and skewed
  toward overestimating accuracy.
- Not deductive in ARC's sense: it evaluates the model on fixed points. No compact mechanism-level formula was found.
