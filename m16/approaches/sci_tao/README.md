# sci_tao: structure vs randomness on the tie faces, small cases first

## Question
Can the accuracy of M_16,10 be estimated without random model inputs, by computing the "structured" part
exactly and integrating the rest deterministically? And where, as the sequence length T grows, does
structure give way to pseudorandomness?

## Method
1. **Exact co-area identity (model-free geometry), `core.py`.** R23 = {v2-v3 < v1-v2}, R12 = the rest. Every
   input lies on exactly one segment through a tie face {x_a = x_b} in direction (e_a-e_b)/sqrt2, so
   P(wrong) = Z23 E_face23[F] + Z12 E_face12[F]. F is the phi-mass of the misclassified part of the segment.
   Unlike tube (E15), there is no blind-region cut: B is sliced the same way.
   Z and the face laws are 1-D model-free tables. Check: Z23 E[window] + Z12 E[window] = 1.0000 for
   T = 2, 3, 5, 10 (`check_core.py`).
2. **F computed exactly.** The walk follows the linear pieces along each segment: value plus tangent, cost
   counted from the step whose gate flipped. In each piece the correct set is solved in closed form, with
   numpy's first-index argmax tie rule. Matches a 40k-point grid to 8e-6 (`check_core.py`).
3. **Deterministic face rule** (`est.py`, `estimator.py`):
   - a fixed scrambled-Sobol rule (the seed is part of the rule and is the same for every model);
   - position configurations stratified uniformly by exchangeability (no per-model tables);
   - a deterministic two-stage R23/R12 Neyman split, with the pilot counted in the cost.
   `DEDUCTIVE = True` in the "no random inputs" sense. It is *numerical quadrature*, not a heuristic
   argument (see Claims).

## Status
Done. Estimator works unchanged on any (H,T) model. It does not solve Q4 in ARC's strict sense.

## Results
**Small cases / T-scan** (`python small.py`, `python scan_T.py 1024 8`, `python rate_T.py 6 ...`).
MSE gain = sampling variance at equal cost / spread over fixed rule seeds.

| model | truth | gain at ~20k cost |
|---|---|---|
| M_2,2 | 0.99992 | exact (err 0.000 pp) |
| M_4,3 | 0.98444 | err -0.008 pp at 10k (sampling sd 0.12) |
| M_16,3 | 0.99951 | 507x |
| M_16,4 (3 seeds) | 0.9977-0.9983 | 38-266x |
| M_16,5 (3 seeds) | 0.9901-0.9925 | 3-264x (seed-dependent, mostly 3-30) |
| M_16,6 / 8 | 0.985 / 0.976 | 3.4x / 4.6x |
| M_16,10 orig | 0.9529 | 2.5x |

The deterministic rule's advantage collapses between T=4 and T=5. Face dimension is T-1. From T=5 on, F is
heavy-tailed: the top 1% of faces carry 50-76% of E[F^2] (`diag_var.py`).

**Leaderboard models, 18 = orig + seeds 0-4 + 12 noisy** (`python run_rules.py 6 16000 64000`: 6 fixed
rules per model, bootstrap CI over models x rules; truths from `results/leaderboard_truths.json`):

| budget | group | RMSE | sampling RMSE | MSE ratio (est/samp) [95% CI] | mean err |
|---|---|---|---|---|---|
| 16k | all 18 | 0.187 pp | 0.216 pp | 0.75 [0.45, 1.10] | +0.074 pp |
| 16k | seeds | 0.200 | 0.223 | 0.80 [0.36, 1.35] | +0.116 |
| 16k | s0.005 | 0.229 | 0.238 | 0.92 [0.50, 1.59] | +0.079 |
| 64k | all 18 | 0.084 pp | 0.108 pp | 0.60 [0.33, 0.95] | +0.031 pp |
| 64k | orig / seeds / s.0025 / s.005 | 0.068 / 0.084 / 0.065 / 0.100 | 0.084 / 0.111 / 0.097 / 0.119 | 0.66 / 0.57 / 0.46 / 0.71 | |

Single fixed rule (`python run_eval.py 16000 64000`, `eval_*.json`): at 64k seeds lose (1.44); everything
else wins.

**Mean error is positive: a skew effect from the heavy tail, not a floor.** With few faces, the rare
large-F faces are under-represented, so a typical run reads accuracy high. It shrinks from +0.074 to +0.031
pp between 16k and 64k. seed1 is the worst case: 4 rules at 4096 faces give 0.8815 +- 0.0014 (s.d.)
against an independent truth of 0.87989.

**Cancellation vs T** (`python cancel_T.py`), for INDEX H-E9: the median cancellation factor of the tie
margin, sum|J_j y_j| / |m|, *falls* with T: 2550 (T=3), 930 (4), 313 (5), 224 (6), 166 (8), 110 (10). So
heavy cancellation alone does not make the margin pseudorandom: at T <= 4 it is exact and structured. What
grows with T is the s.d. of the residual (0.52 -> 2.68) and its irregularity.

## Compute accounting
- Unit = 1 forward pass. A walk cell = 2(T - restart step)/T. The prefix before min(a,b) is counted once.
  Pilot faces are counted.
- Model-free tables (face CDFs, Z) and Sobol points are free and reused across all models.
- Root-finding and output-layer work are counted as negligible.
- Experiments total about 1e8 pass-equivalents, mostly the T-scans and 6-rule leaderboard repeats.

## Claims
- [measured] The co-area identity is exact, and the walk is exact. On T <= 4, deterministic quadrature
  beats sampling by 40-500x in MSE: the deductive estimate "works" on small models.
- [measured] For T >= 5 the gain falls to a constant 2-5x. On the 18 leaderboard models it is 1.3x (16k,
  CI includes 1) and 1.7x (64k, CI 0.33-0.95 for the MSE ratio). That is the same class as tube, stratified
  and joint.
- [inferred] The limit is the pseudorandom part of F (heavy tail, roughness on scales 0.05-1, H-E9's
  cancellation residual). No fixed-dimension structure found here captures it. The method integrates the
  low inputs distributionally (cf. H-E10), so it has no Jensen bias, but it pays for that with ~sampling
  rates.
- [inferred] A deterministic Sobol rule over faces is "deductive" only in the weak sense. It is not a
  mechanistic or heuristic argument, so Q4 is **not solved** in ARC's sense.

## Alternatives not ruled out
- The T=4 -> 5 collapse could be a QMC dimension effect (3 vs 4 face dims) rather than a structural change.
  Against this: CV(F) and tail weight jump at T=5, and the gain varies 3-264x across T=5 seeds. Untested:
  the same scan at H=8 or H=32.
- The 64k win could shrink with more rule repeats: the CIs are wide (6 rules, 18 models).
- Allocating by a heavy-tail indicator could help, but the large-F faces were not identifiable by tie
  value u on seed1 (`diag_var.py`, inline check).

## Failure modes
- Heavy-tailed models (seeds 1, 4; s0.005_1) give skewed per-rule errors, positive on average at low budget.
- The cost per face grows with gate flips along segments.
- Not a closed form: cost grows linearly with the precision demanded, like sampling.
