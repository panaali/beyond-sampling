# sci_tao: structure vs randomness on the tie faces, small cases first

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [STATUS](../../../STATUS.md)

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


**Follow-up: is the T=4 -> 5 collapse structural or a quadrature-dimension effect?** (`python scan_HT.py 1024 8 8:3 ... 32:10`,
results in `scan_HT.json` / `scan_HT.log`)
- Models: best-of-5 zoo seeds, H in {8, 16, 32}, T in {3, 4, 5, 6, 8, 10}.
- Rules: 1024 faces per kind, 8 rule seeds.
- Gain = MSE gain over sampling at equal cost. The 95% CI is chi-square on 7 d.o.f., about x0.24 to x2.3.
- iid = the same faces drawn at random, so the quadrature (low-discrepancy) benefit is removed.
- Tail = share of E[F^2] carried by the top 1% of faces.
- Cancel = median cancellation factor of the tie margin.

| H \ T | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| 8: acc | .9995 | .9875 | .8815 | .7899 | .4656 | .4262 |
| 8: Sobol gain [CI] | 550 [133,1259] | 13 [3,31] | 2.0 [0.5,4.5] | 0.8 [0.2,1.9] | 0.6 [0.2,1.5] | 0.5 [0.1,1.2] |
| 8: iid gain / tail / cancel | 78 / .29 / 1700 | 6.3 / .27 / 100 | 0.9 / .34 / 21 | 0.5 / .28 / 22 | 0.1 / .15 / 7 | 0.3 / .13 / 8 |
| 16: acc | .9995 | .9983 | .9925 | .9842 | .9762 | .9488 |
| 16: Sobol gain [CI] | 487 [118,1113] | 57 [14,130] | 7.7 [1.9,18] | 2.1 [0.5,4.7] | 8.2 [2.0,19] | 1.3 [0.3,3.1] |
| 16: iid gain / tail / cancel | 104 / .26 / 2454 | 17 / .33 / 930 | 5.7 / .53 / 323 | 1.2 / .68 / 217 | 1.0 / .78 / 172 | 0.5 / .60 / 86 |
| 32: acc | .9995 | .9988 | .9973 | .9937 | .9931 | .9947 |
| 32: Sobol gain [CI] | 1533 [370,3506] | 38 [9,88] | 48 [12,110] | 9.5 [2.3,22] | 1.0 [0.2,2.2] | 3.4 [0.8,7.8] |
| 32: iid gain / tail / cancel | 158 / .25 / 1888 | 28 / .35 / 1024 | 28 / .32 / 761 | 2.4 / .47 / 396 | 1.0 / .79 / 1013 | 0.8 / .86 / 1732 |

**Reading.**
- **Not a pure quadrature-dimension effect.** At a fixed T (same face dimension), the gain depends strongly
  on H.
- The collapse point moves with H:
  - H=8 collapses at T=4-5;
  - H=16 at T=5-6;
  - H=32 holds to T=5 (48x) and collapses by T=8.
- So the collapse depends on the model, not only on dimension [measured, CIs wide].
- **Quadrature still matters.** Sobol gives 5-10x more than iid faces at T=3, but at most about 1-4x from
  T >= 5. Most of the large small-T gain is low-dimensional quadrature acting on a regular integrand.
- **The cancellation factor does not track the collapse.** H=32, T=10 has cancellation 1732 and has
  collapsed; H=8, T=4 has 100 and still gains 13x.
- **The tail statistic tracks it among well-trained models (acc > 0.97).** Gain >= 38x always goes with
  tail <= 0.35; gain <= 9.5x with tail >= 0.47, except H=16, T=8 (8.2x at tail 0.78, CI 2-19).
- Undertrained models (H=8, T >= 5; acc 0.43-0.88) lose for a different reason. Their errors are bulk,
  not thin tie bands, so tie slicing does not help even though their tails are light.
- [inferred] The collapse marks where the network's error set stops being a regular thin band and becomes
  a heavy-tailed scatter of wide bands (or, for weak models, bulk error). It happens at larger T when H is
  larger, so it is structural.
- Untested alternatives:
  - only one training seed per (H,T);
  - R=8 gives wide CIs;
  - a tail-aware allocation might move the boundary.

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
