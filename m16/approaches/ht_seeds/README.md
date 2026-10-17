# ht_seeds (T2): does the H x T tie-slicing collapse replicate across training seeds?

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [TASKS](../../../TASKS.md) ·
[sci_tao (original scan, E25)](../sci_tao/README.md)

## Question
H-E11 (from sci_tao's E25 scan) says deterministic tie-slicing beats sampling by 38-507x while the error band is
regular, and collapses at T≈5 (H=8), 5-6 (H=16), 6-8 (H=32), with the collapse tracked by the band integrand's
tail share. That scan used ONE training seed per (H,T) (the best-of-5) and R=8 rule seeds (CIs ~x0.24-x2.3).
Does it hold for all five zoo training seeds per (H,T), with tighter CIs and an independent ground truth?

## Method
- **Models:** `alg_zoo.zoo_2nd_argmax(H, T, seed=s)`, s = 0..4, H in {8, 16, 32}, T in {3, 4, 5, 6, 8, 10}:
  90 models (full grid, nothing dropped).
- **Estimator:** sci_tao's deterministic Sobol tie-face rule, copied unchanged (`core.py`, `est.py`): 1024
  faces per kind (R23 and R12), **R = 32 rule seeds** (700..731; sci_tao used 700..707). Deterministic in the
  "no random model inputs" sense (numerical quadrature, not a heuristic argument).
- **iid control:** the same faces drawn i.i.d. (face-MC, R = 16), which removes the low-discrepancy benefit and
  leaves only the co-area variance reduction.
- **Tail share:** share of E[F^2] carried by the top 1% of faces (mean over rules); also computed on the
  independent iid faces. **Cancel** = median cancellation factor of the R23 tie margin (as sci_tao).
- **Ground truth:** plain sampling on its own RNG stream, n chosen so truth s.e. <= 0.3 x the rule s.d.
  (2M-32M samples per model; 8.6e8 in total), capped at 32M.
- **Gain** = sampling variance at equal cost / estimator MSE, p(1-p)/cost / MSE, with p = the independent truth.
  MSE includes bias: mean (est - truth)^2 - truth s.e.^2; 95% CI by bootstrap over rules and truth. Where the
  truth cannot resolve the MSE (truth s.e. > 0.5 x rule s.d.; all T=3 cells), the variance gain with a
  chi-square CI (31 d.o.f., ~x0.57-x1.56) is used and marked `*`; the bias z-score is still reported
  (|z| <= 2.7 everywhere, no systematic bias).
- **Budget check:** `budget_check.py` repeats 25 models (H=16 T=4,5,6; H=32 T=5,6) at 4096 faces per kind (4x cost).
- Files: `scan_seeds.py` (scan), `scan_seeds.json` (raw per-rule estimates, costs, tails, truths),
  `scan_seeds.log`, `analyse.py` -> `analyse.out` (full 90-row table and all statistics), `table.json`,
  `budget_check.py` -> `budget_n4096.json/.log`.

## Status
Done (2026-10-17).

## Results
Reproduce: `cd m16/approaches/ht_seeds; OMP_NUM_THREADS=1 python scan_seeds.py --R 32 --Riid 16 --cells 16:3 16:4 16:5 16:6 16:8 16:10 8:3 8:4 8:5 8:6 8:8 8:10 32:3 32:4 32:5 32:6 32:8 32:10`
then `python analyse.py > analyse.out` (bootstrap seed fixed) and
`OMP_NUM_THREADS=1 python budget_check.py --n 4096 --R 16 --cells 16:4 16:5 16:6 32:5 32:6`.
(The last log line, H=32 T=10 seed 4, was lost when git rewrote the log during the run; the row is in the json and
`analyse.out`.)

### Gain per (H, T, seed) [95% CI] (Sobol rule, ~6k-35k forward passes per run; acc = truth)

| H | T | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | acc range | tail mean [range] |
|---|---|---|---|---|---|---|---|---|
| 8 | 3 | 1571 [889,2444]* | 820 [464,1275]* | 2975 [1683,4629]* | 3782 [2140,5884]* | 399 [161,844] | .9992-.9995 | .23 [.16,.28] |
| 8 | 4 | 31 [17,63] | 25 [13,49] | 6.9 [4.4,11] | 23 [13,37] | 28 [16,44] | .972-.988 | .32 [.27,.46] |
| 8 | 5 | 2.2 [1.3,4.6] | 3.0 [2.0,5.0] | 2.1 [1.4,3.8] | 2.4 [1.6,4.0] | 2.3 [1.4,3.8] | .81-.88 | .31 [.25,.36] |
| 8 | 6 | 1.6 [1.1,2.4] | 1.0 [0.6,2.6] | 1.4 [1.0,2.3] | 1.3 [0.8,2.4] | 1.6 [1.0,3.1] | .70-.79 | .23 [.20,.27] |
| 8 | 8 | 0.7 [0.3,1.5] | 0.9 [0.6,1.5] | 0.9 [0.5,1.7] | 0.8 [0.6,1.2] | 0.6 [0.4,0.9] | .42-.47 | .15 [.14,.15] |
| 8 | 10 | 0.9 [0.6,1.7] | 0.6 [0.4,1.0] | 0.8 [0.6,1.2] | 0.6 [0.4,1.2] | 0.7 [0.4,1.5] | .35-.43 | .15 [.14,.15] |
| 16 | 3 | 1139 [645,1772]* | 870 [492,1353]* | 1931 [1092,3004]* | 1737 [983,2703]* | 957 [542,1489]* | .9994-.9995 | .22 [.14,.28] |
| 16 | 4 | 90 [46,188] | 109 [60,163] | 76 [41,134] | 56 [26,131] | 57 [28,114] | .9973-.9983 | .30 [.28,.36] |
| 16 | 5 | 11 [4.7,27] | 7.0 [4.2,11] | 4.7 [2.9,7.4] | 14 [7.4,24] | 15 [8.5,24] | .9873-.9926 | .43 [.33,.55] |
| 16 | 6 | 3.0 [2.0,4.8] | 4.0 [2.5,6.2] | 4.7 [2.9,7.7] | 3.0 [1.7,6.3] | 2.8 [1.6,5.4] | .970-.985 | .60 [.56,.69] |
| 16 | 8 | 2.2 [1.6,3.4] | 3.6 [2.3,6.6] | 2.8 [1.8,5.3] | 4.1 [2.8,6.3] | 1.5 [0.9,3.1] | .958-.977 | .62 [.52,.75] |
| 16 | 10 | 1.9 [1.2,3.1] | 0.7 [0.5,1.0] | 1.8 [1.2,2.9] | 2.3 [1.5,4.1] | 1.8 [1.3,2.8] | .88-.95 | .47 [.39,.59] |
| 32 | 3 | 478 [270,743]* | 951 [538,1480]* | 1348 [763,2098]* | 872 [493,1356]* | 454 [257,706]* | .9994-.9995 | .21 [.16,.27] |
| 32 | 4 | 74 [37,159] | 50 [27,77] | 30 [15,70] | 66 [34,131] | 64 [33,115] | .9985-.9988 | .32 [.30,.34] |
| 32 | 5 | 12 [5.7,23] | 7.7 [3.6,17] | 13 [6.9,26] | 34 [18,65] | 30 [15,66] | .9950-.9973 | .34 [.32,.40] |
| 32 | 6 | 4.0 [2.3,7.2] | 8.5 [5.0,13] | 4.8 [2.5,10] | 7.4 [4.2,12] | 4.0 [2.1,7.2] | .9919-.9939 | .49 [.43,.60] |
| 32 | 8 | 3.2 [1.8,6.3] | 4.7 [2.3,9.5] | 4.3 [2.3,7.5] | 2.1 [1.3,3.6] | 1.5 [1.0,3.0] | .9911-.9931 | .65 [.52,.79] |
| 32 | 10 | 0.9 [0.5,1.7] | 1.3 [0.9,2.2] | 2.4 [1.5,4.1] | 1.4 [1.0,2.3] | 2.3 [1.5,3.4] | .9908-.9944 | .79 [.64,.88] |

`*` = variance gain (truth too coarse to resolve the bias-inclusive MSE). Per-model tails, iid gains, Sobol/iid
ratios, bias z and cancellation factors: `analyse.out`.

**Geometric-mean gain over the 5 seeds:**

| H \ T | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| 8 | 1420 | 20 | 2.4 | 1.4 | 0.8 | 0.7 |
| 16 | 1260 | 75 | 9.4 | 3.4 | 2.7 | 1.6 |
| 32 | 753 | 54 | 16 | 5.5 | 2.9 | 1.5 |
| Sobol/iid (geo-mean, all H) | 6-8 | 2.5-5 | 1.4-4.5 | 1.5-3.7 | 1.9-3.8 | 1.5-3.2 |

**Collapse point (first T at which the gain CI lies wholly below 10x), per seed:**
- H=8: T=5 for all 5 seeds (seed 2 is already at 6.9x at T=4). But H=8, T>=5 models are undertrained
  (acc 0.35-0.88). Their gain goes to <= 1 (CI includes 1 from T=6-8), with LIGHT tails (0.15-0.31): bulk error.
- H=16: T=6 for seeds 0, 1, 3, 4; T=5 for seed 2. Seed 2 is the best-of-5 seed that sci_tao scanned, and it has
  the lowest T=5 gain of the five. All five seeds are >10x at T=4 (56-109x).
- H=32: T=6 for seeds 0, 4; T=8 for seeds 1, 2, 3 (their T=6 gains are 4.8-8.5x, with CIs straddling 10).
  All are >10x at T=4, and seeds 3 and 4 also at T=5.

**Same model as sci_tao (best-of-5 seed), R=8 then vs R=32 now:** consistent within sci_tao's CIs everywhere,
but two of its notable cells shrink:
- H=16, T=8 ("the exception", 8.2x at tail 0.78) is now 3.4x [1.9, 5.4]. It is no longer an exception.
- H=32, T=5 drops from 48x to 32x [18, 49]; H=32, T=10 from 3.4x to 0.9x [0.5, 1.4].

**H dependence (Mann-Whitney over seeds, `analyse.out`):**

| T | H16/H8 gain | H32/H16 gain |
|---|---|---|
| 3 | 0.9 | 0.6 (p=0.15) |
| 4 | 3.7 (p=0.008) | 0.7 (p=0.22) |
| 5 | 4.0 (p=0.008) | 1.7 (p=0.31) |
| 6 | 2.5 (p=0.008) | 1.6 (p=0.03) |
| 8 | 3.5 (p=0.008) | 1.1 (p=0.69) |
| 10 | 2.2 (p=0.06) | 1.0 (p=1.0) |

So the H=16 to H=32 shift is small: 1.6-1.7x at T=5-6 and none at T=8-10.

**Tail share as predictor (Spearman with log gain, 95% bootstrap CI over models):**

| set | tail | T | log(1-acc) | log cancel |
|---|---|---|---|---|
| all 90 | -0.13 [-0.41, +0.15] | -0.89 [-0.93, -0.82] | -0.87 [-0.93, -0.78] | +0.77 [+0.62, +0.89] |
| well-trained (acc > 0.97), n=62 | **-0.96 [-0.98, -0.91]** | -0.95 [-0.97, -0.92] | -0.77 [-0.86, -0.61] | +0.57 [+0.32, +0.77] |
| well-trained, pooled within T (fixed face dimension) | **-0.55 [-0.73, -0.34]** | - | -0.09 [-0.35, +0.17] | +0.23 [-0.00, +0.46] |
| same, tail from the independent iid faces | -0.60 [-0.77, -0.38] | - | - | - |
| undertrained (acc <= 0.97), n=28 | +0.75 [+0.46, +0.91] (wrong sign) | -0.46 | -0.81 | +0.77 |

Within the well-trained set at fixed T, the tail predicts:
- the iid (no-quadrature) gain: rho = -0.42 [-0.61, -0.19];
- but NOT the quadrature benefit Sobol/iid: rho = +0.05 [-0.18, +0.29].

**Budget dependence (4x faces, ~80k-140k passes; `budget_n4096.log`).** Geometric-mean gain ratio (4096 vs 1024
faces per kind):
- H=16: T=4 1.66 (80 -> 133x), T=5 1.48 (8.9 -> 13x), T=6 1.05;
- H=32: T=5 1.08, T=6 1.24.

The rule converges faster than n^-1/2 at T <= 5 (H=16). So the "collapse" T at a fixed 10x threshold drifts
upward with budget for H=16 (T=5 moves from 9x to 13x). At T=6 the gain is budget-independent (~3.5x at H=16).

## Verdict
1. **Collapse point per H:**
   - **H=16: confirmed at T=5-6, and the 10x crossing is at T=6 for 4 of 5 seeds.** The T=5 cell straddles 10x:
     4.7-15x by seed, geo-mean 9.4, 13 at 4x budget.
   - **H=32: confirmed at T=6-8** (seed-dependent: T=6 for 2 seeds, T=8 for 3).
   - **H=8: T=5 replicates, but it is an undertraining collapse** (acc 0.81-0.88 at T=5), not the heavy-tail one.
   - The collapse is **seed-dependent by about one T step** at H=16/32. Gains at fixed (H,T) vary 2-8x across
     seeds (e.g. H=32, T=5: 7.7-34x).
2. **It is a gradual decline, not a sharp transition.** For every H the gain falls by ~15-20x per T step from
   T=3 to 5, then ~2x per step down to ~1-2x at T=10. "Collapse T" depends on the threshold and budget.
3. **The H dependence is weaker than E25 suggested.** H=32 beats H=16 by only 1.6-1.7x at T=5-6 and not at
   T>=8. E25's 48x at H=32, T=5 was a favourable seed plus R=8 noise (now 32x; seeds 7.7-34x).
4. **Tail share is still the predictor among well-trained models** (rho = -0.96 [-0.98, -0.91]), and it keeps
   predictive power at fixed T (pooled within-T rho = -0.55 [-0.73, -0.34]) where accuracy does not (-0.09).
   It fails, with the wrong sign, on undertrained models (bulk errors, light tails).

## Compute accounting
- Unit = one forward pass (FPE). Estimator costs are counted as in sci_tao: a walk cell = 2(T - restart)/T,
  pilot-free equal split.
- Per rule run: 6k-35k FPE (rises with H, falls with T).
- Totals:
  - scan: 90 models x (32 Sobol + 16 iid) runs = 8.7e7 FPE;
  - truths: 8.55e8 sampled sequences (plain forward passes);
  - budget check: 25 x 16 runs ~ 4e7 FPE;
  - plus cancellation Jacobians (2048 faces per model, negligible).
- Wall time: ~65 min on 1 core (paused automatically when load > 3.5).
- Model-free tables and Sobol points are free and shared across models.

## Claims
- [measured] Tie-slicing gain > 10x at T <= 4 on 20/20 models with H >= 16 (30-1931x), and on 9/10 of H=8, T<=4
  (6.9x for H=8 T=4 seed 2).
- [measured] At T=6 all 15 models have point gains <= 8.5x, and 12/15 have a CI wholly below 10x (the exceptions
  are H=32 seeds 1-3). At T=10 the gains are 0.6-2.4x.
- [measured] Per-H collapse points (first T with gain CI < 10x): H=16 T=6 (4 seeds) / T=5 (1 seed); H=32 T=6 (2)
  / T=8 (3); H=8 T=5 (5), the last for undertrained models.
- [measured] sci_tao's R=8 numbers on the same models replicate within their CIs. The H=16 T=8 "exception" (8.2x)
  does not survive R=32 (3.4x).
- [measured] Among acc > 0.97 models, tail share rank-correlates with log gain at -0.96 [-0.98, -0.91], and at
  -0.55 [-0.73, -0.34] within fixed T; accuracy within T: -0.09 [-0.35, +0.17].
- [measured] Tail share does not predict the Sobol/iid quadrature benefit within T (rho +0.05). It predicts the
  co-area (iid) variance part.
- [inferred] The decline is a property of the band integrand (it also hits the iid face-MC gain, which has no
  dimension-dependent quadrature rate), not mainly a QMC dimension effect. Evidence: the iid gain falls from
  67-361x (T=3) to 0.3-3.6x (T>=6) too.
- [inferred] Part of the tail-gain link is close to mechanical: a heavier top-1% share of E[F^2] at a given
  E[F] means a larger Var(F), which lowers the iid gain by definition. The tail is a descriptor of why the
  variance is high, not an independent cause.
- [speculative] The ~1-step seed-to-seed spread of the collapse T reflects how many rare wide multi-cell bands
  each training run leaves (H-E12). Not tested.

## Alternatives not ruled out
1. **Quadrature dimension grows with T (face dimension T-1).**
   - Test: compare Sobol with iid faces, and correlate within fixed T.
   - The Sobol/iid ratio falls from 6-8 (T=3) to 1.4-4 (T>=5), so part of the large small-T gain is
     low-dimensional QMC.
   - But the iid gain also collapses with T, and the tail predicts gain within fixed T. So dimension is not the
     whole story.
   - Not done: a dimension-matched control (e.g. T=6 with frozen extra coordinates).
2. **Seed accuracy differences.**
   - Within fixed T among well-trained models, log(1-acc) does not rank the gain (pooled rho = -0.09), while the
     tail does (-0.55).
   - Across T, accuracy and T are confounded (both rho ~ -0.8 to -0.95). Accuracy alone cannot be separated from
     T with this grid.
   - Undertrained models (H=8, T>=5) fail differently (bulk errors, light tails, gain ~1).
3. **Budget/cost confound.**
   - Runs use a fixed face count, so cost differs: 6k-35k FPE; H=32 runs cost ~1.7x H=16 runs.
   - QMC gain grows with budget at T<=5 (x1.5-1.7 for 4x at H=16), so part of the H=32 vs H=16 difference at
     T=5 could be budget.
   - At 4x budget H=16 T=5 rises to 13x, H=32 T=5 to 19x: the H=32 advantage shrinks to ~1.4x.
   - The collapse T for a fixed threshold therefore drifts with budget. A threshold-free statement is the
     per-step decline factor.
4. **Tail share is partly a restatement of Var(F)** (see Claims). A predictor computed without the walk (e.g.
   from weights or gate statistics) would be a stronger test. Not done.
5. **Truth limits at T=3:** 32M-sample truths cannot resolve the bias-inclusive MSE there, so T=3 gains are
   variance gains. The bias z-scores (|z| <= 2.0 at T=3) do not show a bias.

## Failure modes
- Undertrained models: tie slicing gives ~1x because errors are bulk, not thin bands.
- The fixed-face-count design makes "equal cost" differ across cells; gains at a fixed threshold depend on budget.
- 5 seeds per cell gives seed-level spreads but weak per-H tests (Mann-Whitney with n=5 per group has a
  minimum p of 0.008).
- The lost log line is an artifact of committing a file still being written (shared checkout). The JSON is complete.

## Proposed follow-ups (added to the TASKS backlog)
- Threshold-free collapse metric: per-T decline factor of gain at matched budget (equal FPE, not equal faces).
- A walk-free tail predictor (from weights/gates), to test whether the tail is more than a variance proxy.
