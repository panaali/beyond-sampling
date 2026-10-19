# T394: independent check of T345 / T387 (one-hidden-layer 128-unit MLP, comparators, fixed-Halton great circles)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

Owner: Claude child session_01H3d57ZejhjZM73Jb3Mwccm (parent DRI session_01CD3g9Cvc3qk3pM11eyXXzj), branch
`claude/t394-ffn-check`. Checked claims: INDEX T345, T387 (branch `origin/claude/t345-ffn-control`, commit e47c52b2).

## Question

Do T345/T387's three D1w128 results hold on fresh models with independent code?
1. A bias-free 10-128-10 ReLU MLP trained with T345's recipe reaches >= .95 accuracy.
2. 84-94 of its 128 hidden units are pairwise comparators relu(x_a - x_b), covering all 45 pairs.
3. A fixed scrambled-Halton set of great circles with exact walks gives F < 1 (T387: .03-.28, mean .15) at C ~4k FPE.

## Pre-registration (pushed at 84c576c2; written and pushed before any training or scoring, 2026-10-19)

### Models
- Architecture and recipe exactly as T345's README / `train.py` state: bias-free Linear(10,128)-ReLU-Linear(128,10),
  AdamW (wd .001, lr .04/sqrt(128)), 2^14 warm-up steps, exponential decay to .01x (alg_zoo's `lr_mult_schedule`),
  batch 1024, cross-entropy, label `x.argsort(-1)[..., -2]`, x ~ N(0, I_10), torch default init, one thread.
  n_train = 2^28 (the schedule of the 12-model panel on which T387's F .03-.28 was measured).
  My training script `train.py` is written from that description (not copied).
- **Fresh seeds, fixed now: 101, 102, 103, 104, 105, 106** (6 models; T345 used 0-2). Every model is kept in the
  panel whatever its accuracy; no retries, no replacements.
- Accuracy: 1e7 fresh inputs (numpy, seed 39401), with binomial SE. Claim 1 holds if all 6 reach >= .95.

### Comparator count (rule fixed now)
- Normalise each W1 row to unit L2 norm. Unit k is a **comparator** iff its largest entry is > .6 and its smallest
  entry is < -.6 (one strong +, one strong -; the ideal (e_a - e_b)/sqrt 2 has +-.707). Pair = {argmax, argmin}.
- Reported: count per model, number of distinct unordered pairs covered (of 45), and for comparison T345's own
  rule (2nd-largest |normalised entry| > .6, no sign check). Claim 2 holds if counts fall in about 84-94 and all
  45 pairs are covered on every model; I report the numbers either way.

### Exact great-circle walk (own code, `walk.py`)
- x(t) = u cos t + v sin t, u, v in R^10; for u, v ~ N(0, I) every x(t) ~ N(0, I), so the exact correct fraction of
  the circle is an unbiased estimate of accuracy. For depth 1, logits on the circle are piecewise A cos t + B sin t.
- Events: hidden unit k is on on a half-circle centred at phi_k = atan2(c_k, a_k) (a = W1 u, c = W1 v), so it
  switches on/off at phi_k -+ pi/2; the target changes where x_i = x_j (2 angles per pair); extra breakpoints at
  0, pi/2, pi, 3pi/2 so every arc is shorter than pi. Between events the gates and target are fixed; A, B are
  updated by a rank-1 term per gate event; the target is tracked by rank swaps at pair-crossing events.
  On each arc, "logit_t - logit_k > 0" is a sinusoid with at most one zero on the arc; the correct set on the arc
  is the intersection over k of those positive parts (one interval). Zeros (atan2) are computed only for (arc, k)
  with a sign change between the arc's ends, and only those are charged.
- Validation before scoring: (a) dense brute force on >= 20 random circles (1e5 equally spaced angles through the
  plain forward pass), agreement to the grid's resolution; (b) the mean over 2e4 random circles agrees with the
  1e8-sample truth within 3 SE on each model.
- **Cost (charged as executed, FLOPs; trig/atan2 = 20 FLOPs, as T345):** W1 u, W1 v (4hT); h atan2; 45 pair
  differences x2 and 45 atan2; sort (n log2 n); cos+sin at every event angle; initial gates and A, B at t = 0
  (2hT·2 + 2h); per gate event 4T (rank-1 update of A and B); per pair event 2 (rank swap); per arc T (target
  lookup) and 4T per end for the 9+1 logit differences at both ends (8T); per sign change one atan2 + 4; per arc
  6T for the interval intersection. FPE = forward FLOPs of the net = 2·(10·128 + 128·10) + 128 = 5,248.
  If numpy does work beyond this count (vectorised redundancy), it is listed in the code and in Compute accounting.

### Scoring
- Truths: 1e8-sample plain MC per model (numpy, seed 39402, chunked), never seen by any estimator choice.
- Number of circles per fixed set: n = round(4000 / c), c = mean executed cost (FPE) per circle over 200 random
  circles (seed 39404) on model s101, measured before any truth is computed. C_exec = n·(cost per circle as
  executed on that model); C_eff = max(C_exec, 1000).
- **Fixed sets (identical for every model):** q = first n points of `scipy.stats.qmc.Halton(d=20, scramble=True,
  seed=s)`, (u, v) = (Phi^-1(q_1..10), Phi^-1(q_11..20)).
  - Primary: **seeds 11, 12, 13, 14, 15** (new, so the check does not reuse T387's sets).
  - Secondary: seeds 1-5 (T387's own seeds), for direct comparison.
- F per set = mean_m (e_m^2 · C_eff,m) / mean_m p_m(1 - p_m) over the 6 models, e_m = estimate - truth.
  Reported per set, with the 5-set mean/min/max per family, RMS and mean error.
- Random-circle comparison (sampling-class): per model 2e4 random circles (seed 39403); F_rand = mean_m(Var_m · c_m)
  / mean_m p_m(1-p_m), c_m = mean executed cost per circle (FPE). Also plain MC control F = 1 by construction.
- **Go (claim 3):** all 5 primary Halton sets give F < 1 on the fresh 6-model panel. Reported with the 5-set spread.
- Classification: deterministic quadrature over (u, v) (QMC-class, as T387), not deductive in ARC's sense.

### Alternatives / what could make this check misleading (stated in advance)
- A fixed set's error is shared by models with similar weights; fresh retrains are less correlated than T387's
  noised copies, so the 6-model panel should spread more. The 5-set spread is the main uncertainty measure.
- If my walk cost differs from `fast1.py`'s (17.2 FPE/circle) n will differ from 232; F compares at equal C, not equal n.
- If seeds fail .95, F may be worse (T345: F rises as accuracy falls); kept in the panel anyway.

## Method
Key files (all written for T394; `fast1.py` / `cone_test.py` of T345 were read, not copied or imported):
- `train.py`: D1w128 training from T345's stated recipe (seed, optional log2 n_train). `models/s<seed>.json`.
- `walk.py`: exact great-circle walk (`circle`), plain forward pass, MC accuracy, brute-force grid check (`brute`).
  Structure: one sorted event table (gate flips, pair crossings, quadrant breaks); A, B by cumulative rank-1 updates
  over gate events; the target by cumulative rank swaps over pair events; per arc, logit differences at the two
  ends, an atan2 only where a difference changes sign, then a max/min interval intersection.
- `score.py`: the pre-registered stages (`acc`, `n`, `validate`, `truth`, `random`, `fixed`), outputs in `results/`.

## Status
Done (2026-10-19). All stages ran as pre-registered; no deviations except chunk sizes (memory cap).

## Results

### Accuracy (1e7 inputs, seed 39401; `python score.py acc` -> results/models.json) and 1e8 truths (seed 39402)

| seed | 101 | 102 | 103 | 104 | 105 | 106 |
|---|---|---|---|---|---|---|
| acc (1e7, SE ~7e-5) | .9603 | .9488 | .9385 | .9659 | .9461 | .9589 |
| truth (1e8) | .96032 | .94878 | .93849 | .96601 | .94614 | .95896 |

**Claim 1 not replicated at 2^28:** 3 of 6 fresh seeds are below .95 (.939-.949), range .939-.966 vs T345's
.954-.965 on 3 seeds. "D1w128 reaches >= .95 on all seeds" was a 3-seed result at this schedule; the threshold sits
inside the seed spread. (T345's 2^30 retrains reached .990; I did not retrain at 2^30.)

### Comparators (pre-registered rule: normalised row max > .6 and min < -.6)

| seed | 101 | 102 | 103 | 104 | 105 | 106 |
|---|---|---|---|---|---|---|
| comparators (my rule) | 76 | 75 | 74 | 80 | 78 | 77 |
| pairs covered (of 45) | 45 | 45 | 45 | 45 | 45 | 45 |
| T345's rule (2nd-largest abs entry > .6, no sign check) | 96 | 87 | 94 | 99 | 89 | 91 |

- Median distance of a comparator row from the ideal (e_a - e_b)/sqrt 2: .004-.006; max .02-.05 (one unit of s103: .51).
- **T345's rule over-counts.** The units it adds (20 on s101 and s103) have two strong entries of the SAME sign,
  e.g. sorted (-.64, -.62, +.10 ... +.20): T345's own "both low" detectors, not comparators. The sign-checked rule is
  stable under the threshold (.5 instead of .6: 76, 75 on s101, s103).
- On T345's own 2^28 models (s0-s2), my rule gives **77, 76, 76**, and T345's rule reproduces its 94, 88, 84.
  So the corrected claim is: **74-80 of 128 units are comparators, covering all 45 pairs** (9 models: 6 fresh + 3 of
  T345's). The rest (48-54) are mostly "both low" units.
- Every row sums to ~0 (|sum| / norm <= .008), as T345 found.

### Walk validation
- Brute force: 20 random circles per model, 1e6-point grid: max |exact - grid| 1.9e-6 to 3.9e-6 (`results/validate.json`),
  i.e. the grid's resolution. On T345's s0 model (development check) 1.2e-5 at a 2e5-point grid.
- Unbiasedness: the mean of 2e4 random circles vs the 1e8 truth, z = 1.26, -.66, .35, .22, 1.91, .24 (`results/random.json`).
- Cost as executed: 17.61 FPE per circle (T345's `fast1.py`: 17.2). Pre-registered n = round(4000 / 17.61) = **227**
  circles per set (`results/n.json`); executed C = 3,996-3,997 FPE.

### F at C ~4k FPE (6-model fresh panel; plain sampling = 1)

`python score.py random`, `python score.py fixed` (results/random.json, results/fixed.json). Truth SE ~2e-5 (F floor ~4e-5).

| design | F of the 5 sets | mean | min-max | all 5 < 1 |
|---|---|---|---|---|
| **Halton seeds 11-15 (primary, new)** | **.074, .079, .635, .663, .390** | **.37** | **.07-.66** | **yes (go)** |
| Halton seeds 1-5 (T387's seeds) | .089, .174, .286, .347, .057 | .19 | .06-.35 | yes |
| random circles (sampling-class), 2e4 per model | F_rand = **.33** (T345: .28) | | | |

- **Claim 3 replicated (go):** every fixed Halton set gives F < 1 on fresh models with independent code; worst set .66.
- **Not replicated: "Halton is ~2x better than random circles".** On the fresh panel the 10 Halton sets average .28
  vs random .33; primary .37 is worse than random, secondary .19 better. A fixed set's error is largely shared across
  the 6 models (e.g. Halton 4: errors +.0024, +.0024, +.0000, +.0006, +.0030, +.0015), so each set acts like ~1-2
  draws and F per set scatters like F_rand times a chi-square with few degrees of freedom. The 10-set data fit
  "Halton ~ random circles" at n = 227 in 20 dimensions [inferred].
- Seed 1-5 numbers differ from T387's (.03, .28, .19, .10, .14) because n (227 vs 232) and the models differ.

## Compute accounting
- Training: 6 runs x ~17 min (6 in parallel on 4 cores), 2^28 samples each; not counted in F.
- Truths: 6 x 1e8 forward passes (11 min); accuracy 6 x 1e7.
- Walk FLOPs, as executed (pre-registered list): per circle ~92k FLOPs = 17.6 FPE, of which cos/sin at ~350 event
  angles 14k, per-arc logit-difference evaluation 28k, interval intersection 21k, rank-1 updates 10k. Not charged:
  the event argsort beyond n log2 n and index bookkeeping (no arithmetic); the rank cumsum is dense (T adds per pair
  event, charged as T).
- Scoring: 10 sets x 6 models x 227 circles; random 6 x 2e4 circles.

## Claims
- [measured] At T345's recipe and 2^28 samples, fresh D1w128 seeds reach .939-.966; 3 of 6 are below .95.
- [measured] 74-80 of 128 hidden units are sign-checked comparators relu(x_a - x_b), covering all 45 pairs, on 6 fresh
  seeds and 76-77 on T345's 3 seeds; T345's 84-94 counts same-sign "both low" units as comparators.
- [measured] Own exact walk matches dense brute force to <= 4e-6 and is unbiased against 1e8 truths.
- [measured] Fixed scrambled-Halton sets at C ~4k FPE: F .07-.66 (5 new seeds, mean .37) and .06-.35 (T387's seeds,
  mean .19); all 10 < 1. Random circles F .33.
- [inferred] At this n, fixed Halton sets are not clearly better than random circles; their F < 1 comes from the
  exact walk (random circles already give .33), not from the low-discrepancy design.

## Alternatives not ruled out
- At 2^30 training (accuracy ~.99) Halton might separate from random (T345 post-hoc: Halton .045 vs random .104); not run.
- 6 models with errors shared across models: the per-set F is noisy; more models or sets would tighten the
  Halton-vs-random comparison.
- My FLOP list is close to `fast1.py`'s (17.6 vs 17.2 FPE); a cheaper or more expensive charge rescales F linearly.

## Failure modes
- Events closer than float resolution (exact ties in u or v) could misplace a sliver; not seen (grid check).
- The fixed set's F depends strongly on the set (x9 spread among 5 seeds); a single set can be unlucky.
- Accuracy near the .95 threshold depends on the seed, so "reaches .95" claims need many seeds.
