# T394: independent check of T345 / T387 (one-hidden-layer 128-unit MLP, comparators, fixed-Halton great circles)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

Owner: Claude child session_01H3d57ZejhjZM73Jb3Mwccm (parent DRI session_01CD3g9Cvc3qk3pM11eyXXzj), branch
`claude/t394-ffn-check`. Checked claims: INDEX T345, T387 (branch `origin/claude/t345-ffn-control`, commit e47c52b2).

## Question

Do T345/T387's three D1w128 results hold on fresh models with independent code?
1. A bias-free 10-128-10 ReLU MLP trained with T345's recipe reaches >= .95 accuracy.
2. 84-94 of its 128 hidden units are pairwise comparators relu(x_a - x_b), covering all 45 pairs.
3. A fixed scrambled-Halton set of great circles with exact walks gives F < 1 (T387: .03-.28, mean .15) at C ~4k FPE.

## Pre-registration (written and pushed before any training or scoring, 2026-10-19)

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
(filled after the runs)

## Status
Pre-registered; not yet run.

## Results
(none yet)

## Compute accounting
(after the runs)

## Claims
(after the runs)

## Alternatives not ruled out
(after the runs)

## Failure modes
(after the runs)
