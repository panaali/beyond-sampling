# T63: an accuracy estimate for M_4,3 derived from its mechanism

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [ARC's estimators, reproduced (T36)](../arc_m43_repro/README.md) · [labels of M_4,3](../../../algzoo/labels_4_3.py)

## Question

Can the verified description of M_4,3 = `alg_zoo.zoo_2nd_argmax(4,3)` (two mirror pairs of neurons, a soft
"x1 vs x0" switch, readout by ratios within a pair) be turned into an accuracy estimate whose every coefficient is a
formula of the weights (no fitting to model outputs, no enumeration of all linear regions), and does that estimate
beat plain sampling at equal compute (MSE ratio < 1) under ARC's protocol? ARC's own understanding-based estimators
scored MSE ratios of about 10 (A) and 2.4 (B) at ARC's op counts ([T36](../arc_m43_repro/README.md)).

**Short answer:** yes on ARC's protocol. The default estimator (tier B) has RMSE 0.32% and MSE ratio **0.019**
(about 50x better than sampling at equal ops; about 3x lower RMSE than sampling at 13.7k ops). A held-out set of
512 fresh noisy copies gives 0.29% and ratio 0.016. The account does **not** transfer as-is to the other trained
seeds 1-4, which use different mechanisms. There tier B is off by up to 7 points; a more generic tier C stays within
1.5 points. It also costs about as much as brute-force enumeration (17.5k ops, zero error) on this tiny model.

## What the network does and why it errs (plain language)

The model reads x0, x1, x2 one at a time. It has no biases, so only the direction of x matters. Every region
below is a cone through the origin, and its probability is a solid angle.

1. **Step 1 (store x0).** The sign of x0 picks a mirror half. If x0 > 0, neurons 0 and 1 hold x0 (the "u" pair). If
   x0 < 0, neurons 3 and 2 hold |x0| (the "v" pair). Everything after this is the same picture mirrored
   (x -> -x swaps u and v and leaves the median unchanged). We call the two halves "charts".
2. **Step 2 (the x1-vs-x0 switch).** The four step-2 neurons switch on the planes x1 = t_k x0, with
   t_k = -(W_hh relu(Wi sign x0))_k / Wi_k. For the trained model these are t = 1.268, 0.807, 0.788 and 1.239. They cut each chart into
   5 strips of the ratio x1/x0. The middle strip, 0.807 < x1/x0 < 1.239, contains x1 = x0. That strip is the "soft switch" of
   the labels. Inside each strip the network is linear up to step 3.
3. **Step 3 (compare x2 with four thresholds).** Each neuron is h_k = |Wi_k| (±(x2 − tau_k(x0, x1)))_+. The
   threshold tau_k is linear inside a strip and is given by a closed-form matrix product
   (A3 = W_hh diag(g2) A2 + Wi e2ᵀ). When x2 is above the two inputs being compared, the u pair is on. When x2
   is below them, the v pair is on. On the boundary x1 = x2, a "crossover" pair is on: the outer neuron of one
   half and the inner neuron of the other, (0,2) or (1,3).
4. **Readout by a ratio.** Near any decision boundary only two neurons a, b are on. The logits are
   Wo[:,a] h_a + Wo[:,b] h_b, so the class depends only on ν = h_b/h_a. The upper envelope of the three lines
   Wo[i,a] + Wo[i,b] ν gives the thresholds. For the u pair the classes run 2 | 0 | 1 with breaks at ν = 0.756 and
   1.186. For the v pair they run 1 | 0 | 2 at ν = 0.840 and 1.318. For the crossover pairs they run 2 | 0 | 1 at
   0.757 and 1.191, and 1 | 0 | 2 at 0.844 and 1.320 (`python mechanism.py base`).

**Why it errs.** The true answer changes only on the three planes x_i = x_j. The network's boundary is a plane in each
strip (h_b = ν h_a). Its errors are wedges between that plane and the true one. We found three error families
`[measured, exact per-family masses, diag_family.py]`.

- **Crossover strips on x1 = x2 (74% of the clean model's 1.5% error; 60% in the two families where x0 is the extreme input).** Here the class must flip directly between
  1 and 2. But the ratio readout orders the classes 2 | 0 | 1, so a continuous ν must pass through "class 0". Every
  1↔2 boundary therefore carries a thin strip that is wrongly labelled x0. It is worst where x0 is the extreme input
  (0.46% per chart). Next to it lies a small "dead" wedge where all four neurons are off. There the logits tie at 0
  and argmax returns 0. This happens near x1 ≈ x2 ≈ 0.5–0.9·x0, where the tau lines of the crossover pair cross.
- **Tilted pair boundaries (x0 = x2 when x1 is extreme; x0 = x1 when x2 is extreme).** In the clean model these are
  almost exact (each ≤ 0.11%). Under ARC's noise they dominate: the x0 = x2 boundary in the outer strip
  is a difference of two nearly equal neuron rows (A3_1 − 0.756·A3_0, with only |Wi1 − 0.756·Wi0| = 1.70 of signal). It tilts
  from 0.48° (clean) to a median of 10.6° (90th percentile 21°) across noisy copies (`python mechanism.py base tilt`).
  That alone accounts for about 4.5 of the 12 points that noisy copies lose.
- **Lost crossover margin (3% of noisy copies).** The crossover pair works because the slopes of its two
  tau lines bracket the diagonal: 1.24 > 1 > 0.81. Noise of 0.05 can push one slope across 1. A neuron's switching
  plane then runs almost parallel to the x1 = x2 boundary, and a long thin wedge is labelled by a single neuron.

## Method

Files: [est_mech.py](est_mech.py) (the estimator), [mechanism.py](mechanism.py) (prints the closed-form
quantities), [run_mech.py](run_mech.py) (scoring), [heldout.py](heldout.py) (fresh noisy copies),
[report.py](report.py) (tables), [zoo43.py](zoo43.py) (zoo weights). Diagnostics use exact enumeration *only to
locate mis-modelled families*; the estimator never calls it: [diag.py](diag.py), [diag2.py](diag2.py),
[diag_family.py](diag_family.py), [diag_piece.py](diag_piece.py).

**Deductive.** No random model inputs. Accuracy = 1 − Σ (error cone masses).

1. **Families (fixed geometry, model-independent).** The 6 ordering chambers are each split between their two
   adjacent true boundaries by the plane on which the median is equidistant from its two neighbours. This gives 24
   half-chambers: 3 circles x_i = x_j × side (third input max or min) × truth side × chart sign(x0). Each is a
   spherical polygon, computed once for all models and not charged, like the start cones of `exact.py`. The families
   are disjoint and cover the sphere.
2. **Per model and chart:** A2 and the 4 strip thresholds t_k, then A3 for each of the 5 strips (closed form).
3. **Per half-chamber:** it is split by the strip planes that cross it. In each strip piece, the readout is:
   - *x0 = x1 and x0 = x2 families (tier A/B):* the **anchor pair**, i.e. the neurons that are on at one point near
     the x0 → 0 end of the boundary arc (`eps = 0.05`, not tuned). It is used with relu kept for both members: ratio
     sectors from the envelope, and "both off" → class 0. The error is the mass of the wrong-class sectors inside
     the piece.
   - *x1 = x2 crossover family (tier B):* the step-3 gate planes that **cross the boundary arc** inside the piece
     are resolved (the piece is split by them). In each sub-piece the active set is read at the midpoint of its arc.
     Two neurons on: the ratio readout. One or zero on: the readout pair is completed with the off neuron(s) closest
     in angle to switching on ("completion"). Three or more on: frozen linear logits. Gates that do not cross the arc
     are frozen at their value on the arc. This is a first-order expansion in the width of the error band.
   - *tier C:* the crossover treatment applied to all families (no anchor). This is the generic version for models
     whose mechanism differs.
4. Masses are exact solid angles of convex spherical polygons (Sutherland–Hodgman clipping, Van Oosterom–Strackee
   triangles, equivalent to Girard). Errors are summed directly from the slivers that the clipping cuts off, so no
   large polygon is measured twice.

**Noisy copies.** Nothing assumes the mirror symmetry. Each chart, strip and neuron uses its own coefficients
computed from that copy's weights. The symmetry only explains why the anchor pairs are u/v/crossover. Where noise
breaks the structure, tier B handles it through the arc-read active sets on the crossover family. Where an anchor
does not have exactly two neurons on (34 of 1024 copies), the family falls back to the arc-read rule.

## Status

Done (2026-10-17). It beats sampling on ARC's protocol and on a held-out set. On zoo seeds 1-4 only the generic
tier C is accurate. No step-3 gate enumeration is used (all 16 patterns per strip are never listed). But the cost is
close to brute-force enumeration on this 32-parameter model.

## Results

Truth: exact enumeration ([arc_m43_repro/exact.py](../arc_m43_repro/exact.py)). Noisy-copy truths come from
`arc_m43_repro/results.json`. Protocol: 4 noise seeds × 256 copies, absolute σ = 0.05 on all 32 weights.
Sampling RMSE at equal ops is √(mean p(1−p)/(ops/72)), with mean p(1−p) = 0.1024 over these copies. T36 checked this
formula empirically.

Command: `cd m16/approaches/understand_m43 && OMP_NUM_THREADS=1 python run_mech.py && python report.py`
(about 3 CPU-min; writes `results.json` and `run_mech.log`).

| tier | RMSE | bias | max abs err | ops (mean) | sampling RMSE at equal ops | MSE ratio |
|---|---|---|---|---|---|---|
| A anchor pair, anchor strip only | 1.16% | +0.82% | 9.1% | 4,545 | 4.03% | 0.083 |
| A2 anchor pair, all strips | 0.76% | +0.26% | 7.6% | 8,363 | 2.97% | 0.066 |
| B− default without completion | 0.58% | +0.01% | 7.5% | 12,689 | 2.41% | 0.059 |
| **B default (A2 + resolved crossover)** | **0.32%** | **+0.06%** | 4.9% | 13,718 | 2.32% | **0.019** |
| C resolved everywhere | 0.31% | +0.01% | 5.0% | 23,035 | 1.79% | 0.029 |

Per noise seed (tier B): RMSE 0.32%, 0.24%, 0.37% and 0.32%. On the clean model (= zoo seed 0) tier B gives 0.98449 against
the exact 0.98455.

**Held-out copies** (noise seeds 4 and 5, 2 × 256, never looked at during design; truth computed fresh by
`exact.py`): `OMP_NUM_THREADS=1 python heldout.py` (writes `heldout.json`, `heldout.log`).

| tier | RMSE | bias | ops | MSE ratio |
|---|---|---|---|---|
| A | 1.27% | +0.85% | 4,553 | 0.098 |
| B | 0.29% | +0.04% | 13,800 | 0.016 |
| C | 0.28% | −0.00% | 23,147 | 0.025 |

**Comparison on the same protocol** (T36 numbers):

| estimator | RMSE | ops | MSE ratio |
|---|---|---|---|
| ARC A (Girard subregions + latent-Bernoulli gates) | 3.29% | 71,040 (ARC's 5k³ rule) / 9,489 (counted) | 10.4 / 1.4 |
| ARC B (symmetry reduction + ε covariance) | 4.83% | 7,680 (6-D volumes sampled, not charged) | 2.4 |
| constant "clean model's accuracy" | 11.5% | 0 | n/a |
| this work, tier A | 1.16% | 4,545 | 0.083 |
| this work, tier B | 0.32% | 13,718 | 0.019 |
| exact enumeration | 0 | 17,506 | 0 |

**Zoo seeds 0-4** (different training runs; exact accuracies 0.985, 0.893, 0.759, 0.904, 0.945). Each cell gives the
error (estimate − exact) and, in brackets, that model's MSE ratio against sampling at equal ops.

| tier | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| A | +0.23% (0.02) | +1.08% (0.09) | +8.32% (2.53) | +3.97% (1.47) | −4.04% (2.17) |
| B | −0.01% (0.00) | −7.12% (9.36) | −1.63% (0.26) | −1.49% (0.58) | −7.36% (22.97) |
| C | −0.02% (0.00) | −0.56% (0.09) | −1.49% (0.33) | −0.94% (0.36) | −0.97% (0.60) |

**Where the residual comes from** (tier B, per family: exact error mass vs modelled, over the 1024 copies; printed
by `run_mech.py`). Every pair family (x0 = x1, x0 = x2, and x1 = x2 where x0 is the middle-ranked extreme) is within
0.01–0.03 points RMS. The two large crossover families (x1 = x2 with x0 the extreme on the chart's own side) have
RMS 0.18 and 0.24 points. 77% of the total squared error comes from the 34 copies (3.3%) whose crossover anchor does
not have exactly two neurons on. Those copies have RMSE 1.5%; the other 990 copies have 0.15%. The mis-modelled
piece in those copies is the frozen-gate approximation: one crossover neuron's switching plane is nearly parallel
to x1 = x2, so it lies inside the error band along the whole arc without crossing it.

## Compute accounting

- Ops follow the T36 counter (`m43.Ops`): multiply-add = 1, and sqrt, division or atan2 = 1. Every clip costs
  3·(vertices) plus 10 per new vertex. Every solid-angle triangle costs 21. Building A2, t_k and A3 costs about 4 per
  active entry. Each ratio threshold costs 3. Each read of an active set costs 12. Sorting and comparisons are
  charged a few ops each. A forward pass costs 72.
- **Not charged:** the 24 model-independent half-chamber polygons, computed once (the analogue of exact.py's start
  cones), and Python bookkeeping. Sampling is not charged for RNG. Ground truth is never charged.
- Totals per model (mean): A 4.5k, B 13.7k, C 23.0k ops. Exact enumeration costs 17.5k. As implemented, a
  single-core CPU run is about 19 ms per model for B (numpy overhead dominates).
- The full scoring (5 tiers × 1024 copies + diagnostics) takes about 3 CPU-min. The held-out run takes about 2 CPU-min.

## Claims

- [measured] An estimator built only from closed-form mechanism quantities (strip thresholds t_k, per-strip
  neuron rows A3, pair-ratio thresholds from Wo, anchor pairs) gives RMSE 0.32% and bias +0.06% at 13.7k ops over
  ARC's 1024 noisy copies, an MSE ratio of 0.019 to sampling. On 512 held-out copies it gives 0.29% and ratio 0.016.
  ARC's understanding-based estimators scored 10.4 (A) and 2.4 (B) on the same copies.
- [measured] The cheapest tier (one strip per family, 4.5k ops) still has MSE ratio 0.08 (RMSE 1.2%).
- [measured] The clean model's 1.5% error splits as follows: crossover strips plus dead wedges on x1 = x2 account
  for about 74%, and tilted pair boundaries account for the rest. Under noise the tilted x0 = x2 boundary (x1 extreme) is the largest single family (about 2.3
  points per chart).
- [measured] For the pair families, the 2-neuron ratio readout reproduces the exact family error mass to within
  0.03 points RMS per family under noise.
- [measured] The account derived on seed 0 does not transfer to seeds 1-4 (tier B errors −7.1%, −1.6%, −1.5% and −7.4%). Those
  models have no mirror-pair structure (e.g. seed 1 has Wi = (6.5, −6.2, −8.7, 0.22)). The generic tier C (active
  set read on each boundary arc, gates resolved where they cross it) is within 1.5 points on all five seeds and
  beats sampling on each (per-model ratio 0.09–0.60). But it costs more than exact enumeration.
- [inferred] On M_4,3 understanding does not buy compute. Tier B costs about 80% of brute-force enumeration and is
  less accurate. What it buys is a short list of error families with closed-form boundaries that explains *where*
  and *why* the network errs, and it beats sampling by about 50× in MSE.
- [inferred] The residual is concentrated in one approximation: freezing step-3 gates that do not cross the boundary
  arc. It fails when a gate plane is nearly parallel to the boundary, i.e. when a crossover slope reaches about 1.
- [speculative] The noise sensitivity of the x0 = x2 boundary (0.5° → 10.6° median tilt) comes from the cancellation
  A3_1 − 0.756·A3_0 in the u-pair readout. A model trained with weight noise would widen this margin.

## Alternatives not ruled out

Before building, we listed three alternatives and the cheapest test for each.

1. *The errors are not thin wedges at the true boundaries but large interior regions,* such as the dead tie
   region. Test: decompose the exact error by nearest boundary and gate pattern (`diag2.py`). Result: all error
   mass sits at boundaries. The dead region is only 0.2 points, and it touches x1 = x2.
2. *Noise switches on a third neuron near the boundaries, so a 2-neuron readout is wrong.* Test: compare exact and
   modelled error per family (`diag_family.py`). Result: true for the crossover family only. That finding
   motivated the arc-resolved treatment.
3. *The chamber split double-counts or misses mass.* Test: exact enumeration per half-chamber sums to the exact
   total (checked: `exact_by_half` totals equal `exact.py` to within 3e-16 on the clean model and on 3 noisy copies).

Still open:

- **The design was iterated on the test copies.** The discrete choices (which family to resolve, completion) were
  made after looking at per-family residuals on noise seeds 0-1 and at individual outliers from seeds 0-2. No
  continuous parameter was fitted. The held-out seeds 4-5 (RMSE 0.29%) suggest little overfitting, but these are
  the same noise model and the same base weights.
- The anchor offset eps = 0.05 and the chamber-split rule are choices, not derived quantities. We did not scan
  them (deliberately, to avoid tuning).
- We counted sampling's equal-ops RMSE analytically. T36 verified that formula empirically, but we did not repeat
  the empirical check at our op counts.
- We did not try a cheaper volume formula: a thin-wedge closed form (∫ arcsin along the arc) instead of exact
  polygon clipping. It could cut ops by several times at a first-order accuracy cost.

## Failure modes

- **A different mechanism (zoo seeds 1-4):** the anchor-pair rule picks wrong pairs, giving errors of up to 7 points.
  The account has to be re-derived per model, or tier C has to be used.
- **Lost crossover margin** (a tau slope near 1): frozen gates mislabel a long thin wedge. This is the source of
  the 4.9% worst-case error.
- **Cost:** on M_4,3 brute force is cheaper than this. The method only pays off if its cost scales better than the
  number of linear regions on larger models, which is untested.
- **Ties:** the dead region uses argmax's tie rule (index 0), copied from the runtime. A different tie convention
  would change the dead-wedge mass.

## Recipe for the next rung (M_6,T / M_8,T; Backlog T73)

1. Charts = sign pattern of the step-1 pre-activations (the sign of x0 picks the active set). For each later step t,
   list the gate hyperplanes A_t,k·x = 0 as closed-form matrix products restricted to the chart.
2. The true boundaries are the hyperplanes x_i = x_j between adjacent ranks. Split each ordering chamber between its
   two boundaries by the equidistance plane, which gives disjoint families.
3. On each boundary piece, read the active set on the boundary. Resolve only the gates that switch on the
   boundary piece, freeze the rest, and complete lone neurons with the nearest-to-threshold neuron. The readout is a
   ratio sector if two neurons are on, frozen linear otherwise.
4. Error per piece = Gaussian mass of the wrong-class part of a polyhedral cone. Girard's formula is 3-D only. For T ≥ 4
   use the thin-wedge identity (mass ≈ φ(0)·(|b|/|a|)·E[|m·y|; y in piece], one dimension lower) or a deterministic
   orthant routine (Genz). Measure ops against exact enumeration where it is still feasible (T46).
5. Done when bias/RMSE/MSE-ratio vs sampling at equal ops are reported on the zoo seeds and noisy copies, together
   with the number of resolved gates against the number of linear regions (the scaling question).
