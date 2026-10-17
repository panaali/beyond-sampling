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

---

# T75: one generic recipe for all five trained seeds (added 2026-10-17; the T63 sections above are unchanged)

## Question

Is there a single recipe that is deductive, has every coefficient a formula of the weights, and beats sampling at
equal ops on each of zoo seeds 0-4 of M_4,3 **and** on their noisy copies? The owner's ruling (it replaces the earlier
"cheaper than enumeration" target) adds a second requirement: the recipe's cost should scale with neurons, steps and
boundary families, not with the number of linear regions. T74 found that T63's tier C fails on noisy copies of seeds
2-3 (ratios 1.07 and 2.6), and that tiers B and C share a failure window when a crossover τ slope sits near 1.

## How the five seeds differ (plain words; `python mechanism.py <seed>`, `python diag75.py`)

- **Seed 0 (98.5%).** Two mirror pairs, as described in the T63 sections. A step-2 switch band brackets x1 = x0 (0.81–1.24).
  Errors sit in cells where two neurons are on (82%).
- **Seed 1 (89.3%).** No mirror symmetry. Neuron 3 has almost no input weight (Wi3 = 0.22), so it is a purely
  recurrent neuron. x0 is stored in one neuron when x0 > 0 and in two when x0 < 0. There is a sharp step-2 switch at
  x1/x0 ≈ 0.99. The readout often uses three neurons: 44% of the error lies in 3-active cells and 23% in 4-active cells.
- **Seed 2 (75.9%).** Its step-2 gates mostly switch on the sign of x1 (t ≈ −0.03, 0.00, 0.01), not on x1 vs x0. In
  the chart x0 < 0 there is a strip, 0.01 < x1/x0 < 0.96, in which every step-2 neuron is off, so x0 and x1 are
  forgotten. Two thirds of its error is in all-off cells, where the logits tie and the network answers "x0".
- **Seed 3 (90.4%).** Neuron 3 is purely recurrent (Wi3 = −0.03). When x0 > 0, x0 is held by a single neuron. The
  step-2 switches sit at x1/x0 ≈ 0 and 1.00. The readout is dominated by neuron 0 (Wo[2,0] = −69).
- **Seed 4 (94.5%).** One input weight is huge (Wi0 = 45). The chart x0 < 0 has a doubled x1-vs-x0 switch at
  1.046/1.047. At +e0 every neuron is off. Errors come from 3-active cells (34%) and 2-active cells (29%).

**What they share** `[measured, diag75.py]`:
- Every seed has a step-2 switch plane at or near x1 = x0.
- 82–99% of the error mass touches a true boundary x_i = x_j.
- On average only about 2 of the 4 step-3 gates cut a step-2 piece.
- The answer in any cell is a linear readout of the few neurons that are on there.

None of them has seed 0's fixed pair structure, which is why the anchor-pair rule of tier B fails on them.

## Method (frozen recipe `est75.RECIPE = dict(K=(4, 4), rule="adaptive", eps=0.1)`, file [est75.py](est75.py))

1. Use the 24 half-chambers of T63. Each is fixed geometry with a known true class and an edge on its true boundary.
2. Go step by step (t = 2, 3) inside each cell. A gate that does not cut the cell has a known state, so nothing is
   approximated there. For a gate k that does cut the cell, compute its **impact**: depth_k × w_k.
   - depth_k is the largest pre-activation the gate reaches on the side opposite to its state at the arc midpoint.
   - w_k is the largest downstream weight difference: max |W_hh| over the neuron's column for step 2, and
     max_o |Wo[true,k] − Wo[o,k]| for step 3.
3. **Resolve** (split the cell by) every cutting gate whose impact is at least eps = 0.1 times the summed impact of
   the neurons that are surely on in the cell. If no neuron is surely on, resolve all cutting gates. **Freeze** the
   others at their state on the arc.
   - In words: a neuron is frozen only if switching it cannot move the readout by more than 10% of what the
     surely-on neurons contribute.
   - With eps → 0 the recipe is exact. Resolving everything (`rule="impact"`, K = (4,4)) reproduces exact.py to 1e-16
     on all five seeds.
4. In each final cell the logits are linear. The error is the sum of the slivers cut off when the cell is clipped by
   the two "true class wins" halfspaces. An all-off cell gets the tie answer, class 0.

**Development and freezing.**
- The rules I compared were: closest-to-arc K-cap (T63 C-style), impact K-cap, hybrid, and adaptive with
  eps ∈ {0.3, 0.1, 0.03}.
- I compared them on noise seed 0 only (32 copies per zoo seed, truth cached in `dev75_truth.json`) and on the clean
  seeds.
- `RECIPE` was written into `est75.py` before any fresh-seed run.
- One 4-copy smoke run of `eval75.py` on the fresh seeds was made after freezing, to check the script. Nothing was
  changed after it.

## Results: fresh noise seeds 20-23 (4 × 256 copies per zoo seed, σ = 0.05 absolute, never used by T63, T74 or T75 development)

Command: `cd m16/approaches/understand_m43 && OMP_NUM_THREADS=1 python eval75.py && python report75.py` (about 10
CPU-min; writes `eval75.json` and `eval75.log`). Truth comes from exact.py. MSE ratio = MSE / (mean p(1−p) / (ops/72)).
The 95% CI is a bootstrap over copies.
**Ops for T75 are counted strictly**, following the T74 convention: clip costs 4 per vertex + 13 per new vertex, a
triangle costs 23, and every comparison or sign test costs 1. B and C are counted with the T63 counter, which is about
1.24× more lenient.

| zoo seed (clean acc / noisy mean acc) | T75 noisy RMSE (bias) | T75 ops | **T75 MSE ratio [CI]** | T75 clean err (ratio) | T63 B ratio | T63 C ratio |
|---|---|---|---|---|---|---|
| 0 (0.985 / 0.880) | 0.01% (−0.00) | 34.5k | **0.0002 [0.000, 0.000]** | −0.03 pp (0.00) | 0.018 | 0.029 |
| 1 (0.893 / 0.721) | 0.55% (+0.18) | 25.1k | **0.054 [0.039, 0.072]** | +0.18 pp (0.01) | 3.42 | 0.11 |
| 2 (0.759 / 0.736) | 0.12% (−0.04) | 31.7k | **0.003 [0.003, 0.004]** | +0.03 pp (0.00) | 1.30 | 0.69 |
| 3 (0.904 / 0.786) | 0.16% (−0.02) | 30.6k | **0.007 [0.004, 0.011]** | −0.21 pp (0.02) | 2.36 | 2.78 |
| 4 (0.945 / 0.571) | 1.35% (+0.16) | 23.3k | **0.25 [0.20, 0.30]** | +0.23 pp (0.03) | 1.56 | 0.36 |

The ratio per noise seed is stable. The largest single value is 0.28 (seed 4, noise seed 23). The worst single-copy
error is 6.8 pp, on seed 4.

**The τ-slope ≈ 1 failure window is closed** (`python window75.py`). Scanning Wh[0,3] by −0.40…0 (β_0 down through 1)
and Wh[2,3] by 0…+0.8 (β_2 up through 1) on the clean seed-0 model gives T75 errors within ±0.04 pp everywhere. Tier B
on the same scan errs by up to −4.97 pp and +5.93 pp.

## Compute accounting and the scaling law

- **Cost per model:** 23–35k strict ops. Exact enumeration costs 15–18k on the T63 counter and 18–22k when recounted
  strictly with the same primitives (`eval75.py` monkeypatches exact.py's clip and frac in its own process). So T75 is
  about 1.1–1.6× exact enumeration on this model. **On M_4,3 it is not cheaper than brute force.** It resolves 57–93%
  of the cutting gates and ends with 103–170 cells per model. Exact enumeration has about 88 leaves; T75 has more cells
  because the half-chambers split every chamber in two.
- **Scaling law, stated.** Cost = Σ over boundary families F of Σ over cells (clips + triangles). A cell is split only
  by gates that cut it *and* have relative impact ≥ eps. So
  cost ≈ F × ∏_t 2^{r_t} × O(H·(vertices)),
  where r_t is the number of high-impact gates that switch inside a family's cells at step t.
  - For 2nd-argmax with T inputs, the families are the boundaries x_i = x_j between the top-2 or 2nd-3rd ranks:
    F = O(T²) convex cones with O(T) facets. That is polynomial.
  - The term ∏ 2^{r_t} is polynomial only if r_t stays bounded independently of H, and only if cells are re-merged
    between steps. Without re-merging, a fixed per-step cap K gives 2^{K(T−1)} cells per family. That is
    exponential in T: 4⁹ ≈ 2.6e5 at K = 2, T = 10.
- **[inferred]:** the recipe as built does **not** guarantee polynomial cost at H = 16, T = 10. Getting there would
  need two missing pieces.
  1. Re-merging, or projecting, cells between steps. Gate patterns that lead to the same downstream linear map up to
     the eps tolerance would be merged, as in ARC's deduction-projection.
  2. A measured bound on r_t on the larger models.
  On M_4,3 the adaptive rule resolves most gates. So what this model demonstrates is the accuracy of the
  "resolve-by-impact, freeze the rest" rule, not a saving in cost.
- **[speculative]:** cone masses in dimension T ≥ 4 have no closed form (Girard is 3-D only). T73 has to supply them.

## Claims (T75)

- [measured] A single frozen recipe beats sampling at equal (strictly counted) ops on every zoo seed of M_4,3 and on
  their fresh noisy copies. The MSE ratios are 0.0002, 0.054, 0.003, 0.007 and 0.25, and every bootstrap CI is below
  0.30. On the clean models the errors are at most 0.23 pp.
- [measured] It fixes T74's two failures: noisy seeds 2-3 go from 1.30/2.36 (B) and 0.69/2.78 (C) to 0.003/0.007,
  and the τ-slope ≈ 1 window is closed (|err| ≤ 0.04 pp over the scan).
- [measured] The five seeds use different mechanisms (above). The common structure the recipe relies on has four
  parts: a near-x1 = x0 step-2 switch; errors at the true boundaries; few gates cutting each piece; and a linear
  readout of the active neurons in each cell. The recipe does not use any seed-specific pair or symmetry.
- [measured] On M_4,3 it costs 1.1–1.6× exact enumeration and resolves 57–93% of the gates that cut a cell. It is
  close to exhaustive here.
- [inferred] Its accuracy comes from the impact rule. Freezing only neurons that cannot move the readout much keeps
  the frozen-gate error small even when a gate runs parallel to the boundary, which is what broke T63's arc-distance
  rule.

## Alternatives not ruled out / failure modes (T75)

- **This is close to enumeration in disguise on M_4,3.** With 4 neurons and eps = 0.1 most gates are resolved. A
  harder test of the "understanding" content would be a larger eps at equal accuracy, or a larger H.
- **eps and the impact proxy are design choices.** They were picked on dev noise seed 0 by comparing a handful of
  variants, so this is mild selection. The fresh seeds 20-23 confirm the choice, but they use the same noise model.
  - Seed 4 is the weakest case (ratio 0.25, worst copy 6.8 pp). Its frozen step-3 gates still matter at eps = 0.03,
    because logit contributions cancel and the "surely-on" scale overstates the margin.
  - A margin-aware impact (the frozen shift compared with the actual logit margin on the gate's minority side) is
    the natural fix. It is untried.
- **Mixed op conventions.** T75 is counted strictly and B/C leniently. That favours B/C in the table, not T75.
- **Wall-clock time** is again far worse than vectorised sampling (T74's point). The comparison is in ops only.

## Files (T75)

- [est75.py](est75.py): the recipe, with its own strict-count clip and frac.
- [eval75.py](eval75.py), [report75.py](report75.py), `eval75.json`, `eval75.log`: frozen fresh evaluation.
- [window75.py](window75.py): the τ-slope window scan.
- [dev75.py](dev75.py), `dev75_truth.json`: development harness on noise seed 0 only.
- [diag75.py](diag75.py): per-seed error structure.
- `zoo43.noisy_copies`: the ARC protocol for any base model; it reproduces m43.noisy_models for seed 0.

---

# T85: invariance, margin-aware rule, cell merging (rung H=4, T=3) and T87: the first T step, rung (H=4, T=4) (added 2026-10-17)

The T63 and T75 sections above are unchanged.

## T85 changes to the recipe (frozen as `est75.RECIPE85B` before the fresh run)

1. **Margin-aware last step (the seed-4 fix).**
   - Every cutting step-3 gate is frozen at its state on the arc.
   - A gate is resolved (the cell is split by it) only if, somewhere on its minority side, the logit shift it can
     cause, |a_k·v|·max_o|Wo[t,k] − Wo[o,k]|, reaches the frozen readout margin m(v) = l_true − max_other l.
   - It is also resolved if the margin changes sign there with a shift of at least eps·max|m|.
   - The test points are the minority vertices plus the points where the gate cuts the cell's edges. For one gate,
     m − shift is concave on a cell, so the vertex test bounds the whole region.
   - All triggering gates are split at once and the children are re-tested, so the result does not depend on the
     order. A tie (shift = margin exactly, a structural case) counts as triggering.
2. **Scale-invariant step-2 impact.** T63/T75 used max_j|Wh[j,k]| as the weight factor, which depends on the
   arbitrary scale of neuron j. The T84 review found this: rescaled copies moved seed 3 from 0.004 to 0.17. The weight
   factor is now the triangle-inequality bound on neuron k's effect on the readout margin through all step-3 neurons:
   w_k = max_o Σ_j |(Wo[t,j] − Wo[o,j])·Wh[j,k]|.
   - A positive rescaling of any neuron (input and recurrent rows × c, recurrent and readout columns / c) leaves
     depth_k·w_k unchanged.
   - A first variant (effect through the step-3 neurons on at the arc point only) was tried on the dev set and
     rejected: seed 4 had −2.4 pp clean error, because a downstream neuron that is off at the arc point gives w = 0.
   - Two numerical fixes were needed: a relative tolerance for "vertex on the plane" (`cliptol = 1e-12`) and the
     order-independent trigger set.
   - **Permanent invariance test:** `rescale_test.py`, which applies a random rescaling e^U(−2,2) plus a permutation
     of the hidden neurons. `eval85b.py` also repeats every 8th model under such a transform.
3. **Cell re-merging between steps.** Two sibling cells that differ only in neuron k's step-t gate have downstream
   maps that differ only through neuron k's contribution. Merging them when that contribution is below the impact
   tolerance is *exactly* the freeze decision, so merging is already built into the impact rule.
   - Merging non-sibling cells with equal downstream maps would need non-convex cells (unions of cones), which this
     cone machinery cannot represent. That is not done; see the checklist below.

## T85 results at rung (4,3): passes on fresh noise seeds 60-63

Fresh noise seeds 60-63: 4 × 256 copies per zoo seed, σ = 0.05 absolute, never used before. Truth from exact.py. Ops
are strict. Command: `OMP_NUM_THREADS=1 python eval85b.py`, which writes `eval85b.json`; `summary87_85b.log` is
printed from it.

| zoo seed | clean err (ratio) | noisy acc | RMSE | max | ops | **MSE ratio [CI]** | cells | max change under rescaling |
|---|---|---|---|---|---|---|---|---|
| 0 | +0.03 pp (0.003) | 0.882 | 0.016% | 0.07% | 55k | **0.0002** [0.0002, 0.0002] | 183 | 1e-15 |
| 1 | +0.26 pp (0.040) | 0.723 | 0.39% | 1.9% | 42k | **0.046** [0.038, 0.053] | 114 | 1e-15 |
| 2 | +0.01 pp (0.000) | 0.734 | 0.016% | 0.13% | 46k | **0.0001** | 161 | 3e-16 |
| 3 | −0.24 pp (0.040) | 0.785 | 0.14% | 0.44% | 48k | **0.008** [0.007, 0.009] | 147 | 1e-15 |
| 4 | +0.17 pp (0.032) | 0.579 | 0.79% | 5.5% | 42k | **0.15** [0.13, 0.18] | 123 | 4e-15 |

- Seed 4 improves from T75's 0.25 to 0.15. Seed 3 is 0.008, and the recipe is now invariant to 1e-15.
- Cost rose from 23-35k to 42-55k strict ops. Exact enumeration costs 18-22k strict, so on this rung the recipe is
  still dominated by brute force.
- The non-invariant RECIPE85 (margin rule only) was started on noise seeds 40-43. The process died after seeds 0-1:
  ratios 0.0000 and 0.006 (`eval85_43.json`). It is superseded by RECIPE85B.
- Region counts at (4,3): 52-72 linear regions (exact enumeration, `regions87.py`). The recipe uses 114-183 cells,
  because the 24 half-chambers cut regions further.

## T87: rung (H=4, T=4). FAILS

A general-T implementation, [estT.py](estT.py), was needed:
- cells are pointed polyhedral cones in R^T, kept in V- and H-representation;
- clipping goes along the cone's edges (found by tight-set adjacency);
- masses come from a pulling triangulation into simplicial cones;
- each simplicial cone's 4-variate orthant probability uses Plackett's reduction to a 1-D integral, with fixed
  Gauss-Legendre on a graded variable (12 nodes, or 48 if any |ρ| > 0.95).

Checks:
- The orthant rule agrees with 8e6-sample MC on 12 random cones (rms z 0.91).
- Its worst error against a 1024-node reference is 2e-7.
- At T = 3, estT reproduces exact.py to 1e-14 and est75's recipe to 1e-14.
- At T = 4, its exact mode matches 1e7-sample MC on all 5 seeds (z between −0.8 and −0.3).

Command: `OMP_NUM_THREADS=1 python run87.py 8`, which writes `run87.json`. The recipe was frozen T85 (eps = 0.1, impact
rule at intermediate steps, margin rule at the last). The noisy copies are 8 per seed (noise seed 50) with 1e7-sample
MC truth. Sampling uses 96 ops per sample.

| zoo (4,4) seed | exact acc | exact cells / ops | recipe err | recipe cells / ops | MSE ratio clean | noisy acc | noisy RMSE | noisy ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.477 | 2595 / 83M | −1.41 pp | 1550 / 55M | **459** | 0.351 | 0.70% | **91** |
| 1 | 0.602 | 1919 / 65M | −0.62 pp | 1103 / 43M | **71** | 0.447 | 0.44% | **33** |
| 2 | 0.602 | 1897 / 65M | −0.66 pp | 1023 / 40M | **76** | 0.451 | 0.28% | **14** |
| 3 | 0.498 | 2207 / 65M | −0.21 pp | 1410 / 44M | **7.9** | 0.433 | 0.85% | **136** |
| 4 | 0.602 | 1895 / 63M | −0.66 pp | 1028 / 40M | **76** | 0.516 | 0.31% | **17** |

Region counts at (4,4) are ≥ 317-364 linear regions (distinct patterns among 2e6 inputs, a lower bound), against
52-72 at (4,3). The cell count in the half-chamber framework goes from about 150 to about 2000.

### What T = 4 changes (diagnosis; `diag87.py`, `diag87b.py`, `diag87c.py`)

1. **An uncertified intermediate step.** At T = 4, step 3 is an *intermediate* step: its gates are frozen by the
   impact rule, which has no margin certificate, because the readout comes one step later. That freezing causes
   −1.30 of the −1.41 pp on seed 0:
   - resolving the intermediate gates exactly and keeping the margin rule at the last step gives −0.106 pp;
   - doing the reverse (intermediate impact rule, exact last step) gives −1.304 pp.
   At T = 3 the only intermediate step was step 2, which costs little to resolve (4 strip planes through e2), and the
   recipe resolved it almost completely.
2. **4-D cone masses dominate the cost.** In exact mode, 74.5M of 83.2M ops go to orthant probabilities: about
   7.5k ops per simplex against 23 for a 3-D Girard triangle. The cheaper rule (6/16 nodes, 2.6k ops per simplex)
   gives **−0.80 pp even in full enumeration**. The triangulation produces thin, nearly degenerate simplices with
   correlations near ±1, and there the cheap quadrature is biased. So accuracy and cost conflict directly here.
3. **The (4,4) zoo models are bulk-error models.** Their accuracy is 0.48-0.60 (chance is 0.25). On errors the
   median angle from the nearest class-changing boundary is 4-5.5°, against 6.4° for all inputs. On (4,3) seed 0
   it is 0.3° against 10.7°. Only 47-56% of the error lies within 5° of a boundary, and in seeds 1, 2 and 4 all four
   neurons are on for 59% of the errors. The premise "errors are thin wedges at the true boundaries, read by a few
   neurons" fails: the network has not learned a clean mechanism.
4. **Sampling gets cheaper relative to deduction.** Even exact enumeration costs 63-83M ops, about 0.7-0.9M
   forward passes. At that budget sampling's RMSE is 0.07%, so any approximate recipe must be accurate to below
   about 0.07 pp. Exact enumeration (error ≲ 1e-4, against 1e7 MC) does pass the MSE criterion, but it is brute force
   and its cost tracks the region count.

### Exhaustion checklist (rule 5) for the T87 failure

| # | attempt | result |
|---|---|---|
| F1 | smaller eps (0.01) at all steps | −0.105 pp at 70M ops: ratio ≈ 3 on seed 0, fails |
| F2 | resolve intermediate steps exactly, margin rule at the last step | −0.106 pp at 71M ops: fails, about 3× |
| F3 | cheaper orthant quadrature (6/16 nodes) | 2.9× fewer mass ops but −0.80 pp bias even in full enumeration: fails |
| F4 | F2 + F3 | −0.73 pp at 32M: ratio 71, fails |
| F5 | full enumeration, accurate orthants | passes (ratio < 0.04), but this is brute force with cost ∝ regions; it does not meet the owner's criterion 3 |
| — | not tried | (a) a margin certificate propagated through the remaining steps for intermediate gates (bound the downstream logit change of a frozen gate by a product of |Wh| over the later layers); (b) closed-form spherical-tetrahedron volumes in S^3 (Murakami / Schläfli dilogarithms) or the thin-wedge fibre integrals of T73, to make 4-D masses cheap *and* exact; (c) families defined as the O(T²) class regions instead of the T!·2 chamber halves (the chamber count alone grows 24 → 96 → 7M at T = 10); (d) better-trained (4,4) models (T86) to test whether the failure is about T or about undertraining; (e) true re-merging with non-convex cells |

## Cost law (stated, and tested on the two rungs)

Cost per model ≈ (number of half-chamber families) × (cells per family) × (cost per cell).

| factor | how it grows | (4,3) | (4,4) |
|---|---|---|---|
| half-chamber families | 2·T!·2 | 24 | 96 |
| cells per family | 2^(resolved gates); the impact/margin rules resolve most cutting gates on these small models | ≈ 5-8 | ≈ 11-16 |
| cost per cell | clips O(m·d) and adjacency O(m²·f), plus masses: 23 ops per triangle at d = 3, 1.4-7.5k ops per simplex at d = 4 | — | — |
| total cells, recipe / exact | — | 114-183 / ≈ 150-200 | 1000-1550 / 1900-2600 |
| ops, recipe / exact | — | 42-55k / 18-22k | 40-55M / 63-83M |
| regions | — | 52-72 | ≥ 317-364 |
| sampling RMSE at the recipe's ops | — | ≈ 2-3% | ≈ 0.07% |

- **[measured]** On both rungs the recipe's cell count is 50-100% of the full cell count (57-89% of the cutting gates
  are resolved). Its cost therefore tracks the region count, not neurons × steps.
- **[measured]** The masses add a factor of about 300 per cell going from d = 3 to d = 4.
- **[inferred]** As built, the cost law is not polynomial: both the family count (T!) and the cells per family grow
  with the number of regions. Criterion 3 is **not met**. Reaching it would need items (a)-(c) of the checklist:
  certified freezing of intermediate gates, exact cheap masses, and O(T²) class-region families.

## Claims (T85/T87)

- [measured] RECIPE85B passes rung (4,3) on fresh noise seeds 60-63 for all 5 seeds (MSE ratio ≤ 0.15). It is
  invariant to positive neuron rescalings and permutations to 1e-15, and the invariance test is now part of the
  evaluation.
- [measured] The frozen recipe fails rung (4,4) on every seed, clean (7.9-459) and noisy (14-136).
- [measured] Most of the (4,4) error comes from freezing intermediate-step gates, which have no margin
  certificate (−1.30 of −1.41 pp on seed 0). The (4,4) zoo models are 40-52% wrong, with the errors spread over the
  sphere rather than concentrated at the boundaries.
- [measured] At d = 4, cone masses take 90% of the ops. Cheaper quadrature biases thin simplices (−0.8 pp).
- [inferred] The boundary-wedge account needs (i) a trained mechanism with boundary-concentrated errors and
  (ii) cheap exact masses. (4,4) lacks (i), and our machinery lacks (ii).

## Files (T85/T87)

- `est75.py`: `RECIPE85`, `RECIPE85B`, `_cells_margin`, `_resolve_inv`.
- `rescale_test.py`: the invariance test.
- `eval85b.py`, `eval85b.json`, `summary87_85b.log`; `eval85_43.py`, `eval85_43.json` (partial, superseded).
- `estT.py`: general-T recipe and exact mode.
- `zoo44.py`.
- `run87.py`, `run87.json`.
- `diag87.py`, `diag87.log`; `diag87b.py`, `diag87b.log`; `diag87c.py`, `diag87c.log`.
- `regions87.py`, `regions87.log`.
