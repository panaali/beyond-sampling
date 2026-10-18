# T340: can the exact-walk machinery estimate rare error probabilities (1e-4 to 1e-6)?

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · [T339 / T327](../closure_floor/README.md)

## Question
ARC's long-term goal is low-probability estimation. At p = 1e-5 and a few thousand FPE, sampling sees no events.
Does a deductive estimate built from T317's exact gate-cell walks (no tmax cut, restricted to the event) get such
probabilities right within a small factor?

## Pre-registration (2026-10-18, before any estimator code)
**Looked at before writing this (disclosed):** a 2e7-sample magnitude scan (`magnitudes.py`) of P(error type and
g23 > t) on orig, seed0 and seed3. It is used only to choose thresholds in the 1e-4 to 1e-6 range. That scan gives
P(pick rank 3 and g23 > t) of:

| model | t = 1.0 | t = 1.25 | t = 1.5 |
|---|---|---|---|
| orig | 4.4e-6 | 9e-7 | 2.5e-7 |
| seed0 | 1.8e-4 | 6.2e-5 | 2.0e-5 |
| seed3 | 1.6e-4 | 6.3e-5 | 2.3e-5 |

**Events** (input geometry; v1 > v2 > v3 the top 3 of x, g23 = v2 - v3): E(t0) = {the network outputs the position
of the 3rd-largest input AND g23 > t0}, for t0 in {1.0, 1.25, 1.5}. These are the far-gap 2~3 error islands that
T317's tmax = .5 cut declares correct.

**Models (open):** M_16,10 (orig, `arc_eval` weights) and zoo seeds 0 and 3 (`alg_zoo`, as in LB3). That gives 9
(model, event) pairs.

**Truth:** plain MC, 1e9 inputs per model, in chunks of 500k (< 2 GB), one process, own seeds, counts for all three
t0 at once. At 1e9, orig's t0 = 1.5 cell has ~250 events (relative s.e. ~6%); the others are <= 2%.

**Estimator (deductive; cost charged as executed):** T317f's line23 walk machinery (`cont_g.py` with SPREAD = 0, i.e.
T317f's closure: v1 at v2 + g12*, v4 at the latest free position, v3 - v4 linked, lows at mu(v4) linearised), with
these changes:
- no tmax cut (walk to t = 6; the mass beyond is < Phi(0.5)^... negligible, reported);
- no screen C1 (it declares the rest of a window correct, which is exactly what is being tested);
- in each cell, the exact interval where the rank-3 position is the argmax (`correct_interval` with target p3; one
  extra 229-FLOP call per cell);
- that interval's mass restricted to t >= t0 under the exact model-free law of g23 given v2 (the Phi^8 profile);
- integrated over v2 with T317's G = 20 Gauss-Legendre nodes per region (nb and b), with the full g12 mass (1 - Phi(v2)).

Variants:
- **A:** T317's transverse point, one walk per triple per region, as T317f.
- **B:** one walk per triple per GL node, with node-specific transverse points: g12 = E[v1 - v2 | v2],
  v3* = v2 - E[v2 - v3 | v2], v4* = the 1-node Gauss rule for v4 | v3*. This costs about 20x A.

**Reported per (model, event):** estimate, truth (with s.e.), ratio est/truth, C (FPE), F = (est - p)^2 max(C, 1k) /
(p (1 - p)), and the expected number of events sampling sees at C (p C). Also, as a calibration row that is not rare,
t0 = 0 (all far and near 2~3 picks) against the same MC.

**Success line (pre-registered):** "useful rare-event estimate" = ratio within [1/2, 2] on all 9 pairs for a variant.
"Partial" = within [1/3, 3] on >= 6 of 9. Otherwise fail. F is reported, but it is almost automatic here (F ~ r^2 p C
for relative error r), so the ratio is what counts.

**Expected (before running):**
- A will be off by more than 2x on some pairs. One transverse point per region cannot represent the v2-level
  dependence of far-gap islands, and the linearised mu(v4) and fixed v3 - v4 link degrade at large t.
- B is better, but still biased on orig, where p is tiny and the islands are thin.
- My guess: A fail, B partial.

**Alternatives kept in view:**
1. Far-gap rank-3 picks may arise through the lows and v4 (their real values), which the closure pins. Then any
   point closure fails, whatever the walk resolution.
2. The g12-independence (full 1 - Phi(v2)) may bias the estimate if far-gap rank-3 picks need v1 close to v2.
3. MC truth noise only matters for orig at t0 = 1.5.

## Addendum pre-registration (DRI, 13:05, T337 context; written after the A/B runs on E1 and the closure diagnostic, before any E2 run or E2 truth)
**Disclosed:** the E1 estimates of A and B (`results/est.json`) and the closure diagnostic (`results/diag_closure.json`)
were computed before this addendum, but no E1 truth beyond the 2e7 scan has been seen yet.
**Declared deviation:** the CPU truth runs 1e9 inputs for orig and 3e8 for each seed (CPU budget). GPU truths at >= 1e9
from the T192 child replace these where they arrive. The pre-registered success line is judged on the best truth
available.

**E2(t) = {network wrong (any pick) AND g12 >= t AND g23 >= t}**, for t in {0.5, 0.75}. These are T317's interior
region (both gaps beyond tmax = .5), where T337 found the noise failure. Same 3 models.

**Estimator for E2 (deductive, variants A and B as above):** the any-wrong set along line23 (complement of the target's
correct interval in each cell), restricted to g23 >= t under the Phi^8 law, times the g12 mass beyond t,
1 - Phi(v2 + t) (g12 independence: declared), integrated over the v2 nodes.

**Success line for E2:** as for E1 (ratio within [1/2, 2] on all 6 pairs = useful; within [1/3, 3] on >= 4 of 6 =
partial). Reported separately from E1.

**Also reported (DRI question):** whether the uncut walks capture those interior islands on the clean models, i.e. the
E2 ratio for variant B.

## Results, FIRST RUN (SUPERSEDED for A and B by the corrected run below: walk bug; kept for the record)
**Truths:** T192's GPU MC (claude/t192-gpu-cost-law @63d36d38, `t340/rare_mc.json`; float64; orig 1e10, seeds 1e9)
supersedes my CPU runs. My CPU orig run (1e9, independent numpy stream) agrees on E1: 5.40e-6 / 1.138e-6 / 2.16e-7 vs
GPU 5.37e-6 / 1.086e-6 / 2.09e-7, within 1-1.7 s.e. The CPU seed and E2 runs are a cross-check, appended when done.

**Calibration row (t0 = 0, not rare):** A 1.001 / 1.042 / 1.029 and B .953 / 1.000 / .925 (est/truth; orig / seed0 /
seed3). Both get the ordinary 2~3 rank-3 picks right.

### E1(t0): pick rank 3 and g23 > t0 (far-gap 2~3 islands); est / truth
| model | truth t0 = 1 / 1.25 / 1.5 | A (C 11-19k FPE) | B (C 0.21-0.32M FPE) | events plain sampling sees at B's C |
|---|---|---|---|---|
| orig | 5.37e-6 / 1.09e-6 / 2.09e-7 | .010 / .002 / .0003 | **.27 / .20 / .29** | 1.7 / .35 / .07 |
| seed0 | 1.78e-4 / 6.16e-5 / 1.97e-5 | ~1e-14 (sees nothing) | **.89 / .86 / .65** | 37 / 13 / 4 |
| seed3 | 1.65e-4 / 6.26e-5 / 2.19e-5 | .002 / .005 / .014 | **.69 / .73 / .64** | 48 / 18 / 6 |

- **A: FAIL (0/9 within 3x).** One transverse point per region puts the line23 walk where no far-gap island exists.
- **B: PARTIAL by the pre-registered line (6/9 within 3x, all 6 seed pairs within 1.6x; orig 3.5-5x low).**
  - It is a real rare-event estimate on the seeds: within 35% at p ~ 2e-5 to 2e-4, where plain sampling at B's own
    cost sees 4-48 events.
  - The orig model's islands are under-weighted 3.5-5x. The closure diagnostic (`results/diag_closure.json`) rules
    out the closure: the R1W closure on sampled inputs gives 8.4e-6 against the network's 5.2e-6 at t0 = 1.
  - So the walk's other declared approximations lose them, presumably g12 at its conditional mean (with full
    g12 mass), the fixed v3 - v4 link and the linearised mu(v4) at large t.
  - B is one-signed low on all 9 pairs.

### E2(t): wrong (any pick) and g12 >= t and g23 >= t (T317's interior, T337's noise region); est / truth
| model | truth t = .5 / .75 | A | B |
|---|---|---|---|
| orig | 9.38e-5 / 5.22e-6 | **1.11 / 2.25** | 2.31 / 10.95 |
| seed0 | 4.35e-4 / 6.40e-5 | **.82 / .71** | 1.38 / 2.06 |
| seed3 | 1.74e-4 / 1.42e-5 | **.61 / 1.37** | 1.89 / 5.00 |

- **A: PARTIAL, nearly useful:** 6/6 within 3x and 5/6 within 2x (only orig at t = .75 is 2.25x).
  - At A's own cost (11-19k FPE), plain sampling sees .1-5 events.
  - A's F is .02-.42.
- **B: partial (4/6 within 3x), but it OVER-estimates (1.4-11x).**
  - Variant B's node-specific transverse points help the far-gap 2~3 islands (E1) and hurt the interior error set
    (E2).
  - So no single transverse design gets both.
- **DRI's question (do uncut walks capture the interior islands on the clean models?):** to within a factor ~2 with
  T317's own transverse point (A), yes. The interior error mass is visible to the walk once the tmax cut is removed.
  - [inferred] The T337 noise failure therefore comes mostly from T317's tmax cut and screen C1 declaring this region
    correct, which an uncut walk repairs at ~3-5x T317f's cost.
  - [not tested] Whether it repairs the noisy copies themselves.

**F (pre-registered as uninformative here):** F ~ r^2 p C. A's E1 rows have F < 3 while being 100-1e14x wrong, so F
rewards being cheap, not being right, for rare p. The ratio is the measure.

## Claims
1. [measured] T317's exact walks, uncut and with per-node transverse points (B), estimate far-gap rank-3 error
   islands at p ~ 2e-5 to 2e-4 within 35% on two zoo seeds. On the target M_16,10 (p ~ 2e-7 to 5e-6) they are 3.5-5x
   low. Partial by the pre-registered line.
2. [measured] With T317's own single transverse point (A) the same walks miss the far-gap islands entirely (0/9), but
   get the interior error mass (both gaps >= t) within 2.25x on 6/6 pairs, at 11-19k FPE where sampling sees ~0.1-5
   events.
3. [measured] The point closure is not what loses the far-gap islands: on sampled inputs it reproduces their rate
   within 1.6x (orig over).
4. [inferred] The transverse-point approximations (the v2 / g12 / v4 placement of the walked line) decide rare-event
   accuracy. The right placement differs between far-gap 2~3 islands and the interior error set.
5. [speculative] A walk at several g12 quantiles per node, or a 2-D walk in (g12, g23), would close orig's E1 gap.
   Untested.

## Alternatives not ruled out
- Three models only, two events, chosen thresholds; the seeds are LB3 seeds (seen before, not sealed).
- The orig E1 deficit could also come from the g12-independence (full 1 - Phi(v2) mass) if far-gap rank-3 picks
  need v1 close to v2. Not separated from the transverse-point placement.
- B's cost (0.2-0.3M FPE) is far from cheap; A is cheap but only right on E2.

## Failure modes
- Rare islands sit at specific (v2, g12, v4, low) placements. A single transverse point per region can miss them by
  orders of magnitude, and an F-style score would not show it.

## Compute accounting
- Estimator cost charged as executed (`rare.py`): T317's counts for walks, plus one extra correct_interval per cell
  for the rank-3 interval, plus the Phi^8 profile terms. A 11-19k FPE, B 0.21-0.32M FPE.
- Diagnostics:
  - magnitude scan 3 x 2e7;
  - closure diagnostic 3 x 2e7;
  - CPU truth orig 1e9 (36 min), seeds 3e8, E2 2e8 per model.
- GPU truths by T192.
- Correction made during the work: my first walk copy omitted `act = go`. This only over-charged cost (zero-length
  pieces); the estimates were unchanged.

## POST HOC (DRI request, not pre-registered): equal-compute comparison against plain sampling (`python compare.py`, `results/compare.txt`)
Sampling at the run's own executed C (C inputs): typical relative error s = sqrt((1 - p) / (p C)) if p C >= 1, else 1
(with < 1 expected event, sampling usually returns 0). Ours: r = |est - p| / p. A result is a tie when |r - s| < .05
(both about equally wrong, e.g. A's ~0 answers against sampling's zero).

| event | model | variant | p | C (FPE) | p C | sampling s | ours r | result |
|---|---|---|---|---|---|---|---|---|
| E1(1.0) | orig | A | 5.37e-06 | 18739 | 0.101 | 1.00 | 0.99 | tie |
| E1(1.0) | orig | B | 5.37e-06 | 320784 | 1.72 | 0.76 | 0.73 | tie |
| E1(1.0) | seed0 | A | 1.78e-04 | 10764 | 1.92 | 0.72 | 1.00 | loss |
| E1(1.0) | seed0 | B | 1.78e-04 | 209565 | 37.3 | 0.16 | 0.11 | win |
| E1(1.0) | seed3 | A | 1.65e-04 | 15692 | 2.59 | 0.62 | 1.00 | loss |
| E1(1.0) | seed3 | B | 1.65e-04 | 292841 | 48.3 | 0.14 | 0.31 | loss |
| E1(1.25) | orig | A | 1.09e-06 | 18739 | 0.0204 | 1.00 | 1.00 | tie |
| E1(1.25) | orig | B | 1.09e-06 | 320784 | 0.348 | 1.00 | 0.80 | win |
| E1(1.25) | seed0 | A | 6.16e-05 | 10764 | 0.663 | 1.00 | 1.00 | tie |
| E1(1.25) | seed0 | B | 6.16e-05 | 209565 | 12.9 | 0.28 | 0.14 | win |
| E1(1.25) | seed3 | A | 6.26e-05 | 15692 | 0.982 | 1.00 | 1.00 | tie |
| E1(1.25) | seed3 | B | 6.26e-05 | 292841 | 18.3 | 0.23 | 0.27 | tie |
| E1(1.5) | orig | A | 2.09e-07 | 18739 | 0.00392 | 1.00 | 1.00 | tie |
| E1(1.5) | orig | B | 2.09e-07 | 320784 | 0.067 | 1.00 | 0.71 | win |
| E1(1.5) | seed0 | A | 1.97e-05 | 10764 | 0.212 | 1.00 | 1.00 | tie |
| E1(1.5) | seed0 | B | 1.97e-05 | 209565 | 4.13 | 0.49 | 0.35 | win |
| E1(1.5) | seed3 | A | 2.19e-05 | 15692 | 0.343 | 1.00 | 0.99 | tie |
| E1(1.5) | seed3 | B | 2.19e-05 | 292841 | 6.4 | 0.40 | 0.36 | tie |
| E2(0.5) | orig | A | 9.38e-05 | 18739 | 1.76 | 0.75 | 0.11 | win |
| E2(0.5) | orig | B | 9.38e-05 | 320784 | 30.1 | 0.18 | 1.31 | loss |
| E2(0.5) | seed0 | A | 4.35e-04 | 10764 | 4.69 | 0.46 | 0.18 | win |
| E2(0.5) | seed0 | B | 4.35e-04 | 209565 | 91.2 | 0.10 | 0.38 | loss |
| E2(0.5) | seed3 | A | 1.74e-04 | 15692 | 2.73 | 0.61 | 0.39 | win |
| E2(0.5) | seed3 | B | 1.74e-04 | 292841 | 50.9 | 0.14 | 0.89 | loss |
| E2(0.75) | orig | A | 5.22e-06 | 18739 | 0.0979 | 1.00 | 1.25 | loss |
| E2(0.75) | orig | B | 5.22e-06 | 320784 | 1.68 | 0.77 | 9.95 | loss |
| E2(0.75) | seed0 | A | 6.40e-05 | 10764 | 0.689 | 1.00 | 0.29 | win |
| E2(0.75) | seed0 | B | 6.40e-05 | 209565 | 13.4 | 0.27 | 1.06 | loss |
| E2(0.75) | seed3 | A | 1.42e-05 | 15692 | 0.223 | 1.00 | 0.37 | win |
| E2(0.75) | seed3 | B | 1.42e-05 | 292841 | 4.16 | 0.49 | 4.00 | loss |

**Totals (15 (model, event) pairs per variant):**
- A: win 5, tie 7, loss 3. Its wins are all E2; on E1 it ties the zero answer or loses.
- B: win 5, tie 3, loss 7. Its wins are E1 on the seeds and on orig at t0 >= 1.25; it loses all six E2 pairs (it overshoots).
- Taking the better variant per pair (itself a post hoc choice): win 10, tie 3, loss 2.

The DRI's rough count (12 / 15, with approximate C) counted the ties as wins. With executed C and ties separated, the
deductive estimate is clearly better than plain sampling in 5/15 pairs for either single fixed variant, and in 10/15 if
the variant is picked per event family (A for interior errors, B for far-gap 2~3 islands). No single fixed estimator
wins a majority.

## T340c (T218 child, T349 lane; claude/t218-tie-tail-labels @6c651118, r1_cont/ "T349 (T340c)"; pre-registered there)
Variant B with g12 integrated at K exact-law Gauss nodes for g12 | v2 (restricted to g12 >= t for E2) instead of its
conditional mean. K = 3 is their primary; K = 5 is a convergence check. Their gate: B reproduces bitwise from @620eb47b.
I copied their result files (`results/t340c_K3.json`, `t340c_K5.json`) and scored them with my `score.py` and the GPU
truths. The numbers reproduce their message.

| est / truth | E1 t0 = 1 / 1.25 / 1.5 | E2 t = .5 / .75 | C (FPE) |
|---|---|---|---|
| orig, C3 | 1.73 / 1.24 / .49 | 1.30 / 1.61 | 2.74M |
| seed0, C3 | 1.24 / 1.27 / 1.10 | 1.20 / 1.52 | 1.83M |
| seed3, C3 | 1.04 / 1.06 / 1.02 | 1.08 / 1.41 | 2.48M |
| orig, C5 (check) | 1.58 / 1.02 / .84 | 1.29 / 1.65 | 4.53M |
| seed0, C5 | 1.25 / 1.23 / 1.03 | 1.21 / 1.53 | 3.03M |
| seed3, C5 | 1.00 / .98 / .92 | 1.12 / 1.41 | 4.09M |

**Verdict on my pre-registered lines (T340c's primary is K = 3):**
- **E1: partial.** 8/9 are within 2x and 9/9 within 3x. The one miss is orig at t0 = 1.5 (.49, just outside 1/2).
  K = 5 gets 9/9 within 2x, but it is the check, not the primary.
- **E2: useful.** 6/6 are within 2x (1.08-1.61), one-signed high.
- (B, for the record, is partial on E1 by my line: 6/9 within 3x meets the >= 6 condition.)

**Interpretation.** The transverse g12 placement was the missing piece. With g12 integrated at 3-5 exact-law nodes, the
uncut walks estimate both rare-event families within a factor 2 at p from 2e-7 to 4e-4, on the target and two seeds.
- [measured] This is the first deductive estimate in the project that gets probabilities of order 1e-6 to 1e-7 right
  within 2x.
- **Cost is the catch:** 1.8-4.5M FPE (9-14x B).
  - At that compute, plain sampling sees 15-1,300 events for most pairs.
  - So on the post hoc equal-compute table, C3 wins 4, ties 3 and loses 8 of 15 (C5: 3 / 3 / 9).
  - Its wins are where p C <~ 90: orig E1 at 1.25 and 1.5, and the seeds at 1.5.
- For truly rare events (p C << 1 at the method's cost) the deductive route wins. At p >~ 1e-5 and millions of FPE,
  sampling is better.
- [inferred] The cost scales with walks, not with 1/p, so the advantage grows as p falls. That is ARC's regime, but the
  constant (~2-4M FPE per model for 3 models' events) must come down for the moderate-p cases.

**Remaining error.** E2 is one-signed high (+8 to +65%), E1 mostly high on the seeds, and orig E1(1.0) is not converged in
K (1.73 -> 1.58). [speculative] The fixed v3-v4 link and the linearised mu(v4) are the next candidates.

## CORRECTION: walk bug and the corrected results (2026-10-18)
**Bug (found by the T218 child in its T340c runs; confirmed here).** When several pre-activations reach 0 at the same t,
the continuity restart toggled only the argmin gate. This happens, for example, to every step-0 neuron when the input at
position 0 crosses 0 (z = x_0 Wi). The others' roots equal t exactly, the strict > t filter dropped them, and their
gates were never flipped. Along the uncut lines (to t = 6), v3, v4 and the lows cross 0, so this is common.
- `check_walk.py` compares each walk's correct length with a dense forward (20k points per line, 30 rows per line).
- Before the fix: orig nb-23 mean |error| 1.16, max 3.98 (in units of t).
- **T317f's own walk on its [0, .5] window is exact** (0.0000 on all lines of orig and seed0, with and without screen
  C1). The bug does not touch T317f, T317c-e or their HP scores.

**Fix (`rare.py`, `walk3`).** At each restart, every step >= s* is re-gated from its value. A neuron whose |z| is at 0
(tolerance 1e-9 relative) takes the sign of its new slope. Tangents are recomputed in order, and roots exclude neurons at
0. The added work is charged (4H per re-gated step).
- After the fix: all lines (A's region lines and B's node lines, orig / seed0 / seed3) match the dense forwards to grid
  resolution (max .0005).
- The pre-fix estimates are kept in `results/est_prefix_buggy.json`.

**Truth cross-checks.** CPU (numpy, own seeds) against GPU (T192):
- orig E1 at 1e9: within 1-1.7 s.e.
- seeds E1 at 3e8: within ~1%.
- E2 at 2e8: within 1.6 s.e.
- GPU truths are used.

### Corrected est / truth (A and B rerun; C3 / C5 are T340c's PRE-FIX numbers, pending their rerun)
| event | orig A / B | seed0 A / B | seed3 A / B |
|---|---|---|---|
| E1(0.0) | 1 / **0.952** | 1.04 / **0.997** | 1.03 / **0.924** |
| E1(1.0) | 0.0101 / **0.253** | 0 / **0.974** | 0.000383 / **0.693** |
| E1(1.25) | 0.00152 / **0.174** | 0 / **1.03** | 2.97e-05 / **0.764** |
| E1(1.5) | 0.000179 / **0.193** | 0 / **0.933** | 1.88e-06 / **0.73** |
| E2(0.5) | 0.96 / **0.986** | 0.817 / **1.07** | 0.585 / **1.4** |
| E2(0.75) | 0.942 / **0.953** | 0.714 / **1.22** | 1.23 / **3.4** |

C (FPE, as executed): A 10.7-17.6k, B 0.21-0.31M.

**Verdict (pre-registered lines, corrected run):**
- **E1, variant B: PARTIAL (6/9 within 3x).**
  - The seeds are .69-1.03 (seed0 .93-1.03).
  - orig is .17-.25: 4-6x low and not improved by the fix.
- **E1, variant A: FAIL (0/9).**
- **E2, variant A: USEFUL (6/6 within 2x: .59-1.23).**
- **E2, variant B: PARTIAL (5/6 within 2x).** orig .99 / .95, seed0 1.07 / 1.22, seed3 1.40 / 3.41.
- Most of B's earlier E2 overshoot (2-11x) was the bug.
- T340c (C3 / C5): verdict suspended until their post-fix numbers arrive. Their pre-fix numbers were computed with the
  same bug.

**Post hoc equal-compute comparison, corrected** (`results/compare.txt`; ties when |r - s| < .05):
- A: win 6, tie 7, loss 2.
- B: **win 9**, tie 3, loss 3.
- Better variant per pair (post hoc): win 10, tie 4, loss 1.
- B now beats plain sampling at its own cost on 9 of 15 (model, event) pairs: 5/9 of E1 (all of seed0, and orig at
  t0 >= 1.25 where sampling sees < 1 event) and 4/6 of E2.

**What still fails:** orig's far-gap 2~3 islands (E1). The closure keeps them (diagnostic), and B's single g12 point loses
them. T340c's exact-law g12 nodes are the fix to check, once rerun on the corrected walk.

## T340c POST-FIX (T218 child, claude/t218-tie-tail-labels @fe7197b1, r1_cont/ "T354 results"), scored here
Their rows are copied into `results/t340c_fixed_B_C3_S3.json` (their B is relabelled B218) and
`results/t340c_results_fixed_C5_S5.json`, and scored with my `score.py` on the GPU truths. The pre-fix T340c files are
no longer used.
- **Independent check of the fix:** their fixed B (B218, own code) gives estimates equal to my corrected B on all 15
  pairs to the printed precision. Their C is ~3.5% lower (counting differences).
- S3 (their segment walk) gives estimates identical to C3 at .77-.80x its cost.

| est / truth: E1 t0 = 1 / 1.25 / 1.5; E2 t = .5 / .75 | orig | seed0 | seed3 | C (FPE) orig / seed0 / seed3 |
|---|---|---|---|---|
| B (mine, corrected) | .25 / .17 / .19; .99 / .95 | .97 / 1.03 / .93; 1.07 / 1.22 | .69 / .76 / .73; 1.40 / 3.40 | .31M / .21M / .27M |
| C3 = S3 (g12 at 3 exact-law nodes) | 1.77 / 1.24 / .48; 1.03 / 1.01 | 1.32 / 1.44 / 1.43; .97 / .98 | 1.05 / 1.10 / 1.10; .73 / .55 | S3: 1.97M / 1.40M / 1.76M |
| C5 (convergence check) | 1.60 / 1.01 / .78; 1.02 / 1.03 | 1.34 / 1.40 / 1.33; .98 / .99 | 1.02 / 1.01 / 1.01; .78 / .55 | 4.22M / 2.92M / 3.70M |

**Final verdict (pre-registered lines; T340c's primary is K = 3):**

| variant | E1 (far-gap 2~3 islands) | E2 (interior errors) |
|---|---|---|
| A (T317's transverse point) | FAIL (0/9) | **USEFUL** (6/6 within 2x) |
| B (node transverse points) | PARTIAL (6/9 within 3x; orig 4-6x low) | PARTIAL (5/6 within 2x) |
| C3 / S3 (+ exact-law g12 nodes) | PARTIAL (8/9 within 2x, 9/9 within 3x; miss: orig t0 = 1.5 at .48) | **USEFUL** (6/6 within 2x) |
| C5 (check only) | 9/9 within 2x | 6/6 within 2x |

**Post hoc equal-compute comparison** (`results/compare.txt`; ties when |r - s| < .05). Win / tie / loss of 15:

| variant | win | tie | loss |
|---|---|---|---|
| A | 6 | 7 | 2 |
| **B** | **9** | 3 | 3 |
| C3 | 4 | 5 | 6 |
| S3 | 5 | 4 | 6 |
| C5 | 6 | 3 | 6 |
| best per pair (post hoc) | 11 | 2 | 2 |

**Reading:**
- [measured] With exact gate walks (no tmax cut) and the transverse integration done exactly in g12 (3-5 nodes),
  deductive estimates of error probabilities from 2e-7 to 4e-4 come within a factor of 2 on 14 of 15 pairs (C3), or
  15 of 15 (C5). That covers the target and two zoo seeds and both event families.
- Accuracy costs compute: B is 7-14x cheaper and wins more pairs against sampling at equal compute (9/15), because
  sampling at C3's ~2M FPE already sees 15-1,300 events on most pairs.
- The deductive estimate pays where events are rare at the method's own cost: orig E1 and the seeds' t0 = 1.5.
- [inferred] The remaining structure: B's single g12 point loses orig's far-gap islands; exact g12 nodes recover them.
  The rest of the error (C3 high by 24-77% on orig E1 at t0 <= 1.25 and on seed0 E1; seed3 E2 .55-.78) is from the
  other declared approximations (v3 - v4 link, linearised mu(v4), g12-g23 factorisation).
- Unverified: T349 (T241 child) re-checks independently on DEV3.

# T362: conditional Monte Carlo along exact gap lines (SAMPLING-CLASS; Fable T346 idea L)

## Pre-registration (2026-10-18, before any T362 code)
**Class: SAMPLING (random model inputs). Not deductive.** It is the variance-reduction comparator for ARC's
low-probability goal.

**Estimator (unbiased Rao-Blackwellisation).**
1. Draw x ~ N(0, I_10) and sort it: v1 > v2 > v3 > v4 > ..., at positions p1, p2, p3, ...
2. Given all values except v3 (and the positions), v3 has the exact conditional law phi(v) / (Phi(v2) - Phi(v4)) on
   (v4, v2), at position p3. So for any event f(x),

       E[f] = E[ Z ],   Z = int_{v4}^{v2} f(x with x_p3 = v) phi(v) dv / (Phi(v2) - Phi(v4)).

3. Z is computed exactly by walking the network along x_p3 from v2 down to v4 with `rare.walk3` (the corrected walk,
   T340), the other 9 inputs at their sampled values. In each gate cell:
   - E1: the exact interval where p3 is the argmax;
   - E2: the complement of the interval where p2 is the argmax.
   It is intersected with the event's gap constraint and integrated under phi.
4. Events:
   - E1(1.0) = {pick = p3, v2 - v3 > 1}, so the line is restricted to v in (v4, v2 - 1).
   - E2(.5) = {pick != p2, v1 - v2 >= .5, v2 - v3 >= .5}, so the line is restricted to v in (v4, v2 - .5), with
     g12 >= .5 as a factor fixed by the sample.
5. Model-free screen (no model call): lines whose admissible interval is empty (v2 - v4 <= t0, or g12 < .5 for E2)
   have Z = 0 exactly and are not walked. The estimate is sum Z / N_drawn over all drawn inputs.

**Line direction (declared).** This is the e_p3 line, moving v3 alone with its exact conditional law. Fable's sketch uses
the (e_p2 - e_p3) direction, which is not tried here. Both are unbiased; the variance can differ.

**Run (pre-registered).**
- Model: orig (M_16,10).
- Events: E1(1.0) (truth 5.367e-6) and E2(.5) (9.378e-5), GPU truths from T192.
- **1e4 walked lines per event** (draw inputs until 1e4 pass the screen). Seeds 362000 / 362001.

**Reported:**
- mean;
- s.e. from the per-input variance;
- the number of drawn inputs;
- per-line cost (walk3's executed FLOPs / 5948, with input generation uncharged as in LB3);
- total C;
- variance gain G = [p (1 - p) / C] / Var(estimate), against plain MC at the same C (1 FPE per input). It uses the
  per-input sample variance; the truth p is the GPU value.

**Go (DRI):** G >= 10 for an event AND the mean within 2 s.e. of the GPU truth. Reported per event.
**Expected:** Fable predicts 10-50x. My guess: E1 >= 10 (the walk resolves thin far-gap islands exactly), and E2 lower
(its wrong set is broader along v3, and g12 is not integrated).

## T362 results (SAMPLING-CLASS; `python cmc.py 10000`, `results/cmc.json`, `results/cmc.txt`)
**Validation:** on 40 lines per event (seed0), Z from the walk against a dense 40k-point evaluation along v3:
- E2: max difference 8e-16;
- E1: max 2.4e-6, at the dense grid's resolution for thin islands.

**Pre-registered run (orig, 1e4 walked lines per event):**

| event | truth (GPU) | estimate +- s.e. | z | inputs drawn | lines with Z > 0 | FPE per line | C (FPE) | gain G vs plain MC |
|---|---|---|---|---|---|---|---|---|
| E1(1.0) | 5.37e-6 | 1.66e-5 +- 7.2e-6 | +1.56 | 61,700 | 7 | 8.49 | 84,900 | **1.2** |
| E2(.5) | 9.38e-5 | 1.14e-4 +- 2.6e-5 | +.78 | 45,545 | 701 | 7.31 | 73,050 | **2.0** |

**Verdict: NO GO for both events.** The means are within 2 s.e. (unbiased, as expected), but G = 1.2 and 2.0 against the
>= 10 go line.

**Post hoc precision check** (1e5 lines, fresh seeds; `results/cmc_1e5.txt`):
- E1: est 4.49e-6 +- 1.15e-6 (z -.77), 24 hit lines, **G = 4.7**.
- E2: 9.94e-5 +- 7.2e-6 (z +.79), **G = 2.5**.
- The 1e4-line E1 gain rested on 7 hits. The better estimate is ~5x, still below 10.

**Why (measured quantities):**
- G ~ 1 / (c_eff x mbar).
- The walk costs c = 7.3-8.6 FPE per line (9-11 cells, about 3x Fable's assumed 3 FPE). The model-free screen walks
  only 16-22% of drawn inputs, so the cost per drawn input is 1.4-1.6 FPE.
- E1: the v3 line rarely hits a far-gap island (24 of 1e5 lines). When it does, it carries a large share of that
  line's admissible mass (mean Z | hit ~ .15). The rarity of E1 sits in the transverse coordinates, which the v3 line
  does not integrate: Rao-Blackwellising the coordinate along which the island is wide buys little.
- E2: 7% of lines hit, but with small conditional mass (~.007 | hit).
- [inferred] A line direction or coordinate across which the island is thin (e.g. g12 for E2, or the transverse
  direction that carries E1's rarity) would gain more. So would importance sampling of the transverse coordinates
  toward the island (the exact walks give the needed proposals). Neither is tried.

**Labels:** sampling-class throughout; unbiased; no closure, no declared approximation (the walk is exact, validated
against dense forwards).

# T364: deep-tail rare events (p ~ 1e-8 to 1e-10), the regime sampling cannot reach

## Pre-registration (2026-10-18, before any T364 truth)
**Seen before writing this (disclosed):** only K3 estimates on a t0 grid (`results/deep_grid_6.0.json`, no truth), used
by the threshold rule below.

**Estimator (deductive; frozen code, not mine):** the T218 child's K3 = T340c with the simultaneous-flip fix
(`walkfix.walk3_fixed`) and the T354 segment start, `t354.est_seg(model, K=3, t0s, tmax)`, from
claude/t218-tie-tail-labels @e433919d r1_cont/{t340c,t354,walkfix}.py. It runs from a read-only copy through `deep.py`;
cost as executed.
- **T_hi = 6 is primary.** T359's cut (T_hi = 3.5) has a declared tail bound of 7e-8, which is larger than every target
  p here, and s526's thresholds reach 3.5. So the cut is reported only as a secondary arm, for t0 < 3.5.
- Check: orig E1(1.0) gives 9.510e-6, the same as T340c's K3 (1.77 x 5.367e-6).

**Models:**
- the target M_16,10 (orig);
- the first two DEV3 models not used by any rare-event task so far (T350 / T358 / T360 used dev3_s500..525):
  **dev3_s526, dev3_s527** (T342 weights sha256 62c95203...).

**Events:** E1(t0) = {pick = p3 AND g23 > t0}. Rule: per model, the two t0 on the grid {1.75, 2, ..., 3.5} whose K3
estimate is closest to 1e-9 in |log10|. This gives:
- orig t0 = 2.0 (est 1.41e-9) and 1.75 (1.85e-8);
- s526 t0 = 3.25 (1.21e-9) and 3.5 (2.04e-10);
- s527 t0 = 2.0 (9.55e-10) and 2.25 (1.28e-10).

**Truth: conditional-law MC (exact, unbiased; sampling, used for truth only).**
- Draw from the exact law of x given g23 > t0:
  - v2 by inverse CDF of 90 phi(v2) (1 - Phi(v2)) Phi(v2 - t0)^8 on a fine grid;
  - v1 ~ N(0,1) truncated to > v2;
  - v3 = Phi^-1(Phi(v2 - t0) U^(1/8));
  - the 7 lows i.i.d. N(0,1) | < v3;
  - a uniform random assignment of the 10 values to positions.
- p = P(g23 > t0) (exact quadrature) x the conditional frequency of pick = p3.
- This is plain MC on the event's own gap region: no closure, no walk, independent of the estimator. Its variance
  equals plain MC with 1 / P(g23 > t0) (1.7e3 to 1.4e7) times more samples.
- **Validation before any deep truth:** the same sampler at t0 = 1.0 and 1.5 on orig must agree with the GPU truths
  (5.367e-6, 2.09e-7) within 2 s.e.
- N per pair, fixed now (aim >= 400 events at the estimate): orig 1.75: 1e8; orig 2.0: 3e8; s526 3.25 and 3.5:
  1e7 each; s527 2.0: 3e8; s527 2.25: 6e8. If a pair sees < 100 events, it is extended 4x (declared).
- Cross-check: the same sampler on GPU by T192, if available.

**Go (DRI): est / truth within 2x on >= 5 of 6.**

**Reported:** est / truth with the truth's s.e., C (FPE), and the expected number of events plain sampling sees at C
(p C).

## T364 results: FAIL (2/6 within 2x; go needed >= 5/6)
**Truths:** T192 GPU, conditional-law MC (claude/t192-gpu-cost-law @efa9a5fc, `t364/tail_mc.json`, copied to
`results/t364_truth_gpu.json`), 1e8-1e9 conditional samples per pair, float64.
- Its validation pair orig t0 = 1.0 gives 5.389e-6 against T340's plain-MC 5.367e-6 (z .8).
- My CPU run of the same sampler (`truthc.py`, own seeds and code) agrees on every pair finished so far:
  - s526 3.25: 1.009e-9 vs 1.010e-9;
  - s526 3.5: 1.799e-10 vs 1.816e-10;
  - orig 1.75: 4.04e-8 vs 3.93e-8;
  - orig 2.0: 6.81e-9 vs 6.83e-9;
  - s527 2.0: 3.74e-8 vs 3.79e-8.

`results/t364_score.txt`:

| model | t0 | truth (GPU) | K3 estimate (T_hi 6) | est / truth | T_hi 3.5 arm | C (FPE) | events plain MC sees at C |
|---|---|---|---|---|---|---|---|
| orig | 1.75 | 3.93e-8 | 1.85e-8 | .47 | .47 | 514k | .020 |
| orig | 2.0 | 6.83e-9 | 1.41e-9 | **.21** | .21 | 514k | .0035 |
| dev3_s526 | 3.25 | 1.01e-9 | 1.21e-9 | **1.20** | 1.00 | 489k | .0005 |
| dev3_s526 | 3.5 | 1.82e-10 | 2.04e-10 | **1.12** | - | 489k | .0001 |
| dev3_s527 | 2.0 | 3.79e-8 | 9.55e-10 | **.025** | .025 | 604k | .023 |
| dev3_s527 | 2.25 | 8.18e-9 | 1.28e-10 | **.016** | .016 | 604k | .005 |

**The go line is not met:** 2/6 are within 2x (both s526 pairs). Every miss is low, by up to 60x.
- The thresholds were chosen from the estimator, which was far off, so the true p of the s527 and orig events is ~1e-8,
  not ~1e-9.
- On s526 the estimator is right at 2e-10 to 1e-9, where plain sampling at the same compute expects 1e-4 to 5e-4 events.

**Why (diagnostic, `diag_deep.py`, `results/diag_deep.json`; conditional samples given g23 > t0).** Of the hits:

| model, t0 | hits | g12 quantiles 10 / 50 / 90% | share with g12 < .1 | hit kept: R1W closure / lows at mu(v4) / v1 at E[v1 / v2] |
|---|---|---|---|---|
| dev3_s527, 2.0 | 682 | .012 / .034 / .090 | 93% | .62 / .83 / .004 |
| orig, 2.0 | 361 | .012 / .048 / .095 | 93% | .06 / .58 / .000 |
| dev3_s526, 3.25 | 2,497 | .015 / .089 / .27 | 54% | .95 / .95 / .59 |

- **The deep-tail error islands are triple-tie configurations:** v1 within ~.05 of v2 (a near 1~2 tie), with v3 far below.
  At t0 >= 2 the network picks the rank-3 input almost only there.
- K3 integrates g12 with 3 Gauss nodes of its conditional law, and cannot resolve that thin sliver at g12 ~ 0. Moving v1
  to its conditional mean removes 99.6-100% of the hits on the failing models, against 41% on s526, where K3 works.
- On orig, the R1W closure also loses most of them (6% kept). On s527 it keeps 62%.
- [inferred] So the failure is the transverse g12 integration (first) and, for the target, the closure. It is not the
  walk, which stays exact.
- Fixes, untried:
  - g12 resolved near 0: nodes clustered at 0, or a second exact walk along g12, i.e. a 2-D treatment of (g12, g23)
    at large g23;
  - for orig, the real lows / v4 (the closure).

**Compute:**
- K3 runs as executed, ~0.5-0.6M FPE per model, covering all 8 grid thresholds in one walk.
- Truths ~10-20 GPU-min (T192) plus ~45 CPU-min (mine).
- Diagnostics ~15 CPU-min.

# T366 (fresh test of the T218 child's triple-tie fix): pre-registration (2026-10-18, before any T366 truth or estimate)
**Roles.** The T218 child develops and freezes the estimator on T364's three models only (orig, dev3_s526, dev3_s527).
I own this test. Nothing below has been run, and no estimate exists for these models.

**Models (fresh):** dev3_s528, dev3_s529, dev3_s530, and dev3_s531 (the next unused DEV3 model). T342 weights,
sha256 62c95203...

**Thresholds (model-free rule, avoiding T364's selection effect):**
- Rule: p ~ P(g23 > t0) x f, with a fixed factor f = 1e-4 (the log-centre of the conditional rates seen in T364:
  1.2e-5 to 2.6e-3).
- Targets: P(g23 > t0) = 1e-4 and 1e-5, i.e. p ~ 1e-8 and ~ 1e-9.
- Exact quadrature gives t0 = 2.336 and 2.739, rounded to **t0 = 2.35** (P = 9.28e-5) **and 2.75** (P = 9.38e-6),
  the same for every model.
- **Events:** E1(t0) = {pick = p3, g23 > t0}, 4 models x 2 thresholds = **8 pairs**.

**Truth (conditional-law MC, the adopted standard):**
- `truthc.py` (CPU, own seeds), N = 1e8 conditional samples per pair; a pair with < 100 hits is extended 4x (declared).
- Plus a T192 GPU cross-check at 1e9 per pair, if available.
- The truths are computed but **not shared or committed** until the T218 child's estimator is frozen (commit hash and
  file hashes received). The values are then committed with the scores.

**Estimator:** the T218 child's frozen fix, run as delivered (their commit, unmodified), cost as executed.

**Go: est / truth within 2x on >= 5/6 of the pairs, i.e. >= 7 of 8.** Also reported:
- the plain C4 / K3 baseline (T364's estimator) on the same pairs;
- C;
- the expected number of events plain sampling sees at C.

## T366 results: GO (8/8 within 2x; go needed >= 7/8), but not discriminating
**Unsealing.** The frozen estimator arrived first: claude/t218-tie-tail-labels @5b4a1250, all seven sha256 checked
(estimator_t366 6bd6822e, t366 997a1b60, t340c 7e6a749d, t354 abed36c8, walkfix 8e21b6a1; rare.py@620eb47b 09f89a8b,
cont_g.py@620eb47b d34dde36). It was run unmodified (`run_t366.py`, T340_SRC = a 620eb47b export). The truths were
committed only after that, in `results/t366_truth_{gpu,cpu}.json`:
- T192 GPU: 1e9 conditional samples per pair, seeds 3640000 + i;
- my CPU run: 1e8 per pair;
- they agree within |z| <= 1.2 on all 8 pairs.

`results/t366_score.txt` (GPU truths):

| model | t0 | truth | T366 est / truth | K3 (T364 estimator) est / truth | T366 C (FPE) | events plain MC sees at C |
|---|---|---|---|---|---|---|
| dev3_s528 | 2.35 | 3.68e-8 | 1.26 | 1.39 | 1.94M | .07 |
| dev3_s528 | 2.75 | 2.72e-9 | 1.30 | 1.31 | 1.94M | .005 |
| dev3_s529 | 2.35 | 1.43e-7 | 1.22 | 1.09 | 2.09M | .30 |
| dev3_s529 | 2.75 | 1.16e-8 | 1.15 | .88 | 2.09M | .024 |
| dev3_s530 | 2.35 | 7.16e-7 | 1.17 | 1.23 | 1.57M | 1.1 |
| dev3_s530 | 2.75 | 7.02e-8 | 1.23 | 1.23 | 1.57M | .11 |
| dev3_s531 | 2.35 | 2.43e-7 | 1.11 | 1.13 | 1.64M | .40 |
| dev3_s531 | 2.75 | 1.91e-8 | 1.10 | 1.23 | 1.64M | .031 |

**Verdict.**
- **GO by the pre-registered line.** All 8 are within 2x, in fact within 1.10-1.30, one-signed high, at p from 2.7e-9
  to 7.2e-7. At the estimator's own compute, plain sampling expects .005-1.1 events.
- **Caveats (stated plainly):**
  1. T364's plain K3 is also 8/8 within 2x on these pairs (.88-1.39). These four fresh models are not
     triple-tie-dominated at these thresholds, so this test does not discriminate the T366 fix from K3. The fix's
     evidence remains the T218 child's dev result on T364's models (5/6 vs K3 2/6; dev, not fresh).
  2. The model-free threshold rule (fixed factor f = 1e-4) missed the target range. The true p is 2.7e-9 to 7.2e-7,
     not ~1e-8 / 1e-9: the conditional rates here are 3e-4 to 7.5e-3. Only s528 t0 = 2.75 reached ~1e-9.
- [measured] Together with T364 (s526 at 1.8e-10 to 1e-9 within 1.2x), the K-node walk estimates fresh-model deep tails
  within 1.3x on 10 of 12 fresh pairs (s526 and s528-531) at 1e-10 to 7e-7.
- [measured] The failures so far are the triple-tie-dominated models (orig, s527), where only the T366 fix helps
  (orig still .24 at t0 = 2: the closure).
- Unverified.
- Cost: K3 (T364) is 0.36-0.47M FPE on these models, against T366's 1.57-2.09M (about 4.5x). On non-triple-tie models
  the fix buys no accuracy for its cost.

# T367 (discriminating test of the triple-tie fix): pre-registration (2026-10-18, before the pairs are known)
- **Pairs:** 4 unused DEV3 models x 2 thresholds, where the triple-tie sliver dominates the hits. They are selected and
  sealed by the T192 child, who sends me only the model ids and the thresholds. Event E1(t0) = {pick = p3, g23 > t0}.
- **Estimators, both frozen and run unmodified:**
  - (fix) claude/t218-tie-tail-labels @5b4a1250 `estimator_t366.est`, with the 7 sha256 checked as in T366;
  - (K3) the T364 estimator, `deep.py` = @e433919d t340c / t354 / walkfix, T_hi = 6.
- **Order:** run both, commit and push both sets of estimates with C, and only then ask T192 to unseal the truths.
- **Go for the fix:** est / truth within 2x on >= 7/8, AND closer to the truth than K3 (smaller |log(est / truth)|) on
  >= 6/8.
- **Reported:** both estimators' est / truth, C, and the expected number of events plain MC sees at each C.

## T367 change of plan (DRI, 17:01; truth-blind selection). Pre-registration addendum, before any run on these models
This replaces T192's sealed selection. Both estimators are frozen as above.

**Truth-blind selection rule:**
1. Models: dev3_s532 ... dev3_s547 (16 unused DEV3 models).
2. Per model, K3 on the grid t0 in {1.75, 2.0, ..., 3.5}. The candidate t0s are those with a K3 estimate in [1e-9, 1e-8].
   If fewer than 2 qualify, take the 2 grid points nearest to 3e-9 in |log10|.
3. The fix (@5b4a1250) is run on each model's candidate t0s. Disagreement d = |log(fix / K3)|.
4. Per model, keep its 2 candidates with the largest d. Model score = the mean d of those 2. The 4 top-scoring models
   give the **8 pairs**.
5. Both estimators' outputs for all 16 models are committed and pushed **before** the pairs go to T192 for sealed
   conditional-law truths (>= 1e9 conditional samples per pair). The truths are unsealed only after that commit.

**Go for the fix:** within 2x on >= 7/8 AND closer to the truth than K3 (smaller |log(est / truth)|) on >= 6/8.
Selecting on disagreement guarantees that the two differ on these pairs; the test is which one is right.

## T367 results: NO GO by the pre-registered line, narrowly; the fix is far better than K3 overall
**Order:** both estimators' outputs (all 16 models) were committed at 8c636b20 **before** the truths were requested.
- Truths: T192 GPU, conditional-law MC (`t367/tail_g12.py`), float64, 1e9 per pair, seeds 3671000 + i;
  `results/t367_truth_gpu.json`.
- Score: `results/t367_score.txt`.

| pair | truth | share of hits with g12 < .1 | fix est / truth | K3 est / truth | fix closer? | C fix / K3 (FPE) | events plain MC sees at the fix's C |
|---|---|---|---|---|---|---|---|
| s541 t0 = 1.75 | 3.54e-7 | 1.00 | **1.25** | 3.7e-5 | yes | 1.91M / .43M | .68 |
| s541 t0 = 2.0 | 1.14e-7 | 1.00 | **1.35** | 2.9e-6 | yes | 1.91M / .43M | .22 |
| s543 t0 = 2.5 | 3.00e-8 | .29 | **.90** | .42 | yes | 2.20M / .61M | .066 |
| s543 t0 = 2.75 | 5.49e-9 | .30 | **1.09** | .48 | yes | 2.20M / .61M | .012 |
| s546 t0 = 2.5 | 6.80e-9 | .35 | .55 | 1.21 | no | 2.12M / .58M | .014 |
| s546 t0 = 2.75 | 1.33e-9 | .31 | **1.08** | 2.11 | yes | 2.12M / .58M | .003 |
| s547 t0 = 2.25 | 1.23e-8 | .79 | 2.43 | 1.03 | no | 2.08M / .55M | .026 |
| s547 t0 = 2.5 | 1.65e-9 | .80 | 2.12 | 1.25 | no | 2.08M / .55M | .003 |

**Verdict.**
- **The pre-registered line is missed narrowly.** The fix is within 2x on **6/8** (7 needed) and closer than K3 on
  **5/8** (6 needed).
- Both misses of the 2x line are dev3_s547, where the fix over-estimates 2.1-2.4x and K3 is within 1.25x.
- **Overall the fix is decisively better:**
  - K3 is within 2x on only 3/8 pairs.
  - On dev3_s541, where every hit has g12 < .1, K3 is off by 3e4 to 3e5x (3.4e-13 against 1.1e-7). The fix gets
    1.25-1.35x there.
  - On s543 the fix is 1.09 / .90 against K3's .48 / .42.
  - Median |log ratio|: fix .26, K3 .74.
- **Where the fix fails:** dev3_s547, the model where triple-tie hits are 80% but not 100% of the total. Its graded g12
  rule over-weights there (2.1-2.4x).
- **Cost:** the fix costs 3.4-4.4x K3 (1.9-2.2M vs .43-.61M FPE). At the fix's compute, plain sampling expects
  .003-.68 events.
- Unverified.

# T369 (fresh test of the recalibrated triple-tie fix): pre-registration (2026-10-18, before the estimator is frozen)
- **Estimator:** the T218 child's recalibrated graded rule, frozen by them on already-seen models only. Its commit and
  sha256 are recorded here when received; it is run unmodified. K3 = the T364 estimator (@e433919d, T_hi 6) as the
  comparator.
- **Protocol:** identical to T367's truth-blind rule (a9a57359).
  - Models: dev3_s548 ... dev3_s563 (16 unused DEV3 models).
  - K3 on the grid t0 in {1.75, ..., 3.5}. The candidates are the t0s with a K3 estimate in [1e-9, 1e-8], else the 2
    grid points nearest 3e-9.
  - The fix runs on the candidates. Per model, keep the 2 pairs with the largest |log(fix / K3)|. The top 4 models by
    mean disagreement give the **8 pairs**.
  - Both estimators' outputs for all 16 models are committed and pushed **before** the pairs go to T192 for sealed
    conditional-law truths (1e9 per pair on GPU, or CPU).
- **Go: the fix within 2x of the truth on >= 7/8.** Also reported: K3's ratios, |log| comparison, both C, and the
  expected number of events plain MC sees at the fix's C.
- **T369 status (18:43): not run.** The T218 child made no new freeze (T369 results in their README,
  claude/t218-tie-tail-labels). The graded rule is converged on s547. A closure oracle there attributes the 2.1-2.4x
  overshoot to the R1W closure itself:
  - the closure world over-counts the triple-tie sliver 2.2-2.5x;
  - exact lows bring it to 1.5x;
  - v4's random position brings it to 1.04x.
  So there is nothing to fresh-test. dev3_s548+ remain unused.
