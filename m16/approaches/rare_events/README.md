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

## Results (`python run_est.py; python score.py`, `results/score.txt`)
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
