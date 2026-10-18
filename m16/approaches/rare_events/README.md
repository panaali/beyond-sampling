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
