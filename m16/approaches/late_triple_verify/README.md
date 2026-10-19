# late_triple_verify: independent re-check of the late-triple certification failure with library verifiers (T418)

↑ [m16 approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · T414 code: branch
`claude/t192-gpu-cost-law`, `m16/approaches/input_bab/`

**Question.** T398 (forward neuron splitting) and T414 (input-space BaB with an own float64 CROWN) both fail to certify
P(E1(t0 = 2.5)) on late triples such as (3,7,8). Is that failure real (AGENTS rule 5 re-check with different code), and
do the strongest standard verifiers, auto_LiRPA alpha-CROWN and alpha-beta-CROWN (input and ReLU splitting), change the
verdict?

**Status.** Pre-registered 2026-10-19 (this section was written and pushed before any scored run). Owner: Claude
session_011FA8RfY3APG3rkDG2GxScR (own container). DRI session_01CD3g9Cvc3qk3pM11eyXXzj. CPU only.

## Pre-registration (fixed before any scored run)

**Setup (independent code).**
- `net.py`: M_16,10 unrolled as a torch module on T414's homogeneity-reduced coordinates (x = g y, g = x_p2 - x_p3 >=
  2.5, y_p3 = u_r, y_p2 = u_r + 1, y_p1 = u_r + 1 + u_a, y_l = u_r - u_l, u_a, u_l >= 0; root box u_r in [-4, 4], the
  rest in [0, 4]; domain tail 6.2e-12 charged once). Output: the 9 margins L_j - L_p3. A box is pruned iff some
  certified lower bound is > 0. Weights from `m16/common.py` (read only). Own slab sampler, own radial charge (T414's
  formula), own near-exact box mass (2-D Gauss-Legendre over (u_r, g), u_a and u_l integrated exactly).
- Libraries: auto_LiRPA and alpha-beta-CROWN from `github.com/Verified-Intelligence/alpha-beta-CROWN` (commit b58d83c,
  auto_LiRPA submodule e393035), torch 2.11 CPU, float64 everywhere (`general/double_fp`). Python 3.13 (the packages
  pin 3.11; they import and run).
- Feasibility trials (not scored, before this section): auto_LiRPA CROWN ~180 boxes/s, alpha-CROWN (20 iterations)
  ~12 boxes/s on one core. alpha-beta-CROWN input-split BaB with the 9-row OR spec runs (~800 domains/s).
  ReLU-split BaB (beta-CROWN) crashes on the 9-row OR spec, so it is run on single specs 'margin_j > 0' (top 2 j by
  CROWN lower bound); it is slow (~20 CPU-s for a 4-domain proof). Mixed input + ReLU splitting
  (`branching_input_and_activation`, marked experimental) crashes on both spec forms: recorded as not runnable.

**Bounders.** C = auto_LiRPA CROWN; A = auto_LiRPA alpha-CROWN (20 iterations, library default: intermediate bounds
fixed from CROWN); AF = alpha-CROWN with `fix_interm_bounds = False` (intermediate bounds optimised too; looseness test
only, if it runs); ABC-in = alpha-beta-CROWN input-split BaB on the OR spec; ABC-relu = beta-CROWN ReLU-split BaB on the
top-2 single specs.

**E1 Re-check (scored).**
- (a) Root u-box: C lower bounds vs T414's `margins()` (T414's own code, unchanged except the import path).
- (b) Boxes from a 600 CPU-s re-run of T414's `ibab.py 3,7,8` (its own search, here): 2,000 random open boxes plus
  its reservoir of pruned boxes. PASS iff max |lb_C - lb_T414| <= 1e-6 (1 + |lb|) on every box and spec.
- (c) Looseness on the 200 top-charge open boxes of (b): sampled minimum (2,000 uniform points per box, own forward)
  minus the lower bound, for the best spec, under C, A and AF; pruned count under each, and under ABC-in (timeout 10 s)
  and ABC-relu (10 s per spec). T414 reported a median of 176 margin units for its CROWN (10-min trial).

**E2 Certified width (scored, secondary projection).** 200 exact slab points of (3,7,8) (own sampler, seed 418; event
points reported separately). For each, the largest half-width w of the u-cube (clipped to u_a, u_l >= 0) that the
bounder certifies: geometric bisection, 20 steps for C and A; ABC-in on the first 60 points, 6 steps in [w_A, 4 w_A]
(timeout 10 s, also recording BaB domains visited); ABC-relu on the first 20 points, 4 steps in [w_A, 4 w_A]. The
covering estimate is T414's: E_slab[1 / ((2w)^9 p_u)] over the highest-density 99.8% of slab mass, ideal cubes and
w halved; for ABC also multiplied by the mean domains visited per certified cube (the number of bound evaluations).
C should reproduce T414's median w = .103.

**E3 alpha BaB (primary scored run).** `lbab.py 3,7,8 OUT 7200 --stages crown,alpha --K 256`: T414's search
(best-first by radial charge, pop K, bisect along argmax_d |c_d| width_d of the best spec's linear lower bound), bounds
C then A on C's survivors. 2 CPU-h, one thread.

**E4 beta phase (scored).** Resume from E3's open set: `--stages crown,alpha,abc --abc-mode M --abc-timeout 2 --K 64`,
2 CPU-h, with M = whichever of ABC-in / ABC-relu certifies the larger median width in E2 (rule fixed now).

**E5 (if time).** E3 on (9,8,2).

**Charge and reporting.** Primary charge = T414's rigorous radial charge (sum over open boxes + tail); near-exact
box mass reported alongside (float, not a certificate). Reported: looseness, pruned fraction, charge-vs-boxes curves,
projections. Boxes = children evaluated; for E4 also boxes + abcrown BaB domains (bound evaluations).
Projection: log-log fit of charge vs boxes over the last decade of the curve, extrapolated to 1e-10.

**Kill line (DRI's, fixed).** KILL if the projected boxes to reach 1e-10 exceed 1e9 per triple, or the charge stalls
above 1e-9 (less than a 2x decrease over the last decade of boxes). GO only if a projection lands below 1e9 boxes per
triple (E3 or E4; for E4 counted in boxes + domains).

**Alternatives written before results.**
- A failure under A may be the library default of fixed intermediate bounds (the 10-step compounding sits in the
  intermediate layers). Check: AF in E1c.
- abcrown's own split heuristics may beat T414's; E2's ABC numbers use its heuristics, E3 uses T414's.
- If E2 covering estimates fall below 1e9 while E3/E4 curves do not, the obstruction is the search (ordering or split
  rule), not the bounds; that would be reported as GO-able by a different search, not as a kill of the bounds.
- A re-check mismatch in E1 could be a T414 bug or an auto_LiRPA option difference (e.g. lower-slope rule); checked by
  comparing on single-unstable-neuron cases if it happens.

**Deviation D1 (2026-10-19, written before its run; SECONDARY, not used for the kill decision).** Mid-run, E2 gave an
alpha-CROWN covering estimate below 1e9 (ideal 2.9e5, halved 1.5e8), while E3's near-exact mass sat flat at ~2.8e-8
with pruning almost stopped. This is pre-registered alternative 3 (search vs bounds). One secondary run:
E3's search with the pop order switched to the near-exact box mass (`--order quad`), everything else unchanged, 2 CPU-h,
`lbab.py 3,7,8 OUT 7200 --stages crown,alpha --K 256 --order quad`. Reported: the near-exact curve, pruned fraction and
projection. The radial charge stays the rigorous figure.

## Method
- `net.py`: model, coordinates, slab sampler, charges. `lbab.py`: BaB with library bounds.
- Other scripts are added with their results.
