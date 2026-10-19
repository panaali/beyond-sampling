# input_bab: certified rare-event bound by INPUT-space branch and bound (T414 kill test)

↑ [m16 approaches](../README.md) · lane: T398 (lead session_01LhvQ3Abu3KY6KE3AELyQUg), branch `claude/t373-event-cone`

**Question.** Can input-space branch and bound certify P(E1(t0 = 2.5)) <= 1e-10 per triple on the LATE triples where
T398's forward-in-time neuron splitting cannot prune? Splitting input coordinates does not depend on step order. The
method uses CROWN bounds of the event margin over the unrolled 10-step net, and charges surviving boxes an exact
Gaussian-mass upper bound.

**Status.** DONE 2026-10-19: **KILL by the pre-registered line (unverified)**. Pre-registered 2026-10-19 (the
pre-registration section was written and pushed, 7c106fb9, before any T414 run). Owner: Claude
session_016BGedpds19gq95yjWmDMn5 (T192 child). DRI session_01CD3g9Cvc3qk3pM11eyXXzj. CPU first; Colab only if
CPU-bound, cap 10 units. Code lives on branch `claude/t192-gpu-cost-law`, the only branch this session pushes to (the
DRI named `claude/t414-input-bab`; the DRI or root can merge the folder from here).

## Pre-registration (fixed before any result)
**Event.** M_16,10, x ~ N(0, I_10), triple (p1, p2, p3).
- E1(t0) = { x_p1 >= x_p2 >= x_p3 + t0, x_p3 >= x_l for the 7 lows, argmax_j L_j = p3 }, L = Wo h_10, t0 = 2.5.
- Triples:
  - (3, 7, 8): worker B's censored case; b = 7.
  - (9, 8, 2): one b = 8 triple, chosen by this rule before any run: second-smallest position 8, p3 early. No
    look at its data before choosing.

**Coordinates (homogeneity reduction).** The net has no biases, so L(g y) = g L(y) for g > 0 and the event depends on
the direction only.
- Write x = g y with g = x_p2 - x_p3 >= t0, and parametrise y by u in R^9:
  - y_p3 = u_r, y_p2 = u_r + 1, y_p1 = u_r + 1 + u_a,
  - y_l = u_r - u_l for the 7 lows (sorted order);
  - constraints u_a >= 0, u_l >= 0 (exactly the ordering).
- So y = P u + q is affine. The map (u, g) -> x has Jacobian g^9 (unimodular linear part), and the slab is the exact
  product { u in R x [0, inf)^8 } x { g >= t0 }.
- Root domain: u_r in [-U, U], u_a, u_l in [0, U], U = 4.
- Outside the root domain the event mass is at most 2 Phi(-U t0) + 8 Phi(-U t0 / sqrt 2) = 6.2e-12. This bound uses
  |x_p3| > U g >= U t0 and differences of two N(0, 1) variables, and is charged once.

**Charge of a box** (rigorous upper bound on the slab mass over the box, event ignored):
  mass(box) <= vol(box) * 16 (2 pi)^-5 m^-5 Gamma(5, m t0^2 / 2),
- m = sum_i min over the box of y_i(u)^2: a lower bound of min |y|^2 over the box, taking each coordinate's
  interval minimum separately.
- Gamma(5, w) = 24 e^-w sum_{k<=4} w^k / k!.
- Derivation: phi_10(g y) <= (2 pi)^-5 exp(-g^2 m / 2), then the integral over g >= t0 of g^9 exp(-g^2 m / 2) =
  16 m^-5 Gamma(5, m t0^2 / 2).

**Prune rule.** A box is pruned iff, for some j != p3, a CROWN lower bound of L_j(y) - L_p3(y) over y in P box + q
is > 0.
- The sign is invariant to g.
- CROWN: own batched implementation, float64.
  - Back-substitution through all steps for every pre-activation bound.
  - Triangle upper relaxation; adaptive lower slope (1 if u > -l, else 0).
  - Concretisation over the u-box.
- auto_LiRPA is not used; it adds an install risk and nothing this tiny RNN needs.

**Search.**
- Best-first by charge: each round pops the K open boxes with the largest charge and bisects each one.
- Split dimension: the dimension d maximising |c_d| * width_d, where c is the u-coefficient vector of the spec j
  with the largest lower bound in that box.
- Children are evaluated: pruned, or kept with their charge.
- Total charge = sum of the open boxes' charges + 6.2e-12 (domain tail).
- Budget per triple: 2 CPU-h on 1 core (torch float64, <= 2 GB), or 4e6 open boxes, whichever comes first.

**Validation (could fail; recorded pass/fail).**
- (a) The root charge >= the triple's slab mass 5.6e-8. The charge formula is also checked against a 1e7-sample MC
  of the slab mass on 20 random boxes: charge >= MC - 3 s.e.
- (b) Soundness: 1e6 uniform points from a reservoir of pruned boxes are forward-evaluated; none may be events.
- (c) The CROWN bounds are checked on random boxes against sampled margins: the lower bound must not exceed any
  sampled value.

**Reported.**
- The charged-mass curve vs boxes evaluated, per triple.
- The share of boxes pruned.
- Boxes per second.
- Projected boxes and CPU-h to reach a total charge <= 1e-10, from a log-log fit over the last decade of the curve.

**Kill line (from the DRI, fixed).** KILL if the projected boxes to reach 1e-10 exceed 1e9 per triple, or if the
charged mass stalls above 1e-9 (stall = less than a 2x decrease over the last decade of boxes). GO otherwise. A GO
moves to rigour (outward rounding / T400 verification) and alpha-optimised bounds. None of that is done in the kill
test.

**Alternatives written before results.**
- A stall may come from loose CROWN bounds (wide boxes, 10 steps) rather than from a real event mass. Cheapest check:
  compare with sampled max margins in the stalled boxes.
- A stall may also come from boxes containing true event points, whose charge falls only with volume. Check: count
  open boxes with a sampled event point.
- A slow curve may be the split heuristic rather than the method. Check: widest-dimension splitting on one triple
  (reported as secondary, not used for the kill decision).

**Deviation D1 (2026-10-19 ~10:40 UTC, written before its run; SECONDARY, not used for the kill decision).** A 10-min
trial on (3,7,8) and its diagnostics, made while the pre-registered runs started, showed the radial max-density charge
is 1e2-1e5x looser than the box's exact slab mass (u-MC). It also showed two u_l dimensions never get split, because
their margin coefficients are tiny. Pre-registered alternative 3 (heuristic or charge, not method) therefore gets one
secondary run: the same search with the charge replaced by a near-exact box mass. For fixed (u_r, g), u_a and the
u_l integrate to Phi differences exactly, leaving a 2-D integral over (u_r, g), evaluated by composite Gauss-Legendre
(`qcharge.py`; float, not a certificate). Checks: root box 5.59752e-8 vs the independent slab integral 5.59749e-8;
random boxes agree with u-MC within ~1-3%. Run: `CHARGE=quad python ibab.py 3,7,8 OUT 3600 1024`.

## Method
- `ibab.py`: the search (u-boxes, batched float64 CROWN, best-first by charge, split dimension stored per box).
- `validate_ab.py`: validation (a) and (c).
- `diagnose.py`: (b) plus the pre-registered alternative checks.
- `pretest_w.py`: the T398 lead's certified-width pre-test.
- `qcharge.py`: deviation D1.

Two implementation fixes were made before any pre-registered run (trial only):
- The lower bound m of |y|^2 is the exact 1-D convex minimum over u_r. The first per-coordinate version gave m = 0
  (an infinite charge) and was 1e5x looser.
- The parent's split dimension is cached.

Runs: `python ibab.py 3,7,8 t414_378.json 7200 1024` and `python ibab.py 9,8,2 t414_982.json 7200 1024` (1 thread
each); data in data/t414_*.json (not in git; curves copied below).

## Results ([measured], unverified)
**Validation.**
- (a) PASS. The radial formula, with its Jacobian, matches direct x-sampling of the slab in the root box at
  t0 = 0.3 / 1.0 (3.22e-4 vs 3.15e-4 +- 0.08e-4; 6.8e-5 vs 6.1e-5 +- 0.4e-5). The root charge at 2.5 is 864 >= 5.6e-8.
  Charge >= exact box mass on 20 random boxes, at 50-1e7x the mass: the bound is loose.
- (b) PASS. 997,559 uniform points in 3,793 pruned boxes from the trial: 0 events.
- (c) PASS. CROWN lower bounds <= the sampled minimum on all 30 random boxes.
- The triple's slab mass, by an independent 2-D integral, is 5.59749e-8 (the DRI's 5.6e-8).

**Pre-registered runs (2 CPU-h each, ~1,250-1,300 boxes/s on one core):**

| triple | boxes evaluated | pruned | open | total charge | log-log slope (last decade) | projected boxes to 1e-10 |
|---|---|---|---|---|---|---|
| (3,7,8) | 9.02e6 | 5.1e5 (5.7%) | 4.0e6 | 8.70e-7 | -0.49 | 1.1e15 |
| (9,8,2) | 8.89e6 | 1.04e6 (11.7%) | 3.4e6 | 6.42e-7 | -0.54 | 1.0e14 |

The charge curves have no stall (more than 2x over the last decade). But the total charge is still 11-16x the
triple's whole slab mass, and the projection is 1e5-1e6 times beyond the 1e9 kill line. **KILL on both triples.**

**T398 lead's pre-test** (certified u-cube half-width around exact slab samples; 400 points per triple, 0 events):
- (3,7,8): median w .103, p10 .053, min .026. Covering estimate for 99.8% of slab mass: 2.5e8 boxes with ideal
  centred cubes, 1.3e11 with w halved (a dyadic partition).
- (9,8,2): median w .083, p10 .043. Covering estimate: 2.4e9 / 1.2e12.

**Pre-registered alternatives** (top 200 open boxes by charge, 10-min trial on (3,7,8)):
- No box contains a sampled event point. The open mass is not true event mass.
- CROWN looseness, sampled minimum minus lower bound: median 176, p10 86, p90 397 margin units. 88/200 boxes are
  prunable in principle, with a sampled best-spec minimum > 0.
- The charge is spread thin: the top 200 of 4.7e5 open boxes hold 2.2e-9 of 2.6e-6.
- Two u_l dimensions (lows at positions 0 and 1) are never split. Their margin coefficients are tiny, but they inflate
  the max-density charge.

**Deviation D1 (secondary; near-exact box mass in place of the radial charge, 1 CPU-h per triple, ~265 boxes/s).**

| triple | boxes | pruned | total charge (float) | slope (last decade) |
|---|---|---|---|---|
| (3,7,8) | 9.65e5 | 5.1e4 | 2.51e-8 (45% of the slab mass) | -0.04 (stall) |
| (9,8,2) | 9.54e5 | 6.3e4 | 2.85e-8 (51%) | -0.10 (stall) |

With an honest charge, the curve stalls at roughly half the slab mass, 250-290x above the 1e-10 target. In the
pre-registered runs the charge fell only because a loose bound tightened as boxes shrank. The mass that matters falls
only through pruning, and CROWN at these box sizes barely prunes. The kill therefore does not depend on the charge.

## Compute accounting
CPU only, local, 1 thread per process, <= 2 GB:
- pre-registered runs: 2 x 2 CPU-h
- D1: 2 x 1 CPU-h
- 10-min trial
- validation, diagnostics and pre-test: ~15 CPU-min

Colab: none. Model evaluations: the CROWN cost per box is ~1e6 FLOPs (~1.5e3 FPE); the sampled checks used ~1e7
forwards.

## Claims
- [measured] Input-space BaB with own CROWN over homogeneity-reduced u-boxes, best-first by charge with margin-steered
  bisection: after ~9e6 boxes per late triple the radial charge is still 11-16x the slab mass. The log-log slope is
  ~-0.5, which projects 1e14-1e15 boxes to reach 1e-10. KILL by the pre-registered line on (3,7,8) and (9,8,2).
- [measured] With near-exact box masses the charge stalls at ~50% of the slab mass after ~1e6 boxes.
- [measured] CROWN certifies u-cubes of median half-width ~0.08-0.10 around slab points. Ideal coverings need
  2.5e8-2.4e9 boxes per late triple, and realistic partitions ~1e11-1e12.
- [measured] The pruning is sound on every check (no events in 1e6 points from pruned boxes); the open boxes carry no
  sampled event mass. The obstruction is bound looseness (median 176 margin units), not true events.
- [inferred] Like forward neuron splitting (T398), plain input splitting cannot certify late triples at 1e-10. The
  10-step |Wh| compounding that limits interval bounds (T344) also limits CROWN over 9-D boxes.

## Alternatives not ruled out
- alpha-optimised (alpha-CROWN) and lambda-combined specs instead of fixed adaptive slopes and max over single
  specs. These typically tighten the bounds 2-10x in margin units; given the median looseness of 176 and the 9-D box
  count, it is unlikely they close a 1e5x gap, but this was not measured.
- Mixed input and neuron splitting (beta-CROWN), or splitting only the late input coordinates. Early lows have tiny
  margin coefficients.
- A different split heuristic: charge-weighted, or the widest dimension (pre-registered alternative 3 was only partly
  run, via D1).
- Charging boxes at a coarser level of the tree (keeping parents) does not change the D1 floor.

## Failure modes
- The radial charge is 1e2-1e7x looser than the box mass on wide boxes. Its curve mostly measures that looseness.
- Float only; no outward rounding (rigour was scheduled after a GO).
- Covering estimates from 400 points are noisy and tail-dominated.
