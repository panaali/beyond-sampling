# input_bab: certified rare-event bound by INPUT-space branch and bound (T414 kill test)

↑ [m16 approaches](../README.md) · lane: T398 (lead session_01LhvQ3Abu3KY6KE3AELyQUg), branch `claude/t373-event-cone`

**Question.** Can input-space branch and bound certify P(E1(t0 = 2.5)) <= 1e-10 per triple on the LATE triples where
T398's forward-in-time neuron splitting cannot prune? Splitting input coordinates does not depend on step order. The
method uses CROWN bounds of the event margin over the unrolled 10-step net, and charges surviving boxes an exact
Gaussian-mass upper bound.

**Status.** PRE-REGISTERED 2026-10-19 (this section was written and pushed before any T414 run). Owner: Claude
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
