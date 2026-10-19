# halfspace_charge: halfspace charging inside input-space branch and bound (T437 kill test)

↑ [m16 approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · builds on T414
(`claude/t192-gpu-cost-law`, `m16/approaches/input_bab/`)

**Question.** Late triples such as (3,7,8) cannot be certified for E1(2.5) because undecided cells are charged their
whole mass (T398: event feasible on every prefix; T414: input-space CROWN prunes 5.7% of boxes and an honest box-mass
charge stalls at ~45% of the slab mass). Does charging each undecided box only the Gaussian mass of the part where its
own CROWN linear bounds still allow the event (PROVEN's principle, Weng et al. 2019, inside Boetius-style
probabilistic branch and bound) make a certified per-triple bound <= 1e-10 feasible?

**Status.** PRE-REGISTERED 2026-10-19, before any run of the method (this section is committed and pushed before any
charge curve is computed). Owner: Claude session_01WmLP3HR82mj4TLjKCRcE93. DRI session_01CD3g9Cvc3qk3pM11eyXXzj.
CPU only.

## Pre-registration (fixed before any result)

**Setting.** T414's box setting, unchanged unless stated: M_16,10, x ~ N(0, I_10), event
E1(2.5) = { x_p1 >= x_p2 >= x_p3 + 2.5, x_p3 >= every low, argmax_j L_j = p3 }, triple (p1,p2,p3) = (3,7,8).
Homogeneity-reduced coordinates x = g y(u), y = P u + q, g = x_p2 - x_p3 >= 2.5, u = (u_r, u_a, u_l[7]),
root box u_r in [-4, 4], u_a, u_l in [0, 4]; domain tail 6.2e-12 charged once. Same CROWN (float64, own
implementation, copied from T414's `ibab.py`), same prune rule (some spec j != p3 has CROWN lower bound > 0), same
search (best-first by charge, K = 1024 boxes per round, margin-steered bisection), same 2 CPU-h budget on 1 core.
I chose the box setting rather than T398's k = 3-4 forward cells because T414's curve is the stated comparison and the
code is reusable as is.

**The only change: the charge.** For an open box B with CROWN lower bounds margin_j(u) >= a_j . u + b_j (j != p3,
valid on B), the event inside B lies in Q(B) = B ∩ { u : a_j . u + b_j <= 0 for all j }. The charge is a rigorous
upper bound on the slab mass of Q(B):
- For any lambda >= 0 (9 weights), 1[Q] <= 1[B] exp(-sum_j lambda_j (a_j . u + b_j)) (exponential tilt, a Chernoff
  bound restricted to the box; lambda = 0 gives the box mass itself).
- For fixed (u_r, g) the box coordinates u_a and u_l enter independent 1-D Gaussians, so the tilted inner integrals
  are closed-form Phi differences (shifted by lambda-terms / g). This leaves a 2-D integral over (u_r, g).
- The 2-D integral is bounded from above on a grid of rectangles in (u_r, g >= 2.5) by monotone interval bounds of
  every factor (no quadrature error), plus an analytic bound on the tail g > G_max. On each rectangle the integrand
  bound is min(tilted, untilted). All final sums are inflated outward by a relative 1e-10 (covers float64 rounding
  under assumption A1 of H-C1: libm/torch exp and erfc accurate to a few ulp).
- lambda is chosen per box by a few steps of projected gradient descent on a coarse, non-rigorous quadrature of the
  log of the tilted mass (it is convex in lambda); any lambda >= 0 keeps the bound valid.
- CROWN is float64 and not outward-rounded (as in T414). To keep the halfspaces conservative, each b_j is lowered by
  a slack 1e-7 x (|b_j| + sum_i |a_ji| max_B |u_i| + 1) before charging. Full CROWN outward rounding is deferred to a
  GO (as T414 / T400 did); this is the one non-proved link and is stated as such in every claim.

**Reported.**
1. The rigorous halfspace charge (sum over open boxes + tail) vs boxes evaluated, on the same axes as T414's
   pre-registered curve (radial charge) and T414's D1 curve (near-exact box mass, stall at 2.51e-8).
2. In the same run, the rigorous UNtilted box charge (lambda = 0, same rectangle bounds) summed over the same open
   boxes: the control that isolates what the halfspaces buy.
3. Share pruned, boxes per second, projected boxes and CPU-h to reach 1e-10 (log-log fit over the last decade).
4. **Separate labelled statement:** the certified superset S = union of open Q(B) plus the domain tail, with
   m(S) <= final charge. Then P(E1 on (3,7,8)) = m(S) P(E | S), and P(E | S) can be sampled inside S (boxes picked in
   proportion to charge, rejection to Q(B)). This is reported as a superset mass, never as a bound on P(E).

**Kill line (same as T414's, binding).** KILL if the projected boxes to reach 1e-10 exceed 1e9, or if the charge
stalls above 1e-9 (stall = less than a 2x decrease over the last decade of boxes). GO otherwise. Also reported, not
binding: the novelty-review lines (GO if charge < 1e-9 at <= 1e5 boxes; KILL if still > 10% of the slab mass
5.6e-8 with no fall per decade).

**Validation (could fail; recorded pass/fail).**
- (a) The rigorous untilted charge of the root box >= the triple's slab mass 5.5975e-8 (T414's independent integral)
  and within 2x of it; on 50 random boxes the rigorous untilted charge >= T414's D1 near-exact box mass (`qcharge.py`).
- (b) On 50 random open boxes, the rigorous halfspace charge >= a 1e6-point MC estimate of the slab mass of Q(B)
  minus 3 s.e. (checks the tilt and the rectangle bounds).
- (c) Soundness of the halfspaces: forward-evaluated event points sampled in the boxes (if any) all satisfy
  a_j . u + b_j <= margin_j; CROWN lower bounds <= sampled margins on 30 random boxes.

**Secondary (not for the decision).** MC estimate of the true slab mass of Q(B) on the top-200 open boxes at the end,
to separate tilt looseness from halfspace looseness (as T414's D1 separated charge looseness from pruning). If time
allows, the same run on (9,8,2).

**Alternatives written before results.**
- The halfspaces may cut little: CROWN is loose by a median 176 margin units (T414), so a_j . u + b_j may be <= 0 on
  most of each box. Check: secondary MC of m(Q(B)) / m(B).
- The halfspaces may cut a lot but the tilt bound may be loose (Chernoff loses a polynomial factor, more for a
  9-halfspace intersection). Check: secondary MC vs the tilted charge.
- Any fall may come from the best-first order now chasing halfspace mass rather than box mass. Check: the untilted
  control on the same boxes.

**Deviation D1 (2026-10-19 ~22:10 UTC, written before its run; SECONDARY, not used for the kill decision).** The DRI
relayed T418's result (branch `claude/t418-late-triple-verify` 04cc345b): T414's stall came from its margin-steered
split; with a widest-dimension split, mass order and alpha-CROWN, the near-exact mass on (3,7,8) projects 1.6e6 boxes
to 1e-10, and the remaining blocker is a rigorous charge on wide boxes. Halfspaces can only cut boxes whose CROWN
bound is near 0 somewhere, which a widest split may produce more often. One secondary run: identical to the primary
except `SPLIT=widest` (bisect the widest u-dimension), plain CROWN (no alpha), 1 CPU-h:
`SPLIT=widest python hcharge.py 3,7,8 OUT 3600 1024`. Reported against both T414's and T418's (D3) curves: tilted vs
untilted charge, pruned share, and the share of open boxes some halfspace cuts. Rigorous box mass is T455's lane;
this run only reuses the rectangle bound already written here.

**Deviation D2 (2026-10-19 ~23:00 UTC, written after D1's result and before any D2 run; SECONDARY).** D1's open boxes
showed the halfspaces cut hard where the tilt bound does not see it: on its top-200 open boxes (by box charge) the
sampled slab-mass fraction inside Q(B) is median 0, max 0.024, mass-weighted 5e-4, while the tilt saved only ~1% of
the total charge. So the halfspace idea is limited by the Chernoff bound, not by the halfspaces. D2 replaces each
box B by a rigorous box hull of Q(B): interval bound tightening (FBBT) along every halfspace a_j.u + b_j <= 0,
3 passes over the 9 specs and 9 dimensions (v_d <= (-b_j - sum_{i != d} min_B a_ji v_i) / a_jd for a_jd > 0, the
mirror for a_jd < 0), with the new bounds moved outward by 1e-9 x (1 + |bound|). An empty hull prunes the box. The
tightened box replaces B in the search (the event inside B lies in it, so this is sound), then gets the same CROWN,
tilt charge and rectangle bounds. Run: `SPLIT=widest FBBT=1 python hcharge.py 3,7,8 OUT 3600 1024`, 1 CPU-h, same
reporting as D1 plus the hull's volume ratio. Also a float, non-rigorous estimate for scale (`qfrac.py`): on 5,000
uniform random open boxes of D1, sum qmass(B) x (sampled fraction in Q(B)) vs sum qmass(B).
