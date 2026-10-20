# halfspace_charge: halfspace charging inside input-space branch and bound (T437 kill test)

↑ [m16 approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · builds on T414
(`claude/t192-gpu-cost-law`, `m16/approaches/input_bab/`)

**Question.** Late triples such as (3,7,8) cannot be certified for E1(2.5) because undecided cells are charged their
whole mass (T398: event feasible on every prefix; T414: input-space CROWN prunes 5.7% of boxes and an honest box-mass
charge stalls at ~45% of the slab mass). Does charging each undecided box only the Gaussian mass of the part where its
own CROWN linear bounds still allow the event (PROVEN's principle, Weng et al. 2019, inside Boetius-style
probabilistic branch and bound) make a certified per-triple bound <= 1e-10 feasible?

**Status.** DONE 2026-10-20: **KILL by the pre-registered line (unverified).** Pre-registered 2026-10-19 (b10af884,
pushed before any charge curve was computed). Owner: Claude session_01WmLP3HR82mj4TLjKCRcE93. DRI session_01CD3g9Cvc3qk3pM11eyXXzj.
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

## Method
- `hcharge.py`: the search (T414's coordinates, CROWN and best-first loop, copied unchanged in `t414_ibab.py`) with the
  rigorous tilted / untilted rectangle charge, lambda optimisation, `SPLIT=widest` (D1) and `FBBT=1` (D2).
- `validate.py`: validation (a), (b). `diag_q.py`: halfspace-cut statistics and CROWN soundness on sampled points.
- `qfrac.py`: float estimate of sum m(Q(B)) vs sum m(B) on 5,000 random open boxes (near-exact box mass x sampled
  fraction in Q). `cell_probe.py`: T398's forward-cell setting (gate patterns of steps 1..k exact, steps k+1..10
  CROWN with LP intermediate bounds over cell ∩ root box, own Gibbs slab sampler); float, not a certificate.
- `t414_qcharge.py`: T414's near-exact box mass (float), used only for checks.

Implementation fix before the result (recorded): the first primary run gave a NaN tilted charge at iteration ~120
(lambda optimisation produced -inf + inf); a guard falls back to lambda = 0 / the untilted bound for such cells
(9eee5bca), and the run was restarted from scratch (old log `data/t437/t437_378_nanbug.log`). A container restart
then interrupted the new run; it resumed from its 10-min checkpoint (`--resume`), losing ~10 CPU-min of work.

Runs (1 thread, `ulimit -v 2500000`, data in `data/t437/`, not in git):
- primary: `python hcharge.py 3,7,8 data/t437/t437_378.json 7200 1024`
- D1: `SPLIT=widest python hcharge.py 3,7,8 data/t437/t437_378_widest.json 3600 1024`
- D2: `SPLIT=widest FBBT=1 python hcharge.py 3,7,8 data/t437/t437_378_fbbt.json 3600 1024`
- cell probe: `python cell_probe.py 3,7,8 {3,4} 20000 OUT.json`
- diagnostics: `python diag_q.py CKPT 200 2000`, `python qfrac.py CKPT 5000 500`, `python validate.py CKPT 50 100000`

## Results ([measured], unverified)

**Validation.**
- (a) Rigorous untilted charge >= T414's near-exact box mass on 50/50 random open boxes (ratio median 7.9, min 1.7):
  PASS. Root box: 2.65e-5 >= 5.5975e-8 (PASS) but 473x, not within 2x: **FAIL** of the tightness part at the
  run grid NR = 16, NG = 40 (6.5x at 64 x 160). The rectangle bound is loose on wide boxes (r cells of width 0.5).
- (b) Tilted charge >= MC(Q(B)) - 3 s.e. on 50/50 boxes: PASS.
- (c) CROWN halfspaces never above a sampled margin (all 2,000 x 200 points in each `diag_q` call): PASS. 0 sampled
  events in any open box.

**Pre-registered primary run, (3,7,8), T414's margin-steered split, 2 CPU-h:**

| boxes | pruned | halfspace charge (rigorous) | untilted control | slope (last decade) | projected boxes to 1e-10 |
|---|---|---|---|---|---|
| 3.74e6 | 7.1e4 (1.9%) | 5.452e-7 | 5.452e-7 | +0.49 (rising) | none (no fall) |

Curve (boxes: charge): 1.5e3: 1.15e-6; 4.5e5: 1.88e-7; 9.0e5: 2.60e-7; 1.8e6: 3.85e-7; 2.7e6: 4.72e-7; 3.74e6: 5.45e-7.
The charge rises because each child's rectangle bound carries its own discretisation slack while the margin-steered
split rarely narrows u_r. **KILL** (no fall; stalls above 1e-9). T414's radial curve: 8.7e-7 at 9.0e6 (slope -.49);
T414 D1 near-exact: 2.51e-8 (stall).
- Why: on the top-200 open boxes (by box charge) the CROWN linear lower bound's MAXIMUM over the box is <= 0 on 170
  of them (quantiles of the best spec: median -2,954, p90 +3.8 margin units); its variation across a box is ~3-20
  units against offsets of 100-4,000. The halfspaces contain those boxes entirely.
- Scale (float): near-exact mass of all open boxes 2.45e-8 (44% of the slab, T414 D1's floor); Q(B) holds 99.8% of
  it. An exact halfspace charge would gain at most 1.002x on this search.

**T398's cell setting (secondary probe, float, 20,000 Gibbs slab samples of (3,7,8)):**

| k | cells with samples | slab mass pruned | cells any halfspace cuts | max over cells of the linear bound at its samples |
|---|---|---|---|---|
| 3 | 247 | 0 | 0 | -205 |
| 4 | 605 | 0 | 0 | -101 |

Same picture: the cell's own linear bounds are 100-300 margin units below 0 everywhere on its samples.

**Deviation D1 (secondary): widest split, plain CROWN, 1 CPU-h.**

| boxes | pruned | halfspace charge | untilted | slope | projected boxes to 1e-10 |
|---|---|---|---|---|---|
| 2.15e6 | 4.4e5 (20%) | 1.449e-8 | 1.464e-8 | -0.46 | 1.1e11 |

- The halfspaces now cut boxes near the prune boundary: on the top-200 boxes the slab-mass fraction inside Q(B) has
  median 0 and mass-weighted 5e-4; the tilt bound captures little of it (-1% on the total; 108k of 636k open boxes
  have tilted < untilted, saving 6% on them).
- But on 5,000 uniform random open boxes Q(B) holds 82% of the near-exact mass (sum 6.0e-9 -> 4.9e-9): an exact
  halfspace charge would gain at most ~1.2x. The near-exact open mass 6.0e-9 vs the rigorous 1.45e-8 (2.4x).

**Deviation D2 (secondary): D1 + FBBT box hull of Q(B), 1 CPU-h.**

| boxes | pruned | halfspace charge | untilted | slope | projected boxes to 1e-10 |
|---|---|---|---|---|---|
| 1.81e6 | 3.6e5 (19.9%) | 1.240e-8 | 1.242e-8 | -0.46 | 6.5e10 |

- 1.3x below D1 at equal boxes (D1 at 1.8e6: 1.60e-8); near-exact open mass 5.0e-9, Q(B) holds 94% of it.
- Against T418's D3 (alpha-CROWN, widest split, mass order; claude/t418-late-triple-verify 04cc345b): its near-exact
  mass falls with slope -1.13 to 1.6e-9 at 1.37e5 boxes, its rigorous radial charge is 3.85e-7. Our rigorous rectangle
  charge (1.24e-8 at 1.8e6, 2.5x the near-exact mass on these boxes) is far tighter than the radial charge, but our
  plain-CROWN search falls only at slope -0.46. Halfspaces do not change either slope.

**Certified superset (separate, labelled statement; float CROWN, see Failure modes).** From D2: the event E1(2.5) on
(3,7,8) lies inside S = union of the 543,789 open FBBT-tightened boxes (each ∩ its Q(B)) plus the domain tail, with
**m(S) <= 1.24e-8** (22% of the slab mass 5.6e-8). Hence P(E1 on (3,7,8)) = m(S) P(E | S) with P(E | S) to be
sampled inside S; to reach 1e-10 that way, P(E | S) must be estimated at ~1e-2 or below. This is a superset mass,
not a bound on P(E).

## Compute accounting
CPU only, 1 thread per process, <= 2 GB (peak RSS ~0.65 GB): primary 2 CPU-h (+ ~10 min lost to the restart and
~4 min of the NaN run), D1 1 CPU-h, D2 1 CPU-h, cell probes ~6 CPU-min, trials, validation and diagnostics ~20 CPU-min.
Total ~4.6 CPU-h. One box evaluation = one CROWN pass (~1.5e3 FPE, as T414) plus the charge (~2.5 ms: lambda
optimisation 30 Adam steps on 60 nodes, two rectangle bounds on 640 rectangles). Sampled checks ~3e7 forward passes.
Colab: none.

## Claims
- [measured] Halfspace (tilted) charging inside T414's input-space BaB on (3,7,8) gives the same charge as the uncut
  box charge to 4 digits after 3.7e6 boxes; KILL by the pre-registered line.
- [measured] Cause: with T414's split, CROWN's linear lower bounds sit 100-4,000 margin units below 0 over most open
  boxes, so each halfspace contains its box; the same holds in T398's k = 3, 4 cells. Exact halfspace masses would
  gain <= 1.002x there.
- [measured, secondary] With a widest split the halfspaces do cut boundary boxes deeply, but those boxes carry little
  of the open mass: an exact halfspace charge would gain ~1.2x (FBBT hull: 1.3x measured). Projections stay at
  6.5e10-1.1e11 boxes with plain CROWN.
- [inferred] Halfspace charging is not the lever for late triples; the lever is the pruning rate of the search (T418:
  alpha-CROWN + widest split) plus a tight rigorous box mass (T455). Halfspaces at best add a constant factor ~1.2-1.3x.
- [measured, by-product] The rectangle bound in `hcharge.py` (closed-form inner Phi differences, monotone sup bounds
  on (u_r, g) rectangles, analytic tail) is a rigorous box mass about 2.5x the near-exact mass on D1/D2's open boxes
  (8x on the trial boxes), versus the radial charge's 1e2-1e7x. Possibly useful to T455 (not duplicated here).

## Alternatives not ruled out
- Halfspaces from alpha-CROWN (tighter offsets) on T418's search: the boundary boxes would be cut more, but D1/D2 show
  the cut mass is a small share of the open mass; not measured with alpha-CROWN.
- An exact (not Chernoff) halfspace mass: bounded above by the qfrac estimates (gain <= 1.2x on D1).
- Other split heuristics combined with charge-ordered search (T418's mass order was not used here; we ordered by our
  rigorous charge).
- (9,8,2) was not run.

## Failure modes
- CROWN is float64 and not outward-rounded; the halfspaces get a heuristic relative slack 1e-7. The mass bounds are
  rigorous given those halfspaces and assumption A1 (libm erfc / exp accuracy).
- The rectangle bound is loose on boxes wide in u_r (473x on the root box), which makes the primary curve rise.
- The lambda optimisation can fail numerically; it then falls back to the untilted bound (sound, not tight).
- qfrac and cell_probe numbers are float estimates from samples (500 per box; 20,000 slab points).
