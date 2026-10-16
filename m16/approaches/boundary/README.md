# Boundary-layer (tie-surface) integration estimator for M_16,10 accuracy

## Question
Can the error volume of M_16,10 be rewritten as integrals over the tie surfaces of the top order statistics
(x_(2)=x_(3), x_(1)=x_(2)), with the normal direction integrated **exactly** through the network's
piecewise-linear structure, so that an estimate beats plain random sampling at equal compute under ARC's
protocol (`m16/arc_eval.py`)?

## Method
Plain-language version. Most errors sit in thin bands next to ties between the 2nd and 3rd largest
inputs (and the 1st and 2nd). Pick a tie surface, e.g. x_a = x_b. Every input near that tie can be written as
a point y on the surface plus a step t along the normal d = (e_a - e_b)/sqrt2, with t ~ N(0,1) independent of
y. Along that normal line the network (ReLU, no biases) is exactly piecewise linear, so the set of t where it
is wrong is a union of intervals we can find exactly. Its Gaussian mass is the "signed band width times the
density" from the brief, but computed exactly rather than linearised. The 8+1-dimensional surface integral
of that mass is then estimated from random surface points.

Unbiased decomposition (zones depend on the input only, so the split holds for any weights):

    error = A + B + P_R * rate_R + (11/1024) * rate_blind
    blind = {x_(2) <= 0}                       (probability exactly 11/1024)
    A     = P(wrong, non-blind, x_(2)-x_(3) < gA)            gA = 0.15
    B     = P(wrong, non-blind, x_(1)-x_(2) < gB, not A)     gB = 0.10
    R     = the rest (non-blind, far from both ties)         P_R = 0.569069 by 1-D quadrature

- **A (2nd vs 3rd).** Surface point: a pair (a,b) chosen uniformly from 45, y ~ N(0,I) projected onto
  x_a = x_b, kept only if the tie sits at ranks 2-3 (one other input above, the rest below). That rejection
  uses inputs only. The surface measure 45 * P(ybar in (o_(2), o_(1))) = 45 * 0.100049 is an exact 1-D
  quadrature. On the normal line, inside |t| < min(rank-preserving half-length, gA/sqrt2) and masked to the
  non-blind part (|t| > -sqrt2 * ybar), we:
  1. do one forward pass at the tie (value + slope), which gives the exact affine logits on the whole linear
     piece containing the tie;
  2. walk outward through the misclassified band adjacent to the tie, piece by piece, until the first
     correct point (exact);
  3. integrate the rest of the last walked piece exactly (no extra passes);
  4. estimate any misclassified mass in the still-unknown part of the zone without bias, using one
     Gaussian-weighted random point on the same line (mass x indicator).
- **B (1st vs 2nd).** The same construction on x_(1) = x_(2) (surface measure 45 * 0.060608). The segment is
  clipped to keep r1 and r2 the top two, stay outside zone A and stay non-blind.
- **R and blind.** Plain indicator sampling inside the zone. Inputs are drawn by rejection, which is
  model-free.
- **Allocation.** Two phases. A pilot uses 25% of the budget at default fractions, which came from a pilot
  on the original model only. The rest of the budget is split by a Neyman rule (share proportional to
  weight x s.d. x sqrt(cost)) using that model's own pilot. All samples are used.

Files: `zones.py` (zones, surface sampling, full exact zone walk), `zones2.py` (local band walk + one-point
remainder, cost accounting), `estimator.py` (`est_boundary(model, budget, rng)`), `eval_arc.py`
(harness run), `bias_orig.py`. Earlier steps, kept: `seg.py`/`tie23.py` (full rank-preserving segment walk),
`local.py`/`bands.py` (model-defined bands + remainder; superseded), `tune*.py`, `pilot.py`, `decomp.py`.

**Deductive vs sampling.** The normal-direction integral (the boundary-layer width/mass) is deductive: exact
piecewise-linear root finding. So are the zone probabilities and surface measures (closed-form 1-D
quadratures, 11/1024). The 9-dimensional integral over the tie surface, the one-point remainders, and the
R and blind terms are random sampling. **Overall this is a sampling estimator (conditional Monte Carlo /
Rao-Blackwellisation along mechanistically chosen directions), not a deductive estimate.**

## Status
done (see Results). RESULTS_PLACEHOLDER
