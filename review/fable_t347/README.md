# Fable note for T347: is the product rule's +.57 pp overstatement derivable model-free?

Reviewer: Fable (DRI sub-agent, reading and arithmetic only; no model was run). Date 2026-10-18.
Sources read: `origin/claude/t292-oracle-floor:m16/approaches/budget_t317d/README.md` (T324, Addenda A, B1-B3) and its
`budget.py`, `addendum.py`, `corner.py`, `results.json` (80 DEV2), `corner.json` / `corner.log` (10 DEV2);
`origin/ccr-6a61f25c-wn80ii` INDEX rows T317, T324, T330, T333, T339, TASKS row T347; T339 section of
`closure_floor/README.md`; `cont_d.py` (anchor construction). New numbers below are arithmetic on those files plus
deterministic quadrature of the input law (scripts in `scratchpad/t347/`, nothing committed).

## 0. Short answer
No. I find no first-principles, model-free derivation of the +.57 pp. The sign is model-free (two Jensen-type
arguments and a three-body geometric floor, Section 3), the magnitude is not: every mechanism I can write down carries a
model quantity (the cross-gap slope/curvature of the pairwise error tail, or the model's local-linearity scale at a
triple tie). The observed constancy across retrains is a **cancellation of two opposite-sign, error-scaling parts**
(corner vs outside, Section 4), not the absence of model dependence, and it is **convention-specific** (window
[0, .5]^2 only; the same product term without the tmax cut is +.33 pp and anti-correlated with error, Section 2).
A constant that passes on DEV3 would show recipe-level stability of the retrain ensemble, not model-freeness, and
should not count as deductive under T347's own rule.

## 1. Setup (notation used below)
Per ordered top-3 triple, nb region (v2 > 0), coordinates u = g12 = v1 - v2, t = g23 = v2 - v3. Window
W = [0, tmax]^2, tmax = .5. The exact input law of (u, t) over nb inputs (v2 integrated out):

    rho(u, t) = int_0^inf 720 phi(v+u) phi(v) phi(v-t) Phi(v-t)^7 dv            [model-free]

c(u, t) = 1[model outputs p2 at (v2* + u, v2*, v2* - t, line-23 closure at t)], at the region anchor
(v2*, u*, t*) = (1.0137, .5340, .3476) (cont_d.exact_points: conditional means given v2 in nb; b region
(-.137, .851, .130), where the product does not apply because g12 is blind).
Rules on W: P1(u,t) = c(u*, t) c(u, t*); J1(u,t) = c(u, t). Error sets E_P = A* x [0,.5] u [0,.5] x B* with
A* = {t: c(u*,t)=0}, B* = {u: c(u,t*)=0} (two strips); E_J = {c = 0}. The product term is

    N = sum_{720 triples} int_W rho(u,t) [1_{E_J}(u,t) - 1_{E_P}(u,t)] du dt = +.574 pp   (B3, 10 models)

with gross over = P(E_J \ E_P) = 1.43 pp, under = P(E_P \ E_J) = .86 pp.

**Input-law constants [measured, quadrature; they reproduce B3's corner masses .2/.8/3/10.5 %]:**
- rho0 = rho(0,0) = 3.370 per unit^2; window mass int_W rho = .4173 (41.7 % of all inputs; 42.2 % of nb).
- corner masses rho0-approx vs exact: delta .025: .211 vs .204 %; .05: .843 vs .760 %; .1: 3.37 vs 2.91 %; .2: 13.5 vs 10.2 %.
- 23-wall density rho(u,0): int_0^.5 = 1.324, int_.5^inf = 1.140 (total 2.464); rho(u,0)/rho0 = .91/.83/.74/.58 at u = .1/.2/.3/.5.
- 12-wall density rho(0,t): int_0^.5 = 1.094, int_.5^inf = .437 (total 1.531); rho(0,t)/rho0 = .84/.70/.57/.37 at t = .1/.2/.3/.5.
- wall-weighted means: E[u | t=0, u<.5] = .228 (anchor u* = .534 is outside the window); E[u | t=0, all u] = .580;
  E[u | t=0, u>=.5] = .989. E[t | u=0, t<.5] = .209 (t* = .348); E[t | u=0, all t] = .382; E[t | u=0, t>=.5] = .815.
- first-order anchor weights int rho(u,0)(u - u*) du: window -.405, far (u >= .5) +.519, all +.115.
  int rho(0,t)(t - t*) dt: window -.154, far +.204, all +.051.
- L-band and arm masses: delta .1: band 19.4 %, arms (band minus corner) 16.5 %; delta .05: 10.7 % / 9.9 %.

## 2. What the existing data say when re-read (new arithmetic, all [measured])
From `results.json` (80 DEV2, 2e5 paired inputs per model, V1 convention; product term = V1 - off_product):
- product term: mean **+.568**, sd **.275**, min -.04, max +1.43; corr with 1-p **+.072**, OLS slope **+.006 +- .010 pp per pp**
  of error, over an error range 4.0-18.6 pp (mean 8.0, sd 3.2). So B3's "model-constant" replicates at n = 80 in the
  weak sense: uncorrelated with the model's error, slope below 1/3 of the gross over-mass slope (.077). It is NOT constant:
  the per-model sd (.275) is half the mean, so a constant predicts any single model only to +-.3 pp.
- same term in the path convention (no tmax cut, blind walked; path3 - path4): mean **+.332**, sd .374, **corr -.39**.
  The constancy is therefore a property of the window W, not of the product rule: without the cut the far region
  (u >= .5 or t >= .5) enters, where the single slice over-predicts error (Section 3, M2), by about -.24 pp, and that
  part scales with the model's error.
- for comparison, tmax term +.182 (corr +.85: the far soft tail scales with error), blind +.069 (corr -.01),
  transverse -.269 (sd .27). So the whole walk-approximation bias that strict needs removed (+.6-.76 pp) is not constant.
From `corner.json` (10 models), splitting the net by region (pp, slope = OLS vs 1-p in pp/pp):

| region (own gaps) | mass | net mean | sd | corr(1-p) | slope |
|---|---|---|---|---|---|
| corner both < .05 | .8 % | +.081 | .045 | **-.73** | -.011 |
| corner both < .1 | 3.0 % | +.283 | .116 | -.37 | **-.015** |
| arms: min < .1, max >= .1 | 16.5 % | +.240 | .198 | +.02 | +.002 |
| outside: both >= .1 | 21.7 % | +.050 | .065 | **+.59** | **+.0135** |
| total | 41.7 % | +.574 | .306 | .00 | +.0001 |

Per corner input (delta .025 / .05 / .1): over fraction (J wrong, P right) .22 / .24 / .19, falling with error
(corr -.82 / -.49 / +.12); under fraction (P wrong, J right) .17 / .14 / .09, **rising** with error (corr +.72 / +.73 / +.82).
Reading: in the corner the product's strips widen with the model's error and paint over inputs the joint already gets
wrong or right (under grows, over shrinks); outside, worse models have more joint error that the slices miss. The
two slopes (-.015 and +.0135 pp/pp) cancel. Nothing in the geometry forces them to be equal (Section 4).

## 3. Candidate mechanisms, with the integrals
Parametrise the near-wall error: along the 23-wall E_J contains {0 < t < a(u)} (tail width a, a function of the other
gap u; a > 0 means the model's 2-vs-3 boundary is displaced into this triple's window, a < 0 means it lies in the mirror
triple (p1, p3, p2)); along the 12-wall {0 < u < b(t)}. a* = a(u*), b* = b(t*).

**M1 (three-body / triple-tie term, zeroth order in the tail widths).** Near the triple point v1 = v2 = v3, in the
plane v1+v2+v3 = const, the true label is a 6-sector alternating pattern (p2 correct in two antipodal 60-degree
wedges, likewise p3, p1). The six ordered triples' corners [0,delta]^2 carry equal mass rho0 delta^2 (exchangeability
of the iid law). A ReLU net is affine inside each gate cell, and the argmax of three affine logits partitions a cell
into at most three convex sectors; a convex sector contains at most 60 degrees of any one label's truth, so for delta
below the model's local-linearity scale r at the triple point

    mean over the 6 windows of E_J(corner delta) >= 1/2        [inferred: model-free bound, uniform-measure version]

while the product's strips give, in the same inner limit, E_P -> (fraction of the 6 windows that receive a positive
tail) in [1/2, 1]. The net corner term is the competition between a saturating joint error and strips of width ~a*:

    N_corner ~ rho0 [ c1 r^2 - c2 r (a* + b*) + a* b* ]       [speculative form; c1 ~ pi/8 for a half-wrong quarter disc]

Inverting the leading term alone: N_corner(.1) = .283 pp gives r ~ .046; the whole +.574 gives r ~ .066, both in the
range of the gate-cell size near ties (H-E15: ~7 cells in the first .25 of gap). The first-order corner correction is
negative in the tail widths, which is the measured corner slope (-.011 to -.015 pp/pp) and the rising under-fraction.

**M2 (slice non-representativeness along the arms and in the far region; first and second order in a).**

    N_23 = sum_triples int_0^.5 rho(u,0) [a(u) - a(u*)] du  (+ the analogous N_12 with rho(0,t), b)
         = sum { a'(u*) [-.405] + (1/2) a''(u*) int_0^.5 rho(u,0)(u-u*)^2 du + ... }

The anchor u* = .534 (mean of g12 over all nb) sits outside W; the wall-weighted mean inside W is .228. With a
decreasing tail width (the 2-vs-3 decision gets harder as v1 approaches: a' < 0) the first-order window term is
positive. The same slice is also used for u >= .5 (s23 alone), where int rho(u,0)(u-u*) = +.519, so over the whole wall
the first-order weight is only +.115 (u* is already within .046 of the first-order-optimal .580; for the 12-wall .348
vs .382). Consistency check [inferred]: the window arms (+.24 pp at delta .1) imply a' ~ -.0059 per unit u; the far
region's deficit in the path convention (~ -.24 pp) implies a' ~ -.0046. The two independent readings agree, i.e. the
23-wall tail narrows by ~.003 across the window (a ~10 % effect on a ~.03 tail). Because the all-wall first-order term
nearly vanishes, the part of N that survives in the no-cut convention (+.33) is dominated by **convexity** of a(u)
(Jensen: a convex decreasing tail gives positive contributions both inside W and far), proportional to a''. Model
quantity: the cross-gap slope and curvature of the pairwise tail width.

**M3 (one-sidedness across mirror triples).** For a pair wall shared by (p1,p2,p3) and (p1,p3,p2), the product's
total error from that wall is rho_bar |a(u*)| while the truth is int rho |a(u)| du: with the anchor at the wall-weighted
mean and a affine, N_pair = rho_bar (E|a(u)| - |E a(u)|) >= 0, strictly positive only where the model's boundary crosses
the true wall inside the window. Sign model-free; size depends on a0/(a1 sigma_u). It is a sub-case of M2 with
sign-changing a.

**M4 (error islands off the walls).** An island of area eps^2 is caught by a line with probability ~eps/.5 and then
painted as a strip of mass ~eps x .5 (under), or missed (over, eps^2): expected under ~ eps^2 ~ expected over. The
line sampling of islands is unbiased to first order; islands feed the .275 pp model-to-model sd, not the mean.

## 4. Does the slope dependence cancel? No, and why
Write the model's error scale as w (tail widths a, b ~ w). E_P is exactly first order in w (strips, minus the a*b*
overlap); E_J is zeroth order (M1, scale r) plus first order with a u-dependent shape plus curvature. The difference's
w-slope vanishes only if the anchors are representative of the w-scaled shape, which the corner (saturated joint
error, growing strips: slope -.015) and the outside (slices miss growing far error: slope +.0135) each violate with
opposite signs. These are different regions and different mechanisms; their equality at n = 10 (and the n = 80 slope
.006 +- .010) is a measured fact with no identity behind it that I can find. The quantities the net depends on:
(i) r, the local-linearity scale at triple ties (sets M1); (ii) a'' and the sign-change statistics of a(u) (M2/M3);
(iii) the anchor offsets (-.405 / +.519; model-free). None of (i)-(ii) is available from the input law. If r and a''
are set by the training recipe (weight scale) rather than by how well a given retrain learned, a constant across
retrains is expected, but that is a property of the ensemble, exactly what T347 says does not count.

## 5. Correction formulas and predicted values vs +.574
- **C0, model-free constant.** Not derivable. The only closed form I can offer, N ~ (pi/8) rho0 r^2 + first-order
  anchor terms, predicts nothing without r; read backwards it predicts r ~ .05-.07 (testable on the models, not a
  derivation). Predicted value: undefined. [speculative]
- **C1, model-free anchor re-placement (zero cost).** For the strips to be unbiased at first order over the whole
  wall, slice at the rho(u,0)-weighted mean, u* = .580 (now .534), t* = .382 (now .348). Predicted change of the
  all-wall first-order term: 2.464 x .046 x a' ~ -.05 pp with a' ~ -.005, i.e. **no useful gain**: the current anchor is
  already near first-order optimal. Moving the anchor into the window (.228) would fix the window arms (~ -.25 pp) but
  break the u >= .5 population (+.519 weight, twice the window's wall density): net worse. [inferred]
- **C2, model-cheap, zero network cost: first-order tail shift from the additive margin.** The line-12 walk already
  gives the 2-vs-3 logit margin m23(u, t*) for all u. Under the cut-HDMR assumption m23(u,t) ~ m23(u*,t) + m23(u,t*) -
  m23(u*,t*), the tail at u is a(u) ~ a* - [m23(u,t*) - m23(u*,t*)] / d_t m23(u*, a*), and N_23 follows from the M2
  integral with the exact rho. Zero network cost, model-dependent (so deductive in the project's sense), but B1's ADD
  rule (the same assumption applied to the whole verdict) over-predicted errors by 2.5 pp; the restricted first-order
  version is untested. It cannot reach the corner term (M1: the lines pass .35-.53 away from the triple point).
  Predicted value: not computable here (needs the logits); expected to recover at most the arm/far part (~.25-.3 pp).
- **C3, model-cheap with network cost: a second cross-gap anchor for the first error piece only** (u2 = .228 for
  line 23 inside W, t2 = .209 for line 12; the first wrong piece near the wall, retired by the C1 screen). Captures M2's
  within-window first-order part (~.24-.29 pp of the .57), not M1. Cost ~ +0.4-1.1k FPE (720 nb triples x 1-2 partial
  lines at ~.5-1.5 FPE/line). With T339's s = -1 at 3,748 FPE already above the strict allowance (~3.45-3.55k at RMS .455),
  strict stays out of reach by arithmetic even if the bias were fully removed (F ~1.05 at zero added cost). [inferred]
Bottom line: no formula predicts .574 from first principles; the zero-cost deductive lever (anchor) is already used up;
the remaining levers are model-dependent, partial, and cost FPE.

## 6. Cheapest pre-registerable DEV3 test (go line)
Cost: budget.py-style pointwise pass, 2e5 exact-law inputs per model (~20 s/model), closure as it will be submitted
(T339 spread s = -1) **and** the mean-point closure, V1 convention; truths opened after the predictions are frozen.
Record per model: window net/over/under; corner E_J, E_P, net at delta .025/.05/.1 (also E_J itself, which B3 did not
save); arms and outside nets; the no-cut product term; 1-p. Frozen predictions (no DEV2 number enters any of them):
1. Sign: window net > 0 for >= 90 % of models (M1-M3). [structural]
2. Cancellation structure: corner-.05 net has corr(1-p) < -.3 and outside-.1 net has corr > +.3; total |slope| < .02 pp/pp.
3. Convention: the no-cut product term is .15-.35 pp below the window term and has corr(1-p) < -.2 (M2 far deficit).
4. Three-body floor: mean-over-triples E_J in the corner [0,.025]^2 lies in [.45, .70] for every model and is
   uncorrelated with 1-p (|corr| < .4). If E_J(.025) < .45 for most models, M1 is not operative at that scale.
5. The DRI's constant test: |mean_DEV3(window net) - .574| <= .1 pp (needs n >= 30 for SE .05 at sd .275).
**Go line for "derived":** a correction counts as derived only if it is a formula with no fitted number AND predicts
the per-model net on DEV3 with corr >= .5 and RMS <= .2 pp (a constant has RMS .275 by construction). Passing (5)
alone establishes recipe-level stability and does not authorise using .574 in a strict submission. If C2 is computed,
add: corr(C2 prediction, measured arms+outside net) >= .5 and |mean error| <= .1 pp.
Mark the result "negative (unverified)" for the derivation branch if only (1)-(3) and (5) pass.

## 7. Alternatives: why the constancy could be coincidence or selection
- Sample size: at n = 10, corr .001 has SE ~.33 (95 % interval about [-.6, .6]); the n = 80 slope .006 +- .010 is the
  real evidence of flatness, and it is flatness relative to the gross .077, not constancy (sd .275).
- Cancellation, not invariance: corner slope -.015 vs outside +.0135 pp/pp, two regions, two mechanisms (Section 4).
  A different tmax, anchor, or closure moves them separately: the same term without the cut is +.33 with corr -.39,
  and B3 itself warns the geometry may differ under T339's spread lows (untested).
- Reference dependence: the "+.574" is P1 - J1 at v2*, and J1 understates by -.34 (Addendum A); the product's part
  relative to the truth is tangled with the transverse term (-.27, sd .27), so the number is a bookkeeping slice.
- Ensemble effects: all 80 DEV2 models share one training recipe (two code families, fused/torch); a stable r or a''
  across retrains is a recipe property. Errors span 4-19 pp, so flatness is non-trivial, but it says nothing about a
  different recipe or width.
- Forking: B3's delta grid and the corner/L-band framing were chosen after T324 showed the product term; the constancy
  statement was pre-registered as a question ("near-constant or scales?") with no threshold, so "constant" is a reading,
  not a passed test.

## 8. Claims
1. [measured] On 80 DEV2 the V1-convention product term is +.568 pp, sd .275, corr(1-p) +.072, slope +.006 +- .010 pp/pp;
   in the no-cut convention +.332, sd .374, corr -.39.
2. [measured] Input-law constants: rho0 = 3.370, window mass 41.7 %, wall densities 1.324/1.140 (23) and 1.094/.437 (12),
   anchor offsets u* - E[u|wall, W] = .306 and t* - E[t|wall, W] = .139; first-order anchor weights -.405/+.519/+.115.
3. [measured] The net splits into a corner part (both gaps < .1: +.283, slope -.015 pp/pp) and an outside part (+.05,
   slope +.0135) whose slopes cancel; corner under-fraction rises with error (corr +.7-.8), over-fraction falls.
4. [inferred] Sign of the product term is model-free (M1 floor, M2/M3 Jensen); magnitude depends on the model's
   local-linearity scale at triple ties and on the cross-gap curvature of the pairwise tail width.
5. [inferred] The current anchors are within .046/.034 of the first-order-optimal all-wall slices; re-anchoring at
   zero cost cannot remove more than ~.05 pp.
6. [speculative] N_corner ~ (pi/8) rho0 r^2 - ... with r ~ .05-.07; E_J in the inner corner >= 1/2 averaged over the
   six orderings.
7. [inferred] No zero-cost deductive correction exists for the product term; model-cheap ones recover at most the
   arm/far part and cost FPE that strict's arithmetic does not have.

## 9. Recommendation to the DRI
Run Section 6 as a structural test (cheap, informative for the ledger), but do not expect it to yield a deductive
constant. Close the "derive the +.57 model-free" branch as negative (unverified) with this exhaustion list: tried in
thought: anchor placement (first-order optimal already), additive-margin tail shift (untested in the restricted form,
C2), second anchor (C3, costs FPE), three-body geometry (bound only). Not tried: any rule using the gate-cell sizes the
walks already compute as a proxy for r (speculative, would need its own pre-registration).
