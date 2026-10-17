# T138 fixed gate/top-three information and constructive retained margin

`check_gate_symbol_review.py` independently runs the original checkpoint
with exact binary-rational weights and dyadic inputs. Two inputs have the
same full rank order, top indices0/3/6 and values1,63/64,31/32, and all160
gate signs. Their unique network winners are3(correct) and6(wrong).
Exact global per-hidden-coordinate Lipschitz bounds satisfy every saved
gate, rank and output stability inequality with common input-box radius
8.28669939158e-10. The two lower-coordinate boxes share an open top-three
box, so this is not merely ambiguity at an isolated conditional value.
No numerical probability, practically useful error floor or estimator
accuracy claim follows from the tiny conservative radius.

Thus top-three values/indices plus complete rank and gate symbols do not
determine this network's correctness. This concerns reconstruction of the
network's correctness/output, not prediction of the true second index
(which the rank summary already contains). Under independent Gaussian
coordinates, any deterministic reconstruction of correctness from this
summary incurs positive error at least P(common top box) times the smaller
of the two lower-box probabilities. This is a formal positive bound only,
not a measured or useful-sized bound.

The reviewer derives the fixed-cell margin by reverse adjoints, independently
of the author's ten prescribed-gate basis passes. All coefficients exactly
match; every lower-input coefficient is nonzero. The segment crosses
margin3-minus6 at lambda=.6988823149957404. Independent direct exact
recurrence at the crossing retains all160 gates and gives equal logits3/6,
with every other output below by at least2.43219660145. Gate/rank cells are
convex, and output margins are affine within the fixed cell; endpoint and
crossing dominance therefore establish winner3 before the crossing and6
after it. First-index ties select3 exactly at the crossing.

One retained weighted lower-input scalar recovers correctness on this
segment. This constructive alternative remains open and prevents a broad
claim that gate-based or compact state summaries are impossible. No global
sufficiency, probability integration, parameter-budget or FLOP claim was
reviewed. Combined exact review used.0492CPU excluding checkpoint load;
no new search or LP was run. The author's64-anchor/640-LP search and
uncertified other candidates are not promoted to additional certificates.

## T141 conditional Gaussian slice

The independent check integrates exp(-||a+lambda*d||²/2) and lambda times
that density directly at70 digits, splitting at the exact crossing; it does
not use the author CDF formula. Correctness probabilities .724813275187512
and .429435506464070, and conditional means .488446562402731 and
.738552114127653, agree within9.1e-72 on frozen intervals[0,1] and[1/2,1].
Mean substitution gives biases +.275186725 and -.429435506 respectively.
Exact rational exponent-range and signed-moment inequalities independently
certify these opposite signs: each density ratio is below2, and the two
squared interval-length inequalities place each mean strictly on its
claimed side of the crossing. Probabilities lie strictly between0 and1.
This refutes an unrestricted universal bias-sign explanation, not E27's
actual omitted-input distribution or its empirical60 outcomes. No global
estimator accuracy or practical probability floor follows. Review .146CPU.

## T147 joint-input strip continuity

The source's first/middle gate-word chronology matches scalar forced
recurrences. Independently checking all880 coefficients on a noncommuting
rational fixture and all880 trained coefficients reproduces every saved
form. For an affine extension n·(x,y)+c·h+e*u, a crossing along the hidden
segment lies between the endpoint hyperplanes even if that gate word is
invalid at the endpoints. Standard Gaussian strip density is at most1,
and Euclidean normal norm exceeds its maximum coordinate. Thus the rational
coefficient ||c||1/max(|n_x|,|n_y|) is conservative. Final-gate boundaries
cover support/tie changes; adjoining nonconstant margins cover transitions
from tied pieces. A zero input normal with nonzero hidden coefficient must
remain unknown. Permanent support-identical ties do not move.

Exact proportional deduplication includes ALL input, hidden and target
coefficients; it preserves both strips and coefficient ratios. Actual rank
need1 uses the union probability2LU. At u=1, L>1/3 and U>1/27 follow from
integrating over[-1,1] and[1,2] and elementary e/pi bounds, so coefficient
(81/2)K does not rely on floating CDF evaluation.

Independent sums give45 unique hidden-dependent forms and
K=4291.128014579307. Displayed need1 coefficient16073.5954211 and exact
rational elementary coefficient173790.68459046 reproduce. Matched y-only
ablation separates joint-normal gain from exact hidden derivatives and
deduplication. The bound remains too loose for a practical fine-grid
certificate; no interpolation accuracy or estimator-cost claim follows.
Combined independent synthetic/trained coefficient review1.057CPU including
checkpoint import, with no trained input evaluation.

## T155 support-axis restriction and feasible gate words

Hidden motion h=s*e_axis changes the strip coefficient to |C_axis| divided
by the maximum input-normal component. The exact first/middle gate inequalities
in(s,x) restrict the entire frozen hidden interval and x in[-6,6]; discarding
only an EMPTY closed feasible set preserves every relevant crossing word.
Line/point sets must remain, regardless of zero area. Reduced(x,y,s,u) form
deduplication remains exact; multiplying by|Wi_axis| translates the bound
to the initial input coordinate. Both frozen prefix intervals are below u=1,
so the actual rank requirement is need1.

Independent all-pair boundary intersections, checked against every inequality,
reproduce all32 feasibility decisions without using polygon clipping. There
are3 feasible words on axis0 and5 on axis1. Exact directional coefficients,
deduplication and normalization reproduce elementary prefix coefficients
43.1957886368 and147.026534765, with floating CDF displays3.995102684 and
13.598226174. These are bounds only on the two frozen support intervals;
they do not certify the older interpolation into later mixed-support states,
population accuracy, or a practical integration/grid algorithm. Review.023CPU.

### T159 exact crossing-face restrictions and Gaussian damping

The source's equality elimination is valid without reversing inequalities: it substitutes the solved input coordinate into each closed inequality. Both input bounds are retained. Readout faces additionally require target-maximality on the closed final support; final-gate faces conservatively retain only first/middle/rank restrictions. Each proportional boundary survives if any original occurrence on either rank side has a nonempty closed face. Points and lines must survive. The endpoint-extension strip proof therefore still applies. The density factor uses the minimum offset over the entire frozen state interval, including zero if endpoints straddle it; `exp(-q)<=1/(1+q)` and replacing the Euclidean normal norm by its maximum component give a conservative rational allowance.

`check_face_strip_review.py` independently reconstructs all first/middle/final affine expressions from checkpoint fractions, then verifies all 88 occurrence/side constraint sets and every saved face vertex by all-pair boundary intersection enumeration, without importing the clipping implementation. It also checks complete source-occurrence coverage, exact proportional allowances, damping and sums. Axis0 has no feasible moving face and zero bound; this proves pair-value invariance on the frozen local prefix interval and truncated need1 domain, not on arbitrary states. Axis1 retains eight unique boundaries and fourteen feasible occurrences, including two degenerate faces; its undamped K is 0.178267919994959, damped K is 0.091071209489187, and elementary prefix coefficient is 1.610440671220323. Review CPU 0.268. No old-interpolation repair, population error or full-input truncation certificate is implied.

### T164 unbounded-input projection qualification

Fourier–Motzkin elimination combines each positive/negative eliminated-coordinate coefficient pair with nonnegative multipliers, retaining coefficient-zero inequalities and contradictions. Back-substitution and closed interval union are exact. All176 original forms are considered, only zero state derivative is skipped, s>=0 is explicitly included, and both rank sides remain; no earlier local word filter or input cutoff is reused.

`check_prefix_projection_review.py` verifies198 occurrence/side projections and3,296 nonnegative dual combinations, checks exhaustive pair generation, each exact projected endpoint/contradiction, and every lifted original-space witness. Fourteen faces are feasible, all on rank10 and first gate mask(0,1); their union is [0,infinity), so this test supplies no unbounded-input invariant sector. It does not prove actual correctness variation everywhere. The previous [-6,6] truncated-domain invariance is consistent with these witnesses requiring far larger fresh inputs. Review0.136CPU. An initial reviewer-only union assertion compared a Fraction to the None infinity sentinel; the initial script/failure are retained and the short-circuit order was repaired without changing any scientific data or preceding certificate checks.

A possible distinct remainder rescue follows from a shared exact first-gate inequality x>=kappa*s: outside that wedge, correctness can be compared to the s→infinity first-step-reset limit. For actual negative prefix and positive target, the Gaussian disagreement-envelope event has mass `(1/2) integral_0^infinity phi(v)[Phi(v/k)-1/2][Phi(v)^2-1/4]dv`, k=kappa*abs(Wi0). The rank-free wedge bound is atan(1/k)/(2pi); the rank factor even gives3/8 of that bound. This is only a proposed event upper bound, not evidence every feasible face changes correctness. An independently checked analytic enclosure is L-1/(16pi*k^3)<=P<=L, with L=1/(8sqrt(2)pi*k)+atan(1/sqrt(2))/(4sqrt(2)pi^2*k), from the linear/cubic normal-CDF bounds and integration by parts. A fresh claimed phase must verify all shared gate inequalities, reset-limit/tie semantics and numerical constant enclosures before applying it.
