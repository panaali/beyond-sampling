# T119 independent four-dimensional mass reference review

Read-only review of the pinned singular-endpoint mass implementation
and the different-reference code in `cone_mass4_review`. No shared
author source is edited, and the whole trained corpus is not rerun.

For covariance C(t)=I+t(C-I), Plackett differentiation gives
P=1/16+sum asin(rho_ij)/(8pi) plus
integral_0^1 sum rho_ij*asin(pc_kl.ij(t))/sqrt(1-t²*rho_ij²)/(4pi²).
The independently implemented conditional correlation uses the correct
Schur numerator and variances. For row rays R, normals are rows of
inverse(R^T), as used. Thus the50/70-digit reference follows a different
path from selecting one singular endpoint; it is not a high-precision
reimplementation of that endpoint's null-vector branch.

If all rays share one coordinate orthant, a single diagonal reflection
maps R to abs(R) and preserves isotropic Gaussian mass. The positive
L1-simplex reference is valid there. Its already-reviewed integration
bound still excludes determinant/input roundoff. Completed examples
include nearest420, whose author call raises but positive mass is
3.3284420355e-8; nearest10506, whose positive reference is2.66355056e-10
versus the author's-2.01679328e-7; and mixed-orthant nearest495, whose
independent50/70-digit references agree at8.4608273508e-10 versus a
negative author output. These establish specific numerical failures.

An asin input of-1 minus about2e-16 is a narrow floating-domain defect.
Clamping it can prevent that exception but does not validate the rest
of the mass or repair negative results. Persistence of a negative
result at n160 suggests that the tested quadrature refinement is not
the whole problem; it does not prove that every quadrature variant
fails. Near-root subtraction g±sqrt(g²+det*q), determinant/cofactor
conditioning and near-zero null-vector sign assignment remain possible
causes, not a confirmed unique diagnosis.

The signed facet decomposition is mathematically valid up to boundaries
when the chosen apex and original rays have a common positive
cross-section. The facet-normal apex supplies such a cross-section:
perturb it by a sufficiently small positive sum of facet normals to
obtain a linear functional positive on every ray and the apex. The
usual oriented simplex identity then applies. Numerically, however,
signed subtraction of larger piece masses can lose a tiny result;
being mathematically valid does not confer good relative conditioning.
Positive fallback, stable endpoint arithmetic and conditioned high-
precision evaluation remain open repairs coordinated with T109.

No broad negative about cone integration, complete cost claim, or
confirmation of the author's overall estimator follows from this review.

## Finite orthogonal rescue of the mixed-sign failure

The frozen rational Givens rotation c=255/257,s=32/257 has c^2+s^2=1.
An independent Fraction application to the stored dyadic rays of
nearest495 verifies every rotated coordinate has a common sign across
all rays. Isotropic Gaussian mass is invariant under this rotation,
and a global coordinate reflection then gives a positive-orthant cone.
This establishes the geometric premise exactly for the stored rays;
the numerical rotated arrays and quadrature remain floating arithmetic.

The positive result8.46082735079679e-10 agrees with the independently
reviewed MP result8.460827350796833e-10. The integration-only bound is
not a total floating certificate. Fixing the four exposed failures
leaves all corpus values nonnegative, but does not validate every
remaining source-positive value or certify the whole estimator cost.
No further theoretical concern with this finite geometry-only rescue
was found. The original negatives/domain exception remain evidence.

## T123 full frozen quadratic geometry inventory

The independent Fraction checker now verifies the stored dyadic rays for
all4033 native common-orthant cones and all1826 chosen rational rotations.
Every rotation has exact c^2+s^2=1 and every transformed cone shares one
coordinate orthant. The488 uncovered cones remain outside this positive
representation; the finite rotation list failing to cover them is not an
impossibility proof. Geometry checks and T122 raw checks together cost
.2774CPU. Numerical corpus results remain a separate pending review.

## T123 covered numerical mass and acute-ray extension

Independent exact summation of stored binary floats confirms covered
positive mass.46898661743868986 and integration-only bound2.6243548516e-8.
The original covered mass is lower by5.8005977118e-7; even the previous
negative/exception-only repair remains lower by2.1869397547e-7. Thus
positive source outputs can carry accumulated numerical error too.
Three separately implemented MP50/70 references for source-positive
outliers agree to over45 decimal places and match positive quadrature
within4e-20. They validate these cases, not the entire positive corpus.

The488 unrepresented source values remain unverified. Combining them
with covered references gives diagnostic accuracy.4770131239947528;
its agreement with whole-model reference is not a full certificate.

A valid next representation is unit-L2 independent rays with pairwise
nonnegative inner products. For barycentric y, q=y^T G y obeys
q>=sum(y_i^2)>=1/4 and q<=1 by triangle inequality. The general cone
change x=r*R*y has Jacobian |det R|*r^3, so the positive Duffy Gaussian
formula and the existing q-range polynomial remainder argument apply
without common-coordinate-orthant geometry. Exact original dyadic dot
signs suffice before normalization. A negative pair dot rules out every
orthogonal common-orthant image but not more general positive integration,
other radial bounds or subdivision. Floating normalization/determinant
remain outside a real-arithmetic integration bound.

## T126 acute-ray source and saved-result verification

Independent exact dyadic pair products verify all488 remaining cones:
432 acute and56 with a strictly negative pair. Source unit-L2 normalization
and positive Gaussian/Duffy factor agree with the reviewed derivation.
Exact summation of stored floats gives additional mass.02737008814941666,
integration-only bound5.33635685313e-9 and original source discrepancy
-3.26410287568e-9. No numerical integration was repeated. Combined positive
references cover6291/6347 cones, with56 still independently unresolved.

For the remaining arbitrary independent rays, their exact Gram is positive
definite. On each of15 nonempty supports S, solve the equality-constrained
quadratic minimum y=G_S^-1*1/(1^T*G_S^-1*1); retain nonnegative y. The minimum
among feasible candidates is the simplex global minimum a>0, since the
true minimizer is interior to some support. Convexity bounds the maximum
by b=max(G_ii). This is a valid exact-rational geometric discriminator.

Writing alpha=(a+b)/2,delta=sqrt(a*b),rho=(sqrt(b)-sqrt(a))/(sqrt(b)+sqrt(a)),
the reciprocal-square Chebyshev magnitudes are c0=alpha/delta^3 and
|c_k|=2*(alpha+k*delta)*rho^k/delta^3. Summing the geometric/arithmetic tail
past N gives the proposed positive remainder. With N=n-2, the Duffy
polynomial degree in each coordinate is at most2N+2, within GLn exactness.
The quadrature and exact simplex integral each contribute at most one
supremum tail times volume1/6, yielding |detR|*tail/(6*pi^2).
This covers the real-arithmetic formula; near-rho1 cancellation, floating
normalization and a possibly useless fixed-order bound need separate
handling. No general56-cone computation was run by this reviewer.

## T129 all remaining exact radial certificates and full captured reference

Independent check_radial_certificate_review.py verifies each of56 KKT
certificates directly on the original dyadic rays: simplex feasibility,
exact objective a, dual slack Gy-a>=0, complementarity and b=max diagonal.
These suffice for the global convex minimum without trusting the author's
15-face search. Independent permutation determinants match exactly.
Explicit summation of Chebyshev coefficient magnitudes through degree400
at80 digits matches saved50-digit tail values within1.37e-49 relative.
All observed rho<1/2, so the omitted coefficient-sum remainder is far
below that comparison tolerance; this checks formula evaluation, not
outward floating certification.

The three saved reference sets are disjoint and cover5859+432+56=6347
captured error cones exactly once. Exact summation of their stored binary
float values gives accuracy.4770131259941193, with summed real-arithmetic
integration-only bound3.157995918627031e-8. The author's two-stage float
subtraction rounds to.4770131259941194; this one-ulp difference is benign.
The existing independent whole-model reference is.477013125994119.
No captured residual remains unrepresented. Exact original geometry,
float-normalized transformed rays, determinants and floating quadrature
error still require separate end-to-end treatment for a total certificate.
The expensive reference calculation is not a competitive estimator or
solved challenge claim.

## T130 preliminary cheap center-bound/count review

For c=sum rays and m=min_i(r_i dot c)>0, every barycentric point has
norm squared at least m^2/||c||^2 by Cauchy-Schwarz; convexity gives upper
bound max_i||r_i||^2. The Gaussian mass interval is
|detR|/(12*pi^2)*[b^-2,a^-2], so midpoint plus half-width is valid in real
arithmetic. The explicit24-product determinant uses72 multiplies and23
signed sums=95 arithmetic operations. The Duffy setup18*n^3+3*n and
positive kernel39*n^3+1 match their named arithmetic conventions.

Review identified ordinary uninstrumented recomputation of12*pi^2,
6*pi^2 and2*pi^2 in the source. Hoisting/declaring these model-free constants
or separately charging those operations is needed for an as-implemented
count; the author was notified before a full cost conclusion. Legendre
rule generation remains separately unknown. Floating determinant/radial
interval error and finite-order unresolved cases cannot be promoted to
an exact bound or family-wide failure. Final corpus/count review pending.

## T130 final count and moment identity audit

`check_positive_budget_review.py` independently sums every stored row and
checks kernel/selection formulas. After charging each executed pi-square
constant expression, per-cone tolerance1e-7 returns6347 cones using3143
quadratures and3204 intervals:89,513,307 named operations, mass error
+8.864729199e-6, summed bound.000119691583. Aggregate allocation1e-7/6347
returns6315 cones (5461 quadratures,854 intervals), retaining32 finite-order
unresolved cases:364,952,602 named operations, covered mass error
+5.27704927e-10 and bound1.59898999e-8. Counts include cold Duffy plans;
model-free declared reusable plans may be free under leaderboard rule4.
Legendre generation remains uncounted and floating intervals uncertified.
These are finite implementation outcomes, not a positive-integration
family impossibility or full portable cost certificate.

`check_dirichlet_moments_review.py` independently enumerates all256 fourth
index products and checks the55 Gram-quadratic coefficients exactly. For
uniform simplex y and q=y^T G y, write s=sumG,t=trG,r=G1,d=diagG. Then
E q=(s+t)/20 and
E q^2=(s^2+2st+4||r||^2+t^2+2tr(G^2)+8d dot r+6||d||^2)/840.
For q>=a>0, Jensen and Taylor with f''(q)<=6/a^4 imply
mu^-2 <= E(q^-2) <= mu^-2+3Var(q)/a^4.
Multiply by |detR|/(12*pi^2) for cone probability. Rejecting negative
computed variance is necessary but insufficient for floating certification:
a positive variance can underestimate after cancellation. The proposed
moment shortcut requires separate frozen implementation/results review.

## T131 unchanged-ray H8/T4 analytic chambers

`check_leaf_mass_review.py` independently verifies1762 captured ray
certificates:1643 exact nonnegative-Gram bounds and119 exact KKT bounds.
All determinants are recomputed by24-term Fraction expansion; all stored
KKT simplex/slack conditions pass. An independent60-digit tail formula
matches every GL20/32 bound. At descending rank r, a fixed permutation's
coordinate is positive with probability sum_{k=r}^4 binom(4,k)/16.
Dividing by24 and joining the two top-gap halves gives analytic chamber
masses5/128 and5/384 for parent pairs0/1 and20/21.

On precisely the same864 and898 ray simplices, existing source mass
exceeds truth by1.1257156447e-5 and1.5765814581e-5, with99 and45 negative
pieces. Positive GL32 gives errors0 and-1.73e-18, with integration-only
bounds3.3453e-14 and1.0786e-14. Thus the source mass primitive contributes
to the error in these partitions; attributing these discrepancies solely
to geometry is contradicted. This does not apportion the full-model mass
excess or prove absence of compensating geometric overlap. Source import
provenance was explicitly pinned by the author after review. Review cost
14.906CPU, no numerical integration rerun. A bounded whole-corpus replay
and independent geometry controls remain separate steps.

## T135 moment shortcut implementation

The source computes s=||sum rays||^2, row sums r_i=ray_i dot(sum rays),
and the six extra off-diagonal Gram entries. Its moment formula matches
the exact identity above. Intersecting the radial interval with the
Jensen/Taylor interval is valid in real arithmetic. Nonpositive variance
or reversed intersection rejects only the shortcut; unchanged positive
quadrature remains available. Positive cancellation error still prevents
a floating certificate; the author's exact16-ray controls expose4.38%
relative error in a tiny variance and correctly retain that limitation.

`check_moment_budget_review.py` independently verifies every saved moment
intersection/selection, call-ledger aggregate and exact sum of binary64
reference differences. Per-cone allocation uses1576 moment shortcuts,
86,395,006 named operations, mass error+8.795528123e-6 and bound.000100006242.
Aggregate allocation uses1599 shortcuts,362,966,837 named operations,
covered mass error-2.062409388e-9 and bound1.564812848e-8; the same32 cones
remain unresolved. Relative named count reductions are3.4836% and.5441%.
This is a finite cheap-moment rescue, not exhaustion of moment expansions,
adaptive integration, or other positive representations. No full floating
accuracy or portable operation certificate follows. Review CPU.1973.

## T131 full unchanged-ray gate-mass replay

`check_full_leaf_mass_review.py` independently verifies all eight batch
arrays form a disjoint exhaustive43471-index partition and recomputes
all48 analytic fixed-permutation/sign chamber unions. Stored sums,
negative counts, quadrature bounds, reused controls and CPU totals match.
Exact Fraction sums of the stored binary64 values give existing source
mass1.0008880231700568 and positive mass.9999999999999994, with summed
integration-only bound8.5571005563e-11. All chamber discrepancies fit their
reported bounds;2368 source pieces are negative. The replay used104.109653
recordedCPU, reusing1762 previously checked selected controls whose earlier
cost remains separate. This reviewer did not redo all43471 integrations or
exact certificates:1762 were independently exact-audited above, and source
logic/raw aggregation cover the full phase. Review CPU.3846; initial NumPy
integer JSON serialization failure is preserved and repaired only in output
conversion.

Thus the entire reported gate-total excess is reproduced and removed on
unchanged geometry by replacing mass evaluation. This is strong numerical
attribution, not a proof of geometric partition correctness (compensating
overlap remains logically possible) or a corrected network-accuracy result.
The error-region capture/replay is a separately frozen next experiment;
complete floating geometry, integration roundoff and full operation counts
remain qualified. Do not normalize an incorrect gate total and infer that
this fixes the error-region calculation.

## T136 independent section geometry

A cone section a dot x=1 lies at height1/||a|| from the origin. The
four-dimensional determinant pyramid formula therefore gives its simplex
three-volume ||a||*|detR|/[6*prod(a dot r_i)]. QR coordinates in a-perp
have the same volume. Centering/right-SVD rotation and dividing by all
three positive singular values changes volume by1/prod(s); multiplying
back is correct, without rank truncation. Conditioning cannot recover
geometry already rounded away, as the preserved rotated-thin controls
show. The controls are numerical screens, not a certificate.

`check_leaf_geometry_review.py` independently uses coordinate elimination
instead of QR/SVD: remove the coordinate with largest |a_i| and multiply
three-coordinate hull volume by ||a||/|a_i|. Across337 captured leaves and
1762 source simplices, no hull failures occur; maximum source difference
is2.0123e-16 and old raw hull difference2.0817e-16. Every source simplex
vertex exactly equals an original leaf ray. Saved parent aggregates match,
with largest partition residual2.1094e-15. Review CPU.1598. This uses the
same Qhull backend with an independent coordinate map, not an independent
exact hull algorithm. Sum agreement alone does not prove absence of
compensating overlap and holes.

## T137 fixed-rule precision attribution source review

For coefficient covariance from B=(R^T)^-1, C=correlation(BB^T), the path
R_t=I+t(C-I) gives the Plackett derivative over six pairs. Integrating the
conditional1/4 term yields1/16+sum asin(C_ij)/(8*pi); the remaining term
is the integral of C_ij*asin(partial_remaining_pair)/sqrt(1-t^2*C_ij^2),
scaled by1/(4*pi^2). The inverse-matrix partial-correlation sign is correct.
Holding rounded C and rounded rules, then exact-binary-ray C, then ideal
GL48 with unchanged cubic endpoint mapping separates the proposed sources
of error. Cholesky/domain guards and exact binary lifting are appropriate.
80/100-digit agreement is stability evidence, not certified MP rounding.
An inadmissible rounded-C arm must remain a separate failure, not be
silently clipped or attributed to finite quadrature. Numerical fixture
outcomes are pending; no new rule/order/geometry tuning was reviewed.

The next T131 error capture delegates unchanged sequential competitor
clipping, which partitions the error union by the first failed margin.
All-off first-index controls pass; recorded zero coefficient-identical
active pairs avoid the known positive-volume output-tie defect for this
frozen capture. Per-parent/leaf indexing and pinned hashes are consistent.
A corrected error mass remains conditional on captured floating geometry,
and must not be inferred by normalizing the previously incorrect gate total.

## T131 captured output-error replay

`check_error_mass_review.py` independently audits all35098 saved error
simplices exactly once across eight batches, nonnegative new masses,
per-cone target bounds and parent-pair error<=gate mass. Exact Fraction
sums of stored binary64 values reproduce original accuracy.9750473359398664
and yield same-geometry positive accuracy.9761608075466027. The original
error mass is overstated by.0011134716067363745;2984 original pieces are
negative. Total conditional integration bound is1.04221660744e-10, with no
finite-rule target misses. Capture/replay costs14.9911/81.3911CPU; this is
validation, not a cheap-estimator claim. Review cost.2893CPU, no integration
rerun.

The prior final3 raw record reports NumPy4M accuracy.97606325; its README
reports rounded independent Torch10M accuracy.97617. The new value differs
by+1.28 and-.19 nominal binomial standard errors, respectively. The latter
has reporting uncertainty±.000005; both are sampling estimates, not exact
truth or certificates. Comparisons are descriptive and do not establish a
multi-seed/noisy ladder pass. Direct recurrence semantics on selected cone
interiors is being checked independently in T139. Floating captured
geometry/roundoff remain outside the integration-only bound.

T137's initial assumed thin48/cubic branch failed a preregistered metadata
guard before precision evaluation. The actual fixture672 uses wide12 and
quadratic mapping (rmax=.9418468705, determinant1.74464624e-9). The failed
assumption was preserved, then corrected in a separately frozen protocol.
The Plackett formula review above applies unchanged to the actual branch;
no unrun48-point result should be inferred.

## T137 corrected fixed-rule result

`check_plackett_precision_review.py` independently reconstructs covariance
by inverting the ray Gram matrix, then evaluates conditional correlations
through explicit2x2 Schur complements rather than the author's4x4 inverse
kernel. At100digits all three fixed wide12/quadratic arms match within
2.33e-99 and remain negative, approximately-4.80377125e-7. Their positive
reference discrepancy is4.80391856e-7, whereas arithmetic/correlation/node
reconstruction changes the original by only1.34e-17,2.84e-17,3.49e-17.
This fixture's failure is finite-rule approximation, not repaired by high
precision or ideal nodes at the same order. No claim about all source
failures, all precision regimes or an interval certificate follows. Review
CPU.0418; original branch-assumption failure remains preserved.

## T139 captured-error witness semantics

`check_error_semantics_review.py` independently checks saved labels on all
35098 floating ray-sum points, the frozen smallest-margin64 selection,
and reexecutes those64 exact witnesses with integer-scaled dyadic recurrence
rather than Fraction recurrence. Every exact input/output/gate sign matches;
all targets agree with parent rank and all predictions are wrong, with no
zero gates or input ties. Margins range-.0020631204 to-.0004318818854.
These are finite interior-point checks, not whole-cone classification,
partition completeness or a mass certificate.

The first reviewer sort repeatedly decompressed the NPZ margin field and
was interrupted after26CPU observed, exceeding the intended20CPU phase
cap. This failure/accounting is preserved. Materializing that field once
fixes the issue; successful review costs.0572CPU excluding checkpoint load.
An explicit20CPU hard limit is now in the script. No estimator changed.

## T140 uniform doubled-rule discriminator

`check_rule_doubling_review.py` independently audits all35098 arrays and
recomputes the original-source correlation branch with batched arithmetic;
all flags agree and the nearest rmax is5.49e-6 from threshold. Uniform
12->24 and48->96 rules give accuracy.9761652290904052 versus positive
reference.9761608075466027, signed error-mass difference-4.42154380245e-6.
Per-cone absolute error sum falls from.002215623606 to.00005554513503,
but735 negative terms remain and largest cone error is1.744876697e-7.
The73 determinant-cutoff cones carry reference mass2.18853e-13. Aggregate
agreement must not conceal remaining cancellation and lack of positivity.

Named mass count404,438,808 exactly equals2*204,957,048-156*35098;
inherited hypothetical full partial count is440,089,662. This is an
inherited nominal subset, not audited complete cost or competitiveness.
Recorded replay CPU97.6691; reviewer.5874. One frozen checkpoint and rule
pair do not establish a noisy-model ladder pass or general source fix.
A separate fixed recipe108 replay and full-cost audit remain appropriate.

## T142 residual endpoint-tail diagnostic

The reviewer independently integrates the absolute correlation derivative
to reproduce the residual Plackett tail bound at all four frozen cuts.
For the original rounded last node, B=3.48783409322e-5 and the constant-tail
replacement bound D=5.02427322857e-5 exceed the known same-rule error
4.80391855955e-7. The prescribed test is inconclusive. The comparison uses
exact-ray correlations with the ORIGINAL rounded rule, ensuring replacement
above that last node leaves the rule unchanged. The analytic base term is
separate; the bound applies to the residual integrand only. Smaller-tail
bounds are true contribution bounds, not quadrature-error shares.

Independent eigenvalue trace/product checks support minimum eigenvalue
4.02795850443e-8 and condition number8.57667654e7. Individual correlation
thresholds do not reveal this conditioning. No endpoint-cause conclusion or
full floating certificate follows. Reviewer .0777CPU; an initial nonexistent
mpmath.trace call was preserved and repaired by explicit diagonal summation.

## T145 fixed recipe pruning versus numerical mass error

The independent stored-array audit verifies all28000 indices exactly once,
all recorded source hashes, parent batches, and exact rational sums of stored
binary floats. Corrected recipe accuracy .976160677036335 differs from the
corrected exact-geometry value .9761608075466027 by -1.30510267696e-7;
the combined integration-only bound is2.13788310235e-10. The original
recipe numerical accuracy error is -.000906436516688. Thus on this fixed
checkpoint the numerical mass error dominated apparent recipe error.
This does not assert pointwise true-network membership of approximate cells.

Doubled-rule recipe accuracy .9761655721035335 has total difference
+4.76455693077e-6; per-cone absolute numerical discrepancy sums to
4.65761282012e-5 and503 terms remain negative. Exact stored-float arithmetic
checks pruning-plus-numerical decomposition without cancellation ambiguity.
The inherited mass count325162176=2*164765088-156*28000; removing capture
stub/control charges and restoring96 first-gate charges gives geometry
36398526 and original full partial count201163614, reproducing the archive.
Doubled full partial count361560702 is not a complete audited cost.

No full floating geometry certificate, held-out/noisy ladder pass or broad
approximation-family conclusion follows. The independent check costs.604CPU;
an initial final JSON NumPy-int serialization failure is preserved and fixed
by conversion only. Author replay146.222CPU and capture14.537CPU remain
separate. No numerical integration was repeated by the reviewer.

## T144 determinant primitive

The two-row complementary-minor expansion has30 multiplies and17 additions
or subtractions. Its signs match the24-term determinant expansion. Outward
interval operations enclose correctly-rounded binary64 results by one
nextafter in each direction; product extrema over four corners enclose
interval products. Nonfinite outputs reject. Underflow and singularity can
leave sign ambiguous rather than yield a false sign certificate.

Independent audit checks all6347 stored exact-value containments, nonzero
interval signs, point errors and counters. A67-ray subset including the
fixed16, evenly spaced cases and worst errors/width is recomputed with exact
24-term Leibniz sums, independently of author elimination. All agree.
Point47 improves3792, worsens1701, ties854: no universal accuracy ranking.
Interval work154 arithmetic/243 comparisons/340 finite checks/94 nextafter
is separately charged. This certifies stored binary determinants only,
not input geometry, masses or hardware cost. Reviewer.340CPU.

## T143 analytic inner integration and corrected ledger

For q(z)=a*z²+2*b*z+c, delta=a*c-b²>0, the angle substitution gives the
positive identity a/(2*delta^(3/2))*[(Delta-sinDelta)+sinDelta*(1+cosSigma)].
The source squared sine/cosine expression equals1+cosSigma. Small-angle
series through degree11 has separately weighted degree13 remainder.
Endpoint-zero and collinear-origin crossing domains reject; fallback remains
positive GL integration. Ordinary floating arithmetic is not certified.

For any uniform degreeN polynomial in quadratic q, Duffy substitution and
its Jacobian produce outer degrees at most2N+2. Taking N=n-2, n-point GL
in both outer coordinates is exact after exact inner integration. Choosing
positive GLn at some inner nodes preserves the same polynomial exactness.
Both functionals have weight1/6, so the existing Chebyshev-tail mass bound
transfers unchanged in real arithmetic; analytic series remainder is added
separately. This does not cover generic floating analytic evaluation error.

The independent audit verifies all frozen rows/orders/bounds/coverage and
stored values, plus corrected per-row ledger totals. Newly counted guards
cost M arithmetic and12M-2 comparisons; formerly omitted Boolean work adds
(6 or9)M-3 comparisons, for (18 or21)M-5 combined. Totals31,692,407 and
94,913,850 reduce the comparable named subset64.59% and73.99%. The strict
run still leaves the identical32 unresolved cones. Covered mass errors
8.8642074e-6 and5.2774643e-10, and tiny separate series bounds reproduce.
No trained numerical rerun was needed; audit8.68CPU. Node generation,
indexing/memory/runtime and floating certification remain qualified.

## T146 partial cost source review

The AST wrapper preserves native results, operand evaluation and short-circuit
reductions, with setup reported separately. Geometry replaces only mass
values by zero and checks every ordered ray against the captured archive;
geometry does not branch on the returned mass. This is a valid bounded
trace contract. The independent sampler count is16 initial operations,
3*144 recurrence,60 readout and4 decisions=512 under this convention.

A residual quadrature node has48 array operations, six*(4+13) scalar
operations and42 explicit complement-index comparisons=192. Primitive
wide24/thin96/cutoff counts4808/18632/184 and all saved site sums reproduce.
Across fixed branches the explicit total is510,972,352; nominal inherited
mass counter404,438,808 is different. There are2,659,858 opaque4x4 inverse
calls and35,098 determinant calls. These are unpriced, not free. This is
an explicit source-operation subset with separate memory/opaque inventories,
not complete implementation cost or hardware cycles. Geometry totals await
completed trace review; no trained integration rerun was requested.

## T146 completed geometry aggregation

Independent composition of every saved site ledger checks35098 ordered-ray
callbacks,8455 cells, nominal geometry35650854 and explicit geometry42396504.
Eight diagnostic calls duplicate only seven scalar final-return subtractions
within the counted numeric work; these are removed. Cold halves and rule
setup are composed once. Total visible explicit work553400768, plus opaque
2659906 inverse,35098 determinant,275897 matrix-rank and4 leggauss calls,
reproduces exactly. Diagnostic repeated setup is not eight cold estimators.
The single-case conditional break-even budget609416949166.6 follows the
stated sampling variance and integration-only uncertainty. It is algebraic
headroom, not a complete cost or sampling-win conclusion. Reviewer.006CPU.

## T151 held-out error-cone point semantics

The independently integer-scaled recurrence checker is applied unchanged
apart from held-out paths/checkpoint and output name. All30726 saved floating
ray-sum labels and the frozen64 smallest-margin selection agree. All64 exact
inputs/logits/gates match independent integer arithmetic; margins range
-.00244276239988 to-1.16869915524e-5, with no zero gates or input ties.
Review.048CPU excluding imports/load. This remains a finite interior-point
check, not whole-cone validity, completeness or a mass certificate.

## T148 frozen held-out exact/recipe attribution

All16 completed batches are independently checked for exhaustive disjoint
indices, source hashes, parent assignment, exact stored-float sums and
counter identities. The seed1 checkpoint has30726 exact error simplices and
19746 recipe simplices. Original accuracies/counters reproduce their archive.
Corrected exact accuracy .9874600108260589 and corrected recipe
.9874543286232546 differ by -5.68220280441e-6, with combined integration-only
bound1.70946893939e-10 and no per-cone bound misses.

The frozen doubled rule gives exact-geometry accuracy .9874382753767020
and recipe accuracy .9874384324269055; the latter total difference from
corrected exact is -2.15783991534e-5. This held-out error is larger than
seed0 and must remain visible. There are1420/562 negative doubled terms,
with per-cone discrepancy sums7.4197562e-5/5.1463213e-5. Thus the fixed
rule is substantially improved but still neither positive nor certified.

Saved4M MC truth .98744275 has nominal binomial standard error5.56767e-5;
corrected exact is .310SE above it. MC is a diagnostic, not the reference
certificate. Nominal inherited full counts376677671 exact and262248590
recipe remain partial. Replay CPU156.611/103.637 is preserved, plus captures.
Independent audit.592CPU performs no integration or model-input search.
No tuning, full nine-case/noisy ladder, complete-cost or global challenge
claim follows from this held-out success.

## T150 installed inverse floating-expression phase

The saved11 controls show bitwise-equivalent baseline, GESV interception
and temporary own-process kernel-table interception. Dimensions4x4 and four
identity right-hand sides agree, including two calls for the batched fixture.
The pinned small-size path uses left-looking LU and two packed triangular
solves; observed solve shapes are4x2 twice per triangle, with no GEMM calls.
Source-derived floating expressions total56 for LU: strided DOT lengths1,1,2
cost11 including their final accumulation; three outer subtractions,32 GEMV
tail operations, four reciprocals and six scaling multiplications. Each
triangle costs16 diagonal multiplies plus48 updates, including unit-lower
multiplications; upper packing adds four reciprocals. Total188.

Independent event contraction agrees on all normal fixtures; conservative
singular/subnormal counts50/56/185 and batch376 stay below188 per inverse.
This is a floating-expression envelope for this pinned small-size dispatch,
with FMA counted as multiply plus add. It excludes absolute values, comparisons,
integer/index/control work, packing copies, allocation and runtime overhead;
it is neither an exact instruction count nor complete inverse/hardware cost.
Reviewer clock reported0 at its resolution, not zero work. No inverse or mass
was reexecuted in the saved-event audit.

## T153 recipe expression-cost continuation

The local AST adapter correctly recognizes ndarray methods on computed
receivers such as np.abs(mm).max; it preserves evaluation and uses the
original callable. All other transformations remain pinned T146 behavior,
and the geometry adapter asserts exactly one mode substitution. Independent
aggregation checks every saved site ledger and28000 ordered callbacks/6182
cells. Explicit geometry45131971 includes recipe-specific margin routines;
mass410812832 and cold setup31912 give total455976715.

A determinant-only source-identical census identifies54 cutoffs. The fixed
per-node legacy counter implies2110488 nodes and uniquely determines7949
wide/19997 thin calls; branch-ledger contraction matches exactly. Opaque
counts2138536 inverse,28000 determinant,257040 matrix-rank and4 leggauss
remain separate. Conditional single-case break-even524830209006.8 compares
the actual doubled candidate with positive EXACT reference, not with its
approximate-recipe reference. No complete-cost/sampling-win or candidate
selection claim follows. Saved aggregate reviewer.008CPU, no mass replay.

## T152 exact radial rescue of the32 finite-order omissions

The reviewer recomputes exact original-ray Gram matrices and24-term
Leibniz determinants for all32 cases. Thirteen acute bounds use
min(diag(G))/4; nineteen KKT certificates have simplex weights summing to1,
nonnegative weights, exact objective a, nonnegative gradient slack Gy-a,
and complementary slackness. Positive-definite Gram matrices make these
sufficient global minima. All selected orders and100-digit Chebyshev-tail
values reproduce; no geometry, rays or quadrature tolerances were changed.

The determinant-factor correction follows by splitting
D_old*Q-D_exact*I = D_exact*(Q-I)+(D_old-D_exact)*Q. Positivity and total
simplex weight1/6 imply Q(q^-2)<=1/(6a²), so the extra probability bound is
|abs(d_old)-abs(d_exact)|/(12*pi²*a²). Separate analytic series remainder is
added; all32 meet their fixed conditional target. General floating analytic,
rule and geometry errors remain excluded. Maximum added determinant-factor
bound is1.443e-17.

Independent merging confirms all6315 previous values and bounds bitwise.
All6347 now have returned values. Accuracy is .4770131254664665 versus
reference .4770131259941193, difference-5.27652813925e-10; summed conditional
bound1.60878969032e-8. The initial summary mistakenly called error-mass sum
.5229868745335334 accuracy. Reviewer flagged it; the author preserved the
bad report and corrected labels/complements without changing the archive.
Extra kernel named work2038604 is verified; exact rational operations/bitwidth
and MP selection costs remain separate rather than being declared free.
No complete portable cost or full floating certificate follows. Review.120CPU.

### T157 installed inverse numerical categories (independent saved audit)

The actual short IAMAX register semantics, including MAX's right-operand choice on unordered/equal inputs and COMISD/JE unordered acceptance, reproduce all 4,680 saved native cases with a separate scalar transcription. Lengths 4/3/2/1 require at most 32 absolute-value lanes and 44 MAX/comparison lanes total; packed duplicate/irrelevant lanes remain conservatively charged. The inspected four LU pivot guards add eight abs/eight comparisons, three SCAL guards at most six comparisons, and two fixed-beta triangular drivers at most four. Together with the earlier 188 floating expressions this is 290 units in this numerical subset. All eight saved inverse event controls use the expected IAMAX lengths and no GEMM; source parity assertions and pointer restoration were inspected, without another native run.

`check_inverse_numeric_review.py` checks source hashes, reproduces the native digest, and independently composes exact/recipe subtotals 1,324,773,508 and 1,076,152,155 (0.027 review CPU). These combine an upper allowance on selected numeric categories with still-unpriced work: they are neither a complete implementation upper bound nor a complete lower bound. Integer/control/pointer/memory, mask/shuffle and runtime work remain excluded; no hardware-cycle or matched-budget conclusion follows.

### T154 fixed wide-rule cubic transform

The source changes only the wide rule from 12-node power2 to 12-node power3; the thin48 power3 rule and all geometry remain fixed. `check_cubic_rule_review.py` verifies all 35,098 distinct indices, source hashes, and the 24,853 thin/cutoff outputs bitwise against the original saved outputs. Exact sums of stored binary floats reproduce accuracy 0.9761281369336525 versus positive reference 0.9761608075466027: error-mass difference +3.26706129501335e-5, per-cone L1 0.000231516910214, and 2,286 negative terms. The 10,245 wide, 24,780 noncutoff thin and 73 cutoff branches reproduce inherited mass count 204,957,048. Review CPU 0.431, no integration rerun. This improves the original fixed wide rule but remains a known-corpus experiment with substantial per-cone errors; the separately frozen doubled-rule candidate is unchanged. Same inherited count is not a complete portable-cost comparison.

### T158 actual installed determinant categories

Public source and pinned installed wrapper agree on GETRF followed by diagonal sign tests, log accumulation, exp and sign multiplication. The installed small GETRF dispatcher reaches the reviewed LU path. `check_native_determinant_review.py` independently checks all eighteen baseline/interposed fixtures, nineteen actual matrices, sixty-eight log calls and nineteen exp calls. It recomputes all seventeen finite exact determinants by rational elimination, distinct from the author's permutation sum, and checks dependency hashes and census composition (0.003 review CPU, no native rerun).

The wrapper permits five add/multiply, eight negations, four numeric comparisons, four log calls and one exp call. Combined with LU56 floating expressions and T157's98 LU abs/comparison categories, this is176 named numerical units. Eleven separately identified source pivot/info integer expressions give187 under that additional source convention. Exact/recipe determinant censuses give 6,177,248/4,928,000 numeric units, or6,563,326/5,236,000 including those selected source-integer expressions. Each libm call is one named category, not its internal FLOPs. Loop/index/control, allocation/copies and library internals remain unpriced; these subset upper allowances are neither complete estimator upper bounds nor complete lower bounds. The finite parity/reference controls do not certify general determinant roundoff.

### T156 installed SVD dispatch and observed dqds work

The interposition signatures preserve the installed ILP64 GESDD arguments; values-only JOBZ=N is recorded, and workspace queries are distinguished from executions. `check_rank_dispatch_review.py` verifies all114 baseline/interposed fixtures across19 observed Mx4 shapes, their known default/explicit-tolerance ranks and684 successful GESDD calls (342 queries/342 executions). Three dense tridiagonal fixtures independently match spectra `2+2cos(j*pi/(k+1))`; saved nested events have78 Q3 calls and69 ITER increments, maximum one per call. The repaired spy reads the actual integer ITER difference around Q3; it does not infer iterations from a Q2 workspace slot that early-return paths may leave unwritten. The original diagnostic failure remains preserved. Review time rounded to0 at process-clock resolution, not zero work. This is ABI/finite-parity evidence only; observed small iteration counts do not establish worst-case convergence or cost.

### T162 exact-preprocessed Bernstein Schur rescue

For `R=(1-t)I+tC`, generalized-diagonal principal3-minor coefficients in `u^3,u^2*t,u*t^2,t^3` and signed Schur numerator coefficients in `u^2*t,u*t^2,t^3` are algebraically correct. Exact Fraction coefficient construction then one binary64 rounding removes the earlier coefficient-construction cancellation, but signed polynomial evaluation may still cancel. `check_schur_bernstein_review.py` independently verifies320 identities through rational Gauss-Jordan inverse/determinant, rather than the author's explicit permutation minors. It audits hashes and the frozen256 saved indices: no failures/excursions, integrated candidate/source maximum9.78384e-16 and L1 5.57361e-15, three negative masses in both, and essentially unchanged positive-reference L1 3.40658e-7. Maximum individual partial-correlation difference7.14665e-8 remains visible. Review0.076CPU. Candidate/source ordered timing0.561/0.626CPU includes0.0696CPU exact preparation but is not a timing experiment or speedup proof; rational work is not free preprocessing. No full-corpus or universal floating-stability claim follows. Earlier Horner failures remain intact.

### T161 fixed64 H8T4 positive-inner pilot

The unchanged cheap center interval/finite Chebyshev selector plus analytic inner kernel returns29 midpoint intervals,34 center-selected quadratures and one exact KKT rescue. Initial synthetic diagonal nonconvergence was correctly retained as unresolved after fixing a harness assertion that wrongly required every control to resolve; no target/order relaxation was made. `check_positive_h8_pilot_review.py` checks frozen indices/source pins, first qualifying recorded orders, all returned target gates, and reconstructs the entire147,062-unit named floating ledger including cold rules and constant arithmetic. It independently recomputes the sole rescue at index31197 from original binary rays: exact Gram, KKT objective/slack/complementarity,24-term determinant and80-digit Chebyshev plus determinant-factor bounds, selecting order12. Error L1 2.74928e-12 and maximum4.79456e-13 lie within summed conditional bounds2.77228e-11. Review0.012CPU. Rational/MP setup and big-integer internals remain separately reported; general floating setup/kernel errors are not certified. This is a fixed64 discriminator, with no full-corpus accuracy or cost extrapolation.

### T160 preserved cancellation failure and T163 conditional retry scope

`check_schur_failure_review.py` independently confirms both saved near-singular failures through the signed equicorrelation identities, with r=rho*t: minor `(1-r)^2*(1+2r)`, numerator `sign_k*sign_l*r*(1-r)^2`, and partial correlation `sign_k*sign_l*r/(1+2r)`. At rho=1-2^-40 the exact numerator is -2.163439970e-23; evaluating the rounded coefficients exactly gives -2.080721909e-23, whereas actual Horner evaluation gives +1.110223025e-16. The raw correlation error is1.732665769e6. Both coefficient and evaluation effects are real; no trained panel ran under this failed representation. Review0.040CPU. The distinct T162 rescue does not erase the failure or prove a general Schur-family limitation.

The T163 source-only proof was checked against pinned Q2/Q3/Q4/Q5 branches. Q4 sets a negative TTYPE even before guarded early returns; two Q3 negative-failure decrements reach at most-23, so the third negative failure forces zero shift and the fourth attempt at latest exits or falls through to Q6 **if** the active q/e nonnegativity/positive-q-zero and stated IEEE/MIN invariant holds. Opposite-parity writes preserve retry inputs. A NaN trial forces zero earlier, not an extra fifth attempt. This does not establish that every finite initial SVD maintains the invariant. The separate INFO2 cleanup witness I0=I1=2,N0=4,Z3=-1 is syntactically nonprogressing: the body changes indices5 and above, leaves the marker and I1 unchanged, and jumps back. No reachable matrix or installed binary hang has been demonstrated. Thus universal runtime bounds remain unproved; the finite observed-call census is a distinct valid route.

### T165 expanded native observer control gate

The generated explicit ILP64 interfaces for QR/LQ/bidiagonal, reflector/norm and Q5/Q6 calls match their forwarded signatures; all arguments/returns pass unchanged and a bounded static ledger rejects overflow. Q3 records its pre-call block size/parity and actual ITER increment. The geometry adapter changes only the mass callback to zero, verifies every ordered saved ray, checks original legacy counts/cells and cross-checks one query/execution against each independently counted rank call; cold setup is separate. `check_rank_census_controls_review.py` independently verifies117 analytic outputs against prior baselines and four zero/rational exact/recipe geometry controls, including all saved rays and Q5+Q6=ITER increments (0.009CPU). The zero model legitimately has no rank calls. This supports the separately frozen finite trained census; it is not a universal convergence theorem or full operation budget. Nested parent/child kernels must not both be charged as complete work in later arithmetic composition.

### T113 recovered RECIPE112 arithmetic and semantic qualifications

`check_recipe112_recovery_review.py` independently uses65-digit Decimal arithmetic on all165 groups/10,720 saved noisy rows, verifying both historical mean-cost sampling variance and mean per-model-budget variance. Core H4/H5/H6/H8 panels each contain1,280 noisy models and worst historical ratios0.00257813/0.01282239/0.04224566/0.09898579. These are historical denominator conventions, not a completed native-cost comparison.

The raw counter panel already records output differences up to3.28215e-6. A direct independent import of the original Num type confirms missing equality methods make `Num(0)!=0` true. The source uses this condition in geometric edge tests, so instrumentation can change the algorithm rather than merely count it. The owner's frozen H8 diagnostic reproduces the saved difference and an equality-only patch restores the exact float result while changing count330161→326667; H4 retains approximately2e-16 float/wrapped roundoff and changes count137720→137268. No broad repaired counter transfer is established. All four analytic fixtures always select first index and therefore have accuracy1/3 by exchangeability; the preserved all-off/reference and zero/identical-readout failures are valid positive-volume tie counterexamples. They do not alone invalidate every generic noisy truth. Reviewer0.108CPU, no trained model/estimator rerun.

### T109 complete fixed H8T4 positive-inner corpus

`check_positive_h8_full_review.py` checks all35,098 saved indices, exact reuse of64 pilot rows once, source/shard hashes and both canonical/staged rule ledgers. All returned bounds meet the unchanged per-cone target. It independently recomputes all132 original-ray exact certificates (130KKT and2acute), including Gram matrices,24-term determinants, KKT complementarity and80-digit selected Chebyshev/determinant-factor bounds. No numerical cone integration is rerun.

There are14,768 center intervals,20,198 center quadratures and132 exact rescues. Accuracy is0.976160806417595 versus reference0.9761608075466027, exact stored-float difference-1.129007699e-9; summed conditional bound1.307416376e-8. The known floating/category ledger is142,873,286, with18,050 canonical cold-rule arithmetic units versus92,232 actually incurred across diagnostic stages. Rational maxima804/788 bits and MP setup remain separately charged. At4070.70 known units/cone, the original300-unit goal fails; this is a cost target failure for the fixed implementation, not a positive-integration-family impossibility. Full floating/geometry and complete library/runtime costs remain unproved. Reviewer1.168CPU.

### T165 finite native census aggregation

`check_rank_runtime_review.py` independently contracts all16 exact/recipe chunks, validates source pins and previous geometry/callback/cell/legacy totals, and composes the repeated cold setup only once per estimator. Exact/recipe rank executions are275,897/257,040, each including768 cold ranks; Q3 calls239,577/200,618; Q5/ITER increments201,773/165,964; no Q6 or recorded failures; maximum observed Q3 increment3. These data supersede no universal convergence theorem: they measure the two fixed successful evaluations. The initial geometry-only versus cold-inclusive aggregate assertion remains preserved. Reviewer0.042CPU, no trained rerun. Numeric prices for local kernel work and unobserved helper categories are still separate, and full parent plus descendant costs must not be double-counted.

### T108 simultaneous freezing counterexample and cumulative rescue

`check_simultaneous_freeze_review.py` independently reconstructs the exact recurrent synthetic model, all saved controls, normalized-reference signs using rational square-root enclosures, and a strict conic-interior misclassification. Its original rays straddle the nearest-gap initial-half split; this reviewer qualification prompted a fresh, preserved control. `check_half_family_freeze_review.py` verifies the new full-dimensional subcone (determinant -1/50), all initial-half inequalities, exact reference mask001 and source cut2/resolved0. At its minority ray the frozen margin169/500 exceeds each allowance1/5 and6/25, yet actual margin is-51/500. At strict interior coefficients(1,100,1), the frozen margin68977/2000 exceeds each allowance391/20 and9453/400, but actual margin is-4347/500. Thus individual allowances do not justify simultaneous freezing. This does not prove the default solver visits this exact subcone or refute any saved trained panel.

The safe replacement propagates nonnegative simultaneous error delta=ReLU(z)-g*z through absolute recurrent matrices and bounds each output contrast separately. `check_cumulative_review.py` reproduces768 allowances by independent reverse contrast propagation, checks exact original/frozen recurrence labels (106correct,354wrong,308unknown) and104 nonzero conic combinations. First-index ties require strict lower bounds against earlier rivals and weak bounds against later rivals; wrong certificates reverse these appropriately. The final-step all-ray extension is valid because frozen margin minus allowance is concave homogeneous. Wrong whole-cone certification requires the SAME rival on every ray. Nonempty generators/nonzero conic points are assumed; origin is excluded from strict-margin claims. Earlier downstream ReLUs do not admit this vertex argument without more partitioning or bounds. Earlier approximate-prefix modifications need incoming error tracking or their own classification-preserving proofs. No estimator adoption or full-estimator certificate is established. Cumulative review0.194CPU.

### T80 corrected scope and exact second-step means

`check_cumulant_scope_review.py` verifies25 saved logs,43 printed rows/42 unique variants, the duplicate, incomplete T10 populations and header-only endpoint table. It independently disproves the covariance-rank argument: h=(X+,(-X)+) has covariance eigenvalues1/2 and1/2-1/pi, both positive despite a one-dimensional latent input. The binomial even/odd law pair has identical moments0..4 but sign probabilities5/16 and11/16, preserved by common small uniform jitter; this limits moment-only summaries over arbitrary distributions, not all trained-weight methods.

For nonzero fresh Gaussian input coefficient, conditional activation probability and ReLU expectation are strictly positive. Finite-sample zeros cannot establish exact inactivity. For each original binary model coefficient the review independently computes a+=Wh ReLU(Wi), a-=Wh ReLU(-Wi) with Fractions and80-digit means [(hypot(a+,b)+a+)+(hypot(a-,b)+a-)]/(2sqrt(2pi)). All16 trained means and16 analytic controls agree; two naive-cancellation zeros are preserved. Relative Gaussian-closure mean RMS error is1.4941864023 on the fixed historical H16T4seed3 case. This verifies a substantial approximation error, while replacing the incorrect exact-inactivity rationale. The1208 named kernel units exclude complete setup/library work. Review core0.008CPU excludes imports. No global moment-family ceiling or final-accuracy claim follows.

### T100 recovered fibre T4 evidence and exact-zero clip repair

`check_fibre_t4_recovery_review.py` checks pinned saved files and independently contracts five clean rows with65-digit Decimal arithmetic. Historical individual variance-subtracted ratios are0.0404493/0.119888/0/0/0 and group0.0179575; the floors do not prove zero bias. Noise and original checkpoint execution provenance remain absent. Sampled patterns lower-bound true regions, so face/pattern counts upper-bound face/true-region counts and cannot establish the claimed complexity separation.

An exact polynomial rank calculation gives last-index positive/negative second-rank masses11/64 and5/64; each other index contributes7/64 and9/64. These establish the four frozen analytic truths1/4,1/4,5/16,3/16. Executing only the pinned clip function on a triangle independently reproduces loss of both polygons for a zero normal: normalization creates NaNs. Returning the polygon once for exact component-zero normal is correct for this homogeneous vacuous inequality. All four saved repaired controls pass1e-7; the three nonzero cases retain bitwise values. The repaired all-zero error9.97e-9 remains a finite cubature residual, not exactness. Nonzero normalization underflow remains separate. Reviewer0.004CPU; no estimator or trained rerun.

### T80 exact covariance reset and inherited closure continuation

`check_prefix_covariance_review.py` verifies the angular-sector factors E[R]/(2pi) and E[R²]/(2pi)=1/pi and the stable narrow-sector series. It checks all256 trained pair references/six analytic models, then independently integrates six selected positive angular products at60 digits rather than reusing the author's antiderivative or sector matrix summation; discrepancies from saved50-digit references are below4.43e-48. These remain ordinary-floating/high-precision numerical checks, not outward certificates.

All eight saved reference blocks and pooled moments, every method/error table and paired block standard error are independently reconstructed. The exact h2 reset arrays match the separately checked covariance output. At step4, relative mean error improves5.66330 to2.95512 and covariance error56.82295 to20.97410: substantial improvement but substantial remaining closure error. Block SEs are descriptive, not rigorous confidence bounds. No new inputs or propagation run was performed, and no final-accuracy claim follows. Reviewer0.160CPU.

### T167 unchanged held-out positive-inner pilot

`check_positive_holdout_pilot_review.py` checks source pins, all64 fixed seed1 indices/reference rows, first qualifying orders, and the complete saved known-unit ledger. Exact Fraction comparison confirms `(1e-7)/30726` is conservatively rounded relative to1/(10^7*30726); the preserved initial equality-preflight failure made no numerical change. All64 return (35center intervals,29quadratures), no exact rescues; absolute error sum1.092687777e-12, conditional bound2.104969292e-11 and532864 known units. Review0.032CPU, no integrations. This clears the separately frozen full-corpus phase only; model-facing collection of N requires charged geometry/capture/storage, and complete floating/library costs remain unresolved.

### T69 retained exact-prefix sector mixture

Conditional angular moments in `structured_prefix_mixture` correctly divide by sector probability: mean angular vector sinc(d)(cos(mid),sin(mid)), perpendicular second moment1/2-sin(2d)/(4d), with series d²/3-d⁴/15+2d⁶/315-d⁸/2835. Multiplication by radial means gives conditional mean sqrt(pi/2) and raw-second factor2. Components use original independent fresh-input covariance and unchanged Gaussian closure.

`check_sector_mixture_review.py` independently recombines all34 component states via weighted raw seconds minus pooled mean outer product, then reconstructs all error and eight-block paired-SE tables. There are68 Gaussian calls and minimum weight7.0981e-5. Step4 relative mean error0.706845 and covariance error3.192038 improve on collapsed exact-prefix2.95512/20.97410, but remain substantial. Tiny negative component eigenvalues, approximate within-component Gaussian support and floating angular geometry remain qualified; no PSD repair or new reference was introduced. This supports the fixed one-model rescue, not a final-accuracy method or all-width result. Review0.016CPU.

### T115 recovered margin-training panel arithmetic

`check_margin_training_recovery_review.py` independently contracts saved errors with70-digit Decimal arithmetic and reproduces the declared1000-resample law with a sequential bootstrap implementation. All15 panel rows and12 clean rows agree; the12 p20 rows have worst historical72-unit ratio0.1030003472 and upper bootstrap endpoint0.1145357715. Coverage is five seeds at k28 and only seed0 at k30. Three p100 rows permit error-stat reconstruction only; absent per-copy truth/counter arrays prevent independent ratio/CI validation. The bootstrap holds the saved mean cost fixed, so it does not establish cost-variation uncertainty or a common-budget native comparison. No independent truth/estimator run occurred. Review0.398CPU.

### T113 identified counter bypass envelope (source-only)

The repaired Num convention still bypasses comparison counting through `_step2.margin`'s `val(x)>0` and readout filters, and its final `1-val(err)` subtraction. Every A3f third coefficient is Num because it adds wrapped Wi; therefore each evaluated dot performs at least the third product and final addition as Num, and its ReLU adds one counted comparison. Each margin point thus costs at least3H counted operations. All-off points omit at mostH `any` comparisons. If any activation is positive, each of three output sums has a Num product and a 0+Num sum, including numerically zero weights; rival max and subtraction add two more. Active points cost at least3H+8 while omitting at most4H comparisons. Consequently `[1+max(1/3,4H/(3H+8))]*N_count+1` bounds the count after filling THESE identified bypasses (factor1.8/1.869565/1.923077/2 for H4/5/6/8). It is not a complete native/source cost ceiling: plain-zero expressions elsewhere, integer/control/memory operations and libm internals remain excluded. The sampler convention must be reconciled separately. This source argument requires no repeated full panel and does not silently change the preserved raw counter fields.

### T113 repaired original125 counter panel

`check_recipe_counter_repair_review.py` independently verifies all source/checkpoint hashes, exact five-width/five-seed/five-copy coverage, original author-counter reproduction and repaired/equality totals. Maximum finite float/repaired output discrepancy is1.55e-15. Historical sampler-adjusted multipliers are1.220337/1.277031/1.317935/1.390321/1.544201 for H4/5/6/8/16; these remain panel transfers, not direct fresh-panel counts or complete native budgets. Source inspection confirms `sens.py` used broken counted values, so its sensitivity summaries remain unvalidated even though the fresh float-estimator panels are distinct. An initial reviewer filename error (`sensitivity.py` instead of `sens.py`) occurred after numeric checks; it is preserved separately and repaired by changing only the path. Completed review0.004CPU, no model rerun. The separately frozen5120+32 direct recount remains pending.

### T167 complete held-out positive-inner corpus

`check_positive_holdout_full_review.py` applies the previously independent whole-corpus audit to all30,726 seed1 rays, verifying exhaustive indices, shard/reference hashes,64 pilot rows reused bitwise once, every saved ledger and all55 exact original-ray Gram/KKT/determinant certificates and80-digit selected bounds. Accuracy0.9874600098992554 differs from reference0.9874600108260589 by-9.268035379e-10 on exact stored-float sums; conditional bound1.140939366e-8. Counts are16,420 intervals,14,251 quadratures and55 rescues. Known units102,122,861, canonical/staged rule arithmetic18,050/65,236 and rational maxima793/770bits remain separate from full native/model-capture costs. The300/cone goal fails at3323.66; no family impossibility follows. Review0.848CPU, no integration rerun.

### T113 all direct repaired counts and identified-gap envelopes

`check_direct_recipe_counts_review.py` independently checks all5,120 core plus32 borderline rows in81 bounded shards. It verifies source/checkpoint hashes and regenerates every perturbed-weight byte hash from the original checkpoint and declared RNG, without a neural forward or estimator call. All saved estimate/cell parity checks pass. Independent65-digit arithmetic reconstructs per-model integer, continuous and mean-budget ratios and all conditional bootstrap intervals; identified-gap costs use the previously reviewed exact factor, integer ceiling and final+1.

Worst direct historical-convention ratios H4/5/6/8 are0.003121874/0.016240924/0.055613477/0.136569659. The limited numeric-only identified-gap envelopes give0.005772438/0.030943261/0.108468889/0.275535815. The preselected borderline case instead gives1.276916386 with a wide conditional interval; it is retained, not selected away. Author replay441.141095CPU is reconstructed. These are fixed-panel comparisons under declared limited conventions, not complete native cost, certified truths, or a general rung claim. A reviewer whitespace syntax error prevented initial script execution and is preserved; the repaired review core took0.656CPU excluding imports.

### T100 cap observer and first missing-noise side

The source observer adds only a tick at each original range64 iteration and a for-else exhaustion marker. A break/return on iteration63 correctly does not mark exhaustion. F3 resolves F4.trace dynamically; the patched observer reaches that path. Subdivision records depth>=6 with unsatisfied edge criterion without changing decisions. `check_fibre_observer_review.py` checks eight guarded analytic outputs/counters/stats bitwise and independently verifies the noisy model hash and first side normalization by2pi². The preflight observes2631 traces (max11 iterations),784 one-iteration downward walks, subdivision depth3 and no cap events. It is one additive component, not full accuracy or convergence. The remaining runner bijectively maps1..47 to24 families×2 sides, pins observer/model, refuses overwrite and invokes unchanged face_err; final aggregation must reuse preflight0 exactly once in original order. External CPU/memory caps remain required. The tiny saved-data review clock rounded to zero; work is not free.

### T69 fixed random-width diagnostic

`check_random_sector_review.py` verifies all24 saved archives and72 method outputs, reconstructs every He-random weight byte, checks pooled reference means/variances and per-output/block vMSE, and independently recombines each scalar mixture through raw seconds. Mixture beats Gaussian closure12/12 but loses to collapsed h2 on H8seed1. Width-mean trends remain nonmonotone. The1/N subtraction is a nominal reference-noise scale using estimated variance, not exact debiasing or a confidence certificate. Three model seeds per width and different input-dimension-dependent random matrices do not establish asymptotic or causal-width claims. No new reference inputs or closure calls. Review0.072CPU.

### T115 missing-panel pilot provenance

`check_margin_continuation_pilot_review.py` independently reconstructs all73 checkpoint/noise byte hashes, verifies ten dependency/checkpoint pins and every saved truth/seed0 estimator-count-cell control. Maximum independent-reference difference4.44e-16. Index mapping `(noise_seed-20)*256+j` and reuse of the existing pilot rows are consistent with the unchanged noise generator. This clears the separately frozen missing k30 panels; it does not verify their future aggregate or convert the historical72/author-native count into full common-unit cost. Review0.012CPU excludes imports, no model/estimator/truth rerun.

### T171 model-facing adapter source gate and transitive import qualification

The local exact-zero margin guard correctly assigns the whole remaining q to error once when an earlier identical rival owns ties; a later identical rival imposes no constraint. This prevents overlap with already-removed error pieces. The owned two-pass adapter uses zero callbacks solely to collect ordered4x4 ray copies, allocates tolerance from actual N and rounds downward if needed; N=0 is handled. Historical mass/corpus loaders are not called, and capture/storage plus incomplete setup/library costs are explicit.

A follow-up transitive-import audit found that importing pinned m43 merely for Ops nevertheless reads m43_weights.npz at module import. This incidental unrelated weight load contradicts the proposed clean 'no stored models' interface, though no numerical dependence through Ops was found. The owner was notified before adoption and asked to preserve/disclose the issue or isolate the pinned Ops class via AST without executing the module. The earlier source gate covered mass/corpus loaders and is expressly qualified by this finding. Stored floating coefficient equality and full-dimensional cone assumptions remain distinct from an outward exact-weight certificate.

### T172 native RECIPE112 trace control gate

The isolated AST adapter retains the reviewed T128 arithmetic/short-circuit/in-place semantics, dense matmul2k-1 and explicit integer/memory categories. New vector norm2n, cross9 and axis-reduction n-outputs formulas match the stated named-source convention. The owner proactively preserved and corrected a gap: numeric any/all requires comparisons to zero in addition to separately reported Boolean reductions. Legacy ops/STATS diagnostics are intentionally excluded. Builtin sum and libm counts remain named conventions, not compiled internal-operation ceilings.

`check_native_recipe_controls_review.py` independently contracts every site ledger in12 analytic operators and five synthetic models and checks saved bitwise output/legacy/STATS/err_by parity. Positive-volume readout tie failures remain visible. This clears only the two frozen trained parity traces; no full cost or correctness repair is inferred. Review clock rounded to zero, not zero work.

### T115 completed missing k30 panels

`check_margin_completed_review.py` independently regenerates all4,096 perturbed-weight hashes, checks16 complete shards and64 pilot rows reused exactly once, and contracts eight recipe panels plus eight clean rows. Exact aggregate arithmetic and fixed-cost, paired uniform-budget and paired per-copy-budget bootstrap conventions all agree. Worst historical72 aggregate ratio0.1601364314 remains distinct from per-copy performance: r85b loses on one seed1 copy and24 seed4 copies (worst8.03548), and r112 loses on one seed4 copy (1.007864). These failures are preserved. Author shard CPU199.434387 is reconstructed. No estimator or truth run occurred; the remaining truths rely on previously reviewed T101 evidence. Pointwise conditional bootstrap and incomplete native72 costs do not constitute a simultaneous ladder or complete-cost certificate. Review0.340CPU excludes imports.

### T171 import repair and finite analytic outcomes

`check_positive_adapter_review.py` independently loads the corrected adapter with np.load forbidden (zero attempts) and compares17 compiled functions/methods against original engine execution with an Ops-only stub. Only the unrelated import is removed; the local tie guard is the disclosed numerical-semantics repair. All seven saved control ledgers/storage totals are checked. Six active constant-label216-cone cases return1/4 within their conditional bounds; the all-off72-cone case legitimately stops on its first unresolved mass at the unchanged finite order cap. Thus generic total-return remains false. Initial import/source/control evidence stays preserved. Review0.084CPU, import-only and saved arithmetic, no integrations.

### T172 two native trained traces

`check_native_recipe_trained_review.py` reconstructs both perturbed-model hashes, verifies source pins and all output-hex/legacy/STATS/grouped-error parity, then independently contracts every site ledger. H4/H8 named numeric totals165415/458917 differ from repaired scalar137268/326667 and legacy70471/160564; integer categories5756/11593 remain separate. Each source groups24 initial halves into12 err_by keys. The owner's ambiguous24-key wording led to an initial reviewer cardinality assertion, preserved and corrected after exact baseline/traced arrays already matched. No multiplier transfer or full compiled/library/control cost follows. Review0.008CPU excludes imports.

### T156 Q5 exclusive source-price envelope

The pinned Q5 complete-path upper count is4 arithmetic+1negation+2 comparisons in the prologue, at mostL-3 bodies each6arith/4comparisons and two6arith/2comparison tails: `(6L-2,1,4L-6)`, total10L-7. IEEE-only is `(5L+1,1,3L-5)`, total8L-3; observed census lacks the flag, so the conservative envelope is used. Early exits cannot increase it. Index/control/memory work is excluded.

`check_q5_envelope_review.py` checks all64 source/native output arrays bitwise, complete frozen parameter coverage and componentwise bounds. Actual finite exact/recipe Q5 calls201773/165964 produce exclusive named subtotals5949769/4879642. No universal SVD convergence assumption is used; Q3, shift selection, reflectors and other kernels remain unpriced. Do not add full parent costs again without explicit conservative double-count qualification. Review0.004CPU, no native or model rerun.

### T172 first64 native pilot

`check_native_recipe_pilot_review.py` checks all64 compressed site ledgers, source/archive pins, baseline/traced output hex, legacy counters, statistics and grouped errors. Model hashes match the independently regenerated T113 direct panel. Named numeric total30628550 and separate integer categories777410 agree. This clears the frozen continuation under the same source convention; sampler351, compiled/library/control exclusions and lack of multiplier transfer remain explicit. AuthorCPU23.965233, review0.024CPU; no estimator rerun.

### Fresh DRI log-coordinate/cofactor mass source gate (T179)

Pinned estT_fix file ac321d97, SHA25679fccc779e778cdcf28ca521fc0ad59da524b0935b5d6a79aca9bcc31b5d6a6c, uses128 Gauss nodes on v in[0,70], t=1-exp(-v), and cofactor partial correlations. The substitution and cofactor identity are correct for positive-definite exact matrices. For a valid correlation path the omitted true residual tail is bounded uniformly by3*acos(1-exp(-70))/(4*pi), approximately2.13e-16 percone. This is not an interior quadrature bound, and does not establish the comment that every scale is resolved.

The assertion that cofactors remain defined at every singular endpoint is false: rank-one C=all-ones has zero3x3 cofactors, while along I+t(C-I) the exact partial correlation t/(1+2t) tends1/3. This endpoint example is a source-domain qualification, not by itself an invertible-ray counterexample. Floating Rt may round to C before v=70; negative/zero principal minors and raw correlations outside[-1,1] can be hidden by the product floor and clipping. Independent T179 owner56 was asked to record these diagnostics on its fixed rays/fixtures. No numerical ray runs are duplicated here. The reported seed0 aggregate agreement with earlier positive references is useful finite evidence, not a uniform guarantee. Geometry and determinant cutoff are separate limitations.

### T100 first missing noisy fibre case and fixed reference

`check_noisy_fibre_complete_review.py` checks all48 ordered face-side components, source/model pins, legacy counters, guards and observer totals. Estimate0.6056912072571354, legacy1166612480, no walk/trace/depth caps. All100 one-million-sample reference records have consistent exact hits, original initial RNG state and adjacent saved states; total60561725/100000000,312 separately charged parity forwards. This is saved-record verification, not regeneration of100M samples. Independent binomial-tail inversion gives95% interval[0.6055214550337266,0.6057130388520775], containing the estimate. The conditional512-sampler break-even minimum over that interval is4.244161893B; actual estimator cost remains incomplete, so no cost win or noisy-rung claim follows. Author fibreCPU219.944043 and referenceCPU25.585218 are reconstructed. Reviewer initially assumed every component invoked subdivide; the preserved assertion was corrected to accept absent zero-call categories. Review0.012CPU.

### T178 Hopf primitive and saved inference

Pinned hopf.py SHA37e5cb781c74b7c030ac305b5836bc9938963fe5fc56c07eaf1d56ee8d3422fc concatenates all plus roots then all minus roots, but repeats each coefficient mask consecutively. For A=(1,0),B=(0,0), this keeps two pi/2 roots and omits3pi/2, leaving a sign change inside a purported constant interval. Tiling the mask matches root ordering; this repairs that defect alone. `check_hopf_saved_review.py` independently checks eight controls, four original multiset failures, three strict sign-change witnesses and all local-rescue controls.

The same audit verifies25 predecessor rows and independently inverts25 Bonferroni binomial intervals, then contracts30 clean/noise/pooled statistic groups. H64 pooled ratio upper bounds are1.203493,0.818790,0.967619,1.068438,0.835661; all lower bounds are zero. Cross-model independence is unnecessary for the union bound, but each reference must retain binomial provenance. Numerical interval inversion is not outward-rounded. Truncating estimated variance-subtracted MSE at zero is not a certificate of zero deterministic bias. The96-unit denominator and estimator counters remain unclosed cost conventions. Review0.036CPU, no network inputs or estimator runs.

The source's universal continuity statement also needs qualification. For constant class0 output and the Hopf family x=(c cos(phi),c sin(phi),-s sin(phi),s cos(phi)), c²+s²=1, one-sided correct-fraction limits at c=s are1/4 from c>s and1/2 from c<s. Positive cosine-dominant sectors contribute an extra1/4 only on the latter side; sine-dominant sectors contribute1/4 on either side. This rank-degenerate fibre disproves universal continuity, not generic trained convergence. No universal N^-1 rate follows without additional regularity assumptions.

### T179 fixed control source gate

The owner's orthant AST extraction and log-grid expressions preserve the pinned source. NumPy observers return original results and copy diagnostics only. Six groups of128 principal-minor/product/raw-correlation values match source call ordering. Equicorrelation rho1/2 sign-pattern truth1/(5 choose(4,k)) follows exchangeability of five independent Gaussian variables. This clears the frozen256-ray diagnostic panel; the absolute1e-10 control threshold is not a relative-accuracy or every-scale certificate.

### T179 saved controls and frozen256-ray result

`check_log_cofactor_saved_review.py` checks all pinned sources/references,18 control cases and256 original ray indices, and210432 saved raw nodes. Principal-minor products and signed-cofactor partial-correlation reconstruction agree exactly; all floor/clip/negative-minor/nonfinite diagnostics are zero. Maximum absolute difference3.86817e-14, L1 difference2.205955e-13 and signed difference1.467838e-13 agree with saved aggregates. This is a strong finite improvement over the old rule. However149/256 differences exceed the reference's own integration-only bound, and maximum relative difference is1.1297% on a tiny mass. The summed reference bound does not enclose individual source errors or prove the new rule certified. Independent80digit omitted-tail bound2.1287248104e-16 matches the theorem. Review0.180CPU, no source mass rerun.

### T171 trained model-facing capture parity and known cost subset

`check_adapter_capture_review.py` compares every one of35098/30726 captured rays bitwise with prior seed0/seed1 archives, source/model-reference pins and statistics, zero tie-guard branches, saved-value contractions and copied-byte ledgers. New guard units164409/159194 and storage4492544/3932928 bytes are correct. No masses are recomputed and earlier mass work remains charged. Seed0 known subset185465029 includes geometry, cold halves, positive mass, guard and still-executed original-rule setup; separate48-inverse numeric envelope13920 yields185478949. Remaining rank/setup/MP/Fraction/library/control/storage work is explicit, and seed0 geometry cost is not transferred toseed1. The lost110CPU initial validation attempt and all-off finite-order unresolved control remain preserved. Review0.056CPU.

### T172 full H8 native-source budget panel

`check_native_recipe_full_review.py` independently checks all20 archives and1280 complete site ledgers, source/prior pins, baseline/traced outputs and statistics, and model hashes linked to the independently regenerated T113 panel. Named numeric total579318688 and separate integer categories14682950 are reconstructed.65-digit contractions and all five conditional bootstrap calculations agree: ratios0.0003173033,0.0005846604,0.0207068119,0.0248346894,0.1916995096; largest conditional95% upper0.5348052439. Each model receives floor(own named cost/351) sampling draws. This is a fixed H8 panel under a declared source-numeric convention; no multiplier transfer, other-H result, hardware/setup/library/control closure or unconditional certification follows. Author replay465.857325CPU, review0.516CPU, no trace rerun.

### T168/T180/T179 bounded source continuation gates

T168's original moment insertion preserves reviewed T135 geometry/moments, finite variance/reversed-intersection guards, retained counts and unchanged T161 quadrature/exact rescue. Cancellation zeros are preserved and reject the shortcut. This clears only the fixed64 original-form pilot, not a proposed stable-form replacement.

T180's fibre source transformer preserves original numerical functions and the reviewed exact-zero clip guard. Cross9, vector dot2n-1 and mean(n-m)+m are consistent named rules; computed-receiver repair matches prior T153. Lowercase stats diagnostic increments stay in raw totals and are removed separately. Cold setup, SVD and other opaque work remain explicit. Four analytic parity controls clear only one frozen noisy component; no toy-to-H8 multiplier transfer.

T179 full replay amortizes immutable AST/grid construction, clears observer lists and resets percase counters. Persistent STATS simplices does not feed mass values. Reused256 values retain reference equality and their original raw-node archive; new anomalies/sentinels are saved. Prior256 work must be charged once. This is new source mass quadrature, with no new geometry/network inputs/reference integration.

### T178 exact full-circle consequence and continuity controls

The independent angular ordering derivation now agrees with the saved explicit H2T4 model: Wi=(1,0),Wh=0,Wo row1=(1,0), other rowszero; circle a=(2,0,0,1)/sqrt5,b=(0,2,-1,0)/sqrt5. Correctness occurs only in quadrantI with1/2<tan(phi)<1, giving(atan1-atan(1/2))/(2pi)=0.05120819117478336. Original code returns0.125; local tile-only rescue returns the analytic value. Thus the primitive defect does cause an actual synthetic whole-circle error, while saved trained models remain untested by this counterexample. Ten continuity controls also match the derived formula1/2 for r<1 and(3pi/4-atan(r))/(2pi) for r>1. Saved arithmetic check added to the Hopf audit; no new circle runs here.

### T156 reflector exclusive envelope and saved branch counts

`check_reflector_saved_review.py` checks all160 saved native/source outputs bitwise and reconstructs branch counts exactly. Eighteen controls rescale. For nonzero-norm calls with k>=1 rescale passes, exclusive counts are6+3k arithmetic,2neg,2SIGN,1+kabs,2+kcomparisons. At k<=20 this yields the claimed113 bound; no-rescale gives10. Zero norm only uses one numeric comparison, N<=1 none. Child-call counts also match. Shared native children prevent inferring independently priced child arithmetic, and the proposed finite no-rescale census deduction still needs complete observation of all child calls. Review0.028CPU, no native rerun.

### T168 fixed64 original moment-arm result

`check_moment_h8_pilot_review.py` verifies all pins/indices/reference links, saved interval arithmetic and every count ledger.25 moment shortcuts among35 attempts replace quadrature; all39 other values/bounds remain bitwise baseline. Known units109484 versus147062 include cold rules and failed guards; L1 error3.813426e-12 and conditional bound sum1.807217e-11 reproduce. Original floating moment cancellation controls remain preserved, so these conditional bounds are not full numerical enclosures. No full-corpus or complete-cost inference. Review0.000218CPU, no mass rerun.

The proposed stable moment identity is algebraically correct: with symmetric4x4 Gram G,r=G1,d=diag G and mu=(sum G+tr G)/20, Var(y^TGy)=[sum_i(r_i+d_i-5mu)^2+sum_{i<=j}(Gij-mu)^2]/210. Expansion gives the reviewed fourth-moment variance formula. It removes final subtraction of two nearly equal moments; Gram/mean/centering rounding remains outside exact certification. No new stable-form implementation is reviewed in this phase.

### T179 full older-source validation: preserved finite gate failure

`check_log_cofactor_full_review.py` verifies all35098 indices/references,256 pilot rows reused once, all aggregate and corrected determinant-call counts, and288 raw anomaly/sentinel/reused cones. Source accuracy0.9761608074917736 agrees closely with positive0.9761608075466027, but the predeclared absolute gate fails at31514: error2.062884e-10,114.515% relative. Cone4885 is negative; nine cones have108 raw-PC clipping events. L1 error5.466136e-10 and20,059 reference-only bound exceedances remain explicit. No floor/negative-principal-minor events were recorded. The old-source numeric proxy1458032088 is incomplete;80,697,6003x3 and35,0984x4 determinant calls are reconstructed. Initial owner cutoff determinant-count omission stays preserved. Review0.870CPU, no quadrature rerun. This audit applies to ac321d97, not the subsequently changed cf9c17cb kernel.

### T180 first noisy native component

`check_fibre_native_component_review.py` contracts all five control ledgers and one noisy component, verifies prior values/counters/statistics and separates diagnostic stats increments. The trained component has59028138 inclusive units, minus4794 diagnostics=59023344;2595 rows_at calls contribute5436525 named and10380 integer units. Direct dense rows_at expansion gives223 perH2T4 and2095 perH8T4. Ratio1.565406674 to this component's legacy counter is not transferred to the whole model. Cold setup, SVD/other opaque/library/control/memory costs remain unclosed. Initial reviewer empty-mapping assertion was corrected to accept the report's empty-list representation, preserving the failure. Review0.007CPU, no trace rerun.

### T178 source cost inventory qualification

The stated scalar examples check: recurrent coefficients4nH²+2H, gate midpoint/trig/evaluation/masking n(4+6H), root construction11nH plus constant setup with Boolean/control free, readout n(4TH-2T). H4T4 dense sampler executes192 units including correctness; skipping initial zero-state recurrence gives160, or191/159 excluding integer correctness. These do not license changing only the96 denominator while retaining an incompatible numerator, or infer net undercount from selected sites. Sorting, model-free geometry, permodel counts and one consistent convention remain necessary.

### Fresh cf9c17cb source theory gate before numerical controls

Exact cofactor-normal orientation and the atan2 Jacobi formulation are valid for unit-normal SPD correlations. Floating diagonal norms need qualification: the complementary2x2 minor is Rii*Rjj-Rij², not exactly1-t²rho² when Cii differs from1. Raw negative det(Rt) should be observed before the new max floor. The initial inverse is still executed before both new branches, despite the 'no inverse' wording.

A small unit-ray determinant does not imply small cone mass. For rows e1,(-sqrt(1-epsilon²),epsilon,0,0),e3,e4 with epsilon>0, exact mass is atan2(epsilon,-sqrt(1-epsilon²))/(8pi), tending1/8. Every fixed interior Duffy node has a nonzero limiting q because its third/fourth barycentric coordinates are positive, so the fixed1000-node result tends0 with the determinant. This disproves generic thin-determinant accuracy/mass claims; actual ordered-chart restrictions may exclude the counterfamily, so no trained-case failure is inferred. Owner56 was asked to freeze this analytic control before trained validation. No duplicate numerical run here.

### T176 RECIPE85B sort-specific source gate

The isolated flat numeric tuple comparator preserves Python's element identity shortcut, equality before ordering, tuple-prefix length and scalar ordering for the pinned sort() and sort(reverse=True) callsites. It is not a generic key=None adapter. `check_recipe85b_controls_review.py` verifies nine analytic sort ledgers, five initial source-parity models and the separately frozen rational model's12 actual sort comparisons. The initial five models execute no sort comparisons, so their coverage limitation remains explicit. Source/model pins and all saved site totals agree. This clears only two frozen trained controls; it does not transfer T172's other recipe costs. Review0.003CPU, no native rerun.

### T181 stable moment source gate

Only moment.moments is rebound to the reviewed centered-square identity, using the same cached Gram summaries and mean. Diagnostic second=variance+mean² is counted, original selector guards/fallback preserved. Positive recovered nearconstant controls do not resolve the eps2^-28 case where Gram/mean computation already loses information. This clears the fixed64 plus separately charged exact35-moment validation, not a floating certificate.

### T181 stable variance pilot independent exact validation

`check_stable_moment_pilot_review.py` independently reconstructs exact binary-ray Gram matrices and centered-square variances for all35 moment attempts, agreeing with the author's separately enumerated256-monomial references.34 variance errors improve and one worsens near machine precision; worst relative error falls1.139710e-7 to5.087941e-13. All64 saved outputs,39 unchanged nonmoment values, count109414 and70-unit reduction are verified. No integrations rerun; Gram rounding/nearconstant loss and conditional-only mass bounds remain. Review0.024CPU.

### Latest hybrid source fixed-panel qualification

`check_latest_mass_panel_review.py` checks33 controls plus256 saved actual cones and their raw branch diagnostics. Actual117 thin/139 log cases pass the preset absolute gate: max difference1.396907e-15,L1 8.323875e-14;74 differences exceed reference-only bounds. Max unit-diagonal drift4.44e-16 remains. The generic nearly-antipodal unit-ray controls at epsilon2^-20,-24,-30,-40 fail, as predicted; epsilon2^-24 returns7.67e-8 versus trueapproximately1/8. These are explicitly nonblocking for the separately declared actual corpus, not a restored universal primitive guarantee. Review0.064CPU, no source quadrature replay.

### T176 two trained RECIPE85B cases

`check_recipe85b_trained_review.py` verifies source/checkpoint pins, model hashes linked to reviewed T115 data, saved/baseline/traced values, legacy counters, cells and every site total. Named numeric114665/92227, separate integer3838/2772 and40/32 actual sort comparisons reproduce. These two cases do not license a panel multiplier or training-rung conclusion. Review0.001CPU, no trace rerun.

### T178 wider saved evidence

`check_hopf_wider_review.py` verifies all75 source/model-report records and45 groups, previous rows unchanged, uniform/matched budget contractions and75 independently inverted Bonferroni binomial intervals. Three clean point ratios exceed1 (H5s1:1.4873,H6s2:1.9963,H6s4:3.0312), but their simultaneous lower bounds remain below1; do not call them certified losses. Only H4s1/s4 pooled upper bounds and H6s3 noisy upper bound lie below1 among those respective groups. Native96/140/192 costs remain unaudited and larger-grid reference is not exact truth. Review0.128CPU; no model/MC/source run.

### Stable full-corpus resume and Hopf cost control gates

The initial stable full runner could skip a persisted explicit failing row on resume and later mark its shard complete. The owner preserved that source and added a preloaded-row failure/comparison/bound/parity check before even the complete-shard skip; summary rejects the same fields. This harness repair was reviewed before numerical execution. Legitimate finite unresolved rows remain explicit.

T183 Hopf cost adapter preserves original/tile arms, dense einsum, argmax, elementary/modulo and Boolean expression rules, while sorting/search remain opaque with full operand dimensions. Legacy n_ops is diagnostic. The fixed-input actual sampler includes aggregation additions/final division beyond per-indicator192; free-label sorting must remain separate. This clears frozen controls only, not model cost transfer or hardware units.

### Latest T179 full finite corpus, 2026-10-17

Independent `check_latest_mass_full_review.py` checks all 35,098 original indices and positive references, 266 reused records exactly once, source pins, 288 saved sentinel/reused node records, and 65-digit Decimal aggregate sums. All finite absolute gates pass. The source accuracy is 0.9761608075404442 versus positive-reference 0.9761608075466027; maximum per-cone difference is 1.6679639991829976e-14 and L1 difference is 1.018152499059427e-11. Counts independently reproduce 17,955 Duffy10 and 17,143 log128 cases, 35,098 inverses, 13,440,112 3x3 determinants and 2,229,402 4x4 determinants. No candidate or reference integration was repeated.

This is finite saved-corpus validation. The four generic thin-cone stress failures remain; 9,321 candidate discrepancies exceed the reference's truncation-only bound, which is not a shared floating-error enclosure. Neither the minimum sampled radial value nor the excellent aggregate agreement supplies a universal quadrature, floating-point or complete-cost certificate. Cold assignment replay for T180 was separately source-gated with imports and native libraries explicitly outside its counted scope.

### Stable moment full corpus and bounded source gates

`check_stable_moment_full_review.py` independently contracts every one of the 138 saved shards: 35,098 returned rows, 64 pilot rows reused once, 20,329 moment attempts and 12,596 shortcuts. It reconstructs all saved Jensen/Taylor interval intersections, verifies 22,502 non-moment values/bounds bitwise against the prior baseline, and matches all 132 exact rescue certificates to those previously independently reconstructed. All source/reference alignments, saved arrays and cost categories pass. The known-category total is 121,379,672 versus 142,873,286, saving 21,493,614 (15.0438%). Accuracy is 0.9761608085722152 with conditional bound sum 1.2022157253451168e-8. No integrations were repeated. Floating Gram/interval errors and exact-rational, MP, native, storage and full implementation costs remain separate; the original 300-unit goal still fails.

The next thin-Duffy bound primitive was source-gated: exact dyadic center/acute lower bounds and previously certified KKT ranges support the fixed GL10 degree-eight Chebyshev remainder. Its MP evaluation is not directed rounding and does not cover source determinant, nodes, kernel arithmetic or geometry. T183's unchanged numeric timsort shadow was gated only as a source comparator census, not installed-binary equivalence. T180's saved-component outer assembly correctly adds 48 accumulations and a divide/subtract, without repeating component evaluations.

### Native controls, cold setup and outer accumulation

`check_native_cold_hopf_review.py` independently sums all sites/categories for the 15 T183 controls and verifies the exact synthetic root-defect result. The fixed five-input actual sampler records 963 named expressions and 20 separately tagged integer expressions: 960 prediction/correctness expressions plus three within-batch hit additions; two scalar hit accumulations and the final division are integer-tagged. This prevents silent denominator substitution. The small Hopf grid records 34,246 inclusive expressions versus 15,396 historical units and 42 still-opaque sort/search calls; there is no trained-cost transfer.

The same saved audit checks T180's 25 cold-assignment parity records, 6,914 named plus 16 integer expressions, and six SVD/48 inverse/six rule-generation opaque calls. Original outer assembly adds exactly 50 numerical expressions, preserves all 48 family/side visits and the final estimate hex, and repeats no component computation. Imports/library interiors remain excluded. The reviewer initially passed a string to a Path-only JSON helper and received AttributeError before checks; normalizing the helper argument to Path repaired this harness-only failure.

The new three-dimensional determinant controls were source-gated against the pinned small dispatch and runtime function pointers before execution. The visible latest-mass ledger independently sums to common150, thin extra38,001 and wide extra16,953 under its declared expression convention. Combining the qualified small-library category envelopes gives 38,617/121,633 plus two/nine runtime model-free constants. These mixed source/native subtotals are not full compiled-cost upper or lower bounds; native reductions, dot accumulation and Python sum internals remain open.

### Installed determinant3 finite controls and mass-category contraction

`check_det3_saved_review.py` verifies all 18 fixtures/19 matrices across baseline, transparent GETRF/libm interception and runtime child-kernel interception. It independently reconstructs 17 finite exact determinants with the three-term cofactor identity, confirms actual dispatch pointers, and checks IAMAX lengths3/2/1, strided DOT length1, GEMV shapes3x0/2x1/1x2 and SCAL lengths2/1 without solve/GEMM calls. The records contain 19 GETRF calls, 17 successes/two singular outcomes, 51 log calls and 19 exp calls. This validates the finite path observations; the 104-category envelope retains inherited installed-source and numeric-subset assumptions.

Applying the reviewed branch census gives 978,197,934 visible-expression units, determinant3 category envelopes1,397,771,648, determinant4 envelopes392,374,752 and inverse envelopes10,178,420: hybrid subtotal2,778,522,754, with190,197 runtime model-free constants separately. This is not a complete native upper/lower bound and cannot directly supply a sampling-budget ratio. No mass or model evaluations were repeated.

### Actual thin-branch GL10 truncation ranges

`check_thin_duffy_bound_review.py` independently reconstructs all 17,955 stored dyadic Gram matrices, exact determinants via the six-minor identity, center/acute lower bounds and 12 reused KKT certificates. It recomputes every degree-eight Chebyshev tail at 80 digits. The selected bounds use17,909 center/34 acute/12 KKT ranges; 17,901 meet the uniform per-cone allocation and54 are loose. Their total4.769895409406268e-7 exceeds1e-7, despite observed L1 discrepancy2.5714469042375416e-14. Thus this range-bound attempt is insufficient; it does not show the actual rule inaccurate. MP tail evaluation remains non-directed, and source determinant/node/kernel arithmetic, original geometry and the log branch are excluded. No mass integrations were repeated. Sharpening only the54 loose ranges by exact KKT is a distinct bounded rescue, retaining the same rule/target and initial results.

### Hopf sort/search independent saved-order checks

`check_hopf_sort_saved_review.py` independently reconstructs all1,850 saved stable-sort/paired-lexsort permutation hashes using Python stable ordering, with NaN-last semantics, and all40 search outputs using direct rank counts. It verifies the fixed halving-call formula and saved scalar-comparison ranges. No shadow/native rerun was required. These controls establish finite permutation parity and the stated source census, not installed-binary equivalence; compiler child CPU was not separately measured. The separate finite length-four network and closure adapter were source-gated with the integer max_iters allowance and Boolean inversions explicitly retained as mixed categories.

The54-case exact KKT thin-bound rescue wrapper was gated only after adding refusal to overwrite an existing result file; it preserves all initial bounds and charges construction and independent validation separately. Resource protocol for a possible live-stack fibre tracing continuation remains with root: preserving numerical order does not itself authorize resetting a per-process CPU cap.

### Thin-range rescue and finite reflector binding qualification

`check_thin_duffy_rescue_review.py` verifies every one of the54 new exact KKT certificates, reconstructs the fixed degree-eight tails at100 digits and checks the exact decimal merge with17,901 unchanged bounds. The same-rule thin-branch sum becomes3.376800877015286e-11, versus4.769895409406268e-7 initially. No masses or numerical integration rules changed. Non-directed tail evaluation and source floating arithmetic/geometry exclusions remain.

`check_reflector_binding_review.py` verifies12 saved direct/observed controls, all32 saved cold/native phase count equalities, installed DLARFG's two calls through the same DNRM2 PLT entry, and cold-once totals. Every active reflector contributes a norm, rescaling contributes another, and other callers contribute nonnegative counts; equality therefore rules out rescaling only if historical observation/binding was complete. The current binding trace does not retroactively prove that premise. Conditional exclusive totals8,867,710/8,208,700 match; generic113-unit envelopes and unpriced children remain available. No geometry/rank evaluations were repeated.

The Hopf saved-order checker now additionally verifies108 finite network output/permutation records and all ten closure controls. Every original site/opaque ledger, value and legacy count remains identical; each attached lexsort/search/argsort allowance and sum is independently reconstructed. The unchanged one-model h8 original/tile probe was source-gated with saved original parity, model identity and explicit failure preservation; no h64 or wider-panel cost inference follows. Matrix-dispatch controls for the latest mass source were separately gated as transparent GEMM/SYRK shape observations with exact integer fixtures, not kernel prices.

### Matrix dispatch and one-model Hopf cost probe

`check_matmul_dispatch_review.py` reconstructs the exact integer products and every input layout/hash for12 native dispatch controls. Baseline and transparent-interception outputs agree; three contiguous self-transpose cases use SYRK and nine use GEMM. The strided self-transpose case is copied and uses GEMM with distinct pointers. This establishes those finite calls, not their kernel prices.

`check_hopf_model_cost_review.py` checks all pinned source/model dependencies, original saved parity and complete per-site/allowance sums for one H4T4 seed0 h8 probe in original/tile arms. Both return0.4810975064800962 with historical1,032,520 units. Inclusive explicit work plus constant setup is2,324,112; separately qualified source allowances bring it to3,960,665 (lexsort136,633/search42,960/argsort1,456,960). No tile effect is observed in this model, and no generic repair, h64/panel transfer or sampling-cost conclusion follows.

The cellwise T180 resource rescue was source-gated only for preparation and completed control6: original caches are cell-local, exact saved running total/type preserves the original accumulation order, and repeated clamp setup is charged. Independent105/110CPU process caps replace the rejected idea of resetting a live-process cap. Component12 requires successful control6 site/hex/stats/legacy parity first. Process transport and imports remain separate exclusions.

### Directed ideal-rule bound arithmetic

`check_directed_thin_bound_review.py` verifies every17,955 outward radial range and saved formula interval against an independently written100-digit closed form, reconstructs exact endpoint sums and a separate Decimal ceiling sum, and checks all directed category counts. It additionally constructs32 independent rational upper bounds using integer-square-root brackets and exact Machin atan series at the largest bounds and endpoints; each lies below the saved outward endpoint. The directed total is3.3768008770152858034278905999827004789163285339306e-11. This removes the former non-directed evaluation qualification for the ideal GL10 truncation bound under the reviewed Decimal semantics. It does not cover actual source determinant/node/kernel arithmetic, original geometry, the log branch or a full estimator.

The installed DNRM2 source gate confirms N multiplies, N accumulation adds, three final reductions and one square root for positive-stride N>1; wrapper N1 uses an absolute value. Frozen same-order longdouble controls must preserve prior output files and retain architecture, rounding, nonfinite and historical provenance qualifications. No norm/rank execution was performed by this review.

### Resource-safe fibre cell control gate

`check_fibre_cell_rescue_review.py` checks all61 saved cells of completed controls2 and6, ordered parent-cell identity, source/model/previous-file hashes, binary running-total continuity, final hex, stats, actual/corrected legacy and every site-count difference. Nonzero control2 preserves536 pieces and final0x1.4125ee2a14d06p-17; control6 preserves zero output. Only repeated clamp `_fn` setup differs, and it is charged. This clears the unchanged cell-adapter gate for failed component12 under independent process caps; the original114.875CPU timeout and all replay/setup costs remain recorded. No component evaluation was repeated by the reviewer.

The h64 Hopf continuation wrapper was separately source-gated as exact asserted substitutions in the pinned reviewed h8 runner, preserving model/source/closure logic and requiring saved original h64 parity. No broader panel/full-cost claim is authorized by that source gate.

### Installed norm finite census

`check_norm_saved_review.py` checks all192 saved length/stride/pattern controls, including extreme/nonfinite cases, and independently contracts every saved native norm call with one cold phase. Exact/recipe named numeric subsets are6,495,956/5,889,672, each including28,696 cold units. N1 absolute values and N>1 x87 multiplies/adds/sqrt remain distinct; no parent work is double charged. This is a finite installed numeric subset under historical source/binding provenance, not a full native-cost envelope. No norm/rank/model calls were repeated.

The H4T4 seed0 Hopf checkpoint identity83ddefea... agrees with the checkpoint pinned by T110 whose6347 rays feed the reviewed T129 positive reference. That reference can support a one-model comparison with its integration-only3.158e-8 bound and floating geometry/kernel qualifications; it does not make the finite h128 Hopf result exact or justify a panel transfer.

### Saved native dispatch, Hopf h64, and reflector helper reviews
Independent saved matrix child-capture checks preserve all 12 fixture results and reconstruct the actual small-kernel versus ordinary GEMM/SYRK child paths. This establishes finite dispatch evidence, not kernel pricing; layout changes alter dispatch.

Both frozen Hopf H4T4 seed0 h64 arms reproduce .47710001754048736 and legacy65,934,700. All site sums and attached closure records give148,561,137 explicit units and252,465,096 conditional source units. The unchanged result across these two arms is specific to this model. Source-shadow/native-dispatch assumptions, mixed categories and incomplete runtime cost remain explicit; no other-panel ratio is inferred.

All404 saved reflector helper controls and every raw phase contraction pass independently. The conditional reflector/norm/helper subset is31,689,663 exact and29,160,578 recipe; the generic20-rescale helper counterpart is230,514,579/212,199,418. Generic multiplicities are at most2hypot and21scale calls per active reflector, while observed norm calls are charged once. Conditional no-rescale requires the historical complete-observer/binding premise. These omit Q5 and other rank work, character/control/memory and hardware costs.
