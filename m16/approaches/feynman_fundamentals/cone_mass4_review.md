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
