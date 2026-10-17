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
