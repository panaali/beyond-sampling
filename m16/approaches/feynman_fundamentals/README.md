# T55: fundamentals and successive falsifiable attacks

## Question
Can exact low-dimensional calculations be assembled into an efficient deductive
accuracy estimator for the small-model ladder, then the original H16,T10 problem?
Claude remains DRI. This is first-principles reasoning inspired by Feynman's
methods, not impersonation and not a repeat of the failed `sci_feynman` estimator.

## Method
The standing task proceeds in bounded cycles: derive the exact probability,
identify alternative explanations, try the cheapest falsifier, independently
review, and revise. Cycle 1 develops compact normalized-state transport and an
analytic two-input tail primitive. See `theory.md`.

Frozen initial numerical protocol (before implementation): implement weighted
spherical-polygon moments through degree 9; check coordinate octants, rotated
triangles, and thin triangles against an independent positive simplex integral.
Use deterministic cases, no trained checkpoints or model evaluations. Compare
ordinary and high-precision calculations if cancellation appears. Initial CPU
cap 120 seconds, one thread, memory below 2 GB.

Competing explanations and cheapest falsifiers:
1. The radial factor may remain dependent on the normalized state. Prove
   independence only for homogeneous direction summaries; demonstrate that
   absolute-magnitude conditioning invalidates the argument.
2. A plausible boundary recurrence could have wrong orientation or normalization.
   Test full hemisphere assembled from octants and positive simplex integration.
3. An exact expression may be numerically unusable on thin regions. Include
   near-equatorial and nearly coincident vertices, without clipping negatives to
   zero or hiding failed cases.

## Status
Running. Theory derived; T47 agent independently checked the density and Green
identity algebra. No full estimator or solved claim. Runtime recovered after an
outage; the assigned worktree was recreated from snapshot 72edef0. The requested
small-ladder CHECKLIST was absent from that recovered snapshot.

## Results
Protocol commit: `1d68fde`, before numerical implementation and tests.

Reproduce the first primitive check:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python m16/approaches/feynman_fundamentals/check_moments.py
```

`moment_results.json` records nine deterministic triangles, degrees 0 through 9.
Maximum absolute disagreement is 6.89e-15 against independent positive simplex
quadrature and 1.12e-16 against the 100-digit version of the same recurrence.
Summing four octants gives hemisphere probability one within 1.12e-16 for k=1..10.

The thin-equatorial tests falsify unrestricted double-precision reliability:
one degree-9 moment is approximately 1.65e-18 in double precision but 7.56e-70
at high precision and under the positive integral. A geometric upper-bound check
rejects this result and ten other thin-case degree choices. It is a detection
guard, not an error certificate. No probability is silently clipped.

Cycle 2 (`1e5b93c` protocol) now has `tail.py`: it takes a normalized prefix
hidden state plus its top-two values/indices, enumerates the final two steps,
and analytically integrates their rank/readout polygons. Reproduce validation:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python m16/approaches/feynman_fundamentals/check_tail.py
```

`tail_results.json` contains seven fixed synthetic conditional cases (H1T3 and
H2T3/T5/T10). The four H1 cases give the analytic 1/3 to floating precision.
All seven agree with separate 65,536-draw direct validation within 1.83 standard
errors. Multiplying input and output weights each by 1e-16 changes the answers
by at most 1.39e-17. Gate-partition mass differs from one by at most 2.23e-16.
These are conditional probabilities for fixed prefix directions, not solved
long-sequence accuracies. T47's bounded independent implementation review passed;
see `INDEPENDENT_REVIEW.md` and its reproducible `independent_review.py`.

`python m16/approaches/feynman_fundamentals/check_rayleigh.py` adds four closed-form
T4 constant-readout checks, derived by Rayleigh integration by parts in
`theory.md`. They agree within 1.12e-16; see `rayleigh_check.json`. No sampled or
quadrature reference is used for these four tests.

## Compute accounting
Cycle 1 uses zero model forward passes and zero random inputs. Each complete
check uses 184,320 simplex quadrature nodes; the two runs used 368,640 nodes and
approximately 0.481 CPU seconds in total. The saved final run took 0.284 CPU
seconds, including high-precision checks. Full geometric partitioning and outer
transport costs are not yet measured; no efficiency advantage is claimed.

Cycle 2 used 0.241 CPU seconds, 458,752 full model inputs for validation, and
14 deterministic prefix calculations totaling 32 recurrent steps. The analytic
tail calls use no model-input sampling. Normal and rescaled calls together used
1,678 polygon clips, 136 affine products, and 222 weighted-moment calls. No
high-precision fallback occurred; these selected cases did not trip the guard.
The counted operations are not yet converted to an audited FLOP/FPE budget.
The additional four Rayleigh checks used 0.0613 CPU seconds, four deterministic
prefixes/eight recurrent steps, 208 clips, 32 affine products and 29 moment calls;
zero full model-input evaluations.
The independent reviewer additionally used 0.1345 CPU seconds, six 1-D positive
integrals and 262,144 direct conditional validation inputs. These are separate
from the author totals. No shared leaderboard or trained checkpoint was run.

Cycle3 positive fallback: `positive.py` supplies a positive integral with an
analytic quadrature-only error bound. The T48 agent independently checked its
proof in `theory.md`. `tail.py` now automatically invokes the fallback after a
recurrence guard failure and records its nodes/count/error bound in statistics.
The old `moments.tail_mass` still exposes original failures for regression checks.

Reproduce under the shared compute lock:

```
flock /workspace/work/codex-compute.lock env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python m16/approaches/feynman_fundamentals/check_positive.py
```

Seventy configurations (seven polygons, five k values, two tolerances) used
91,472 nodes, maximum order96, and0.088 CPU seconds. Maximum disagreement with
100-digit recurrence was1.48e-15. Twenty-four cases exceed their extremely small
quadrature-only bounds because floating-point errors are separate; all are
preserved in `positive_results.json`. No full numerical certification is claimed.
The T51 agent supplied `trained_moment_failure.json`: its k2 guard failure is
resolved using16 positive nodes, agreeing with high precision to1.17e-27 absolute.
A separate direct regression call used approximately0.038 process wall seconds;
neither check used model forward passes or random inputs.

Cycle4 adapter: `target_tail.target_probability(model,a,b,K,j)` provides the
two-fresh-input terminal on T50's normalized target/count state. It inserts the
preselected target in the correct chronological location and handles count2
absorption, axes/origin, b=0 and first-index ties. Some arbitrary grid states
are unreachable; the callable defines an operator extension there.

Reproduce `check_target.py` with the same flock/thread prefix above. Twelve
fixed cases used786,432 direct validation draws in65,536 chunks and0.332 CPU
seconds. Maximum discrepancy was2.26 SE; input/output scaling by1e-16 changed
the kernel at most1.12e-16. For synthetic T3, summing all three target-specific
passes gives .32421191293570956 versus .3242119129357096 from the separately
partitioned top-two tail (difference5.56e-17). No positive fallback occurred.
The kernel and its validation are not a full estimator or an efficiency result.

Statistics now explicitly use `positive_quadrature_fallbacks`; `positive_fallbacks`
is a compatibility alias. `high_precision_fallbacks` remains0 and does not count
quadrature. Additional fields expose quadrature-node counts, quadrature-only
error bounds, newly generated Legendre rules and their orders' node counts.
`method_class` states `analytic_regions_with_numerical_quadrature_fallback`.
The numerical fallback must not be described as exact arithmetic or as covered
by a floating-point certificate.

## Claims
- [inferred] H2 accuracy admits a target-specific Markov formulation with three
  continuous normalized coordinates and a capped exceedance count, independent
  of sequence length. Low state dimension does not imply a cheap representation.
- [inferred] Exactly two remaining Gaussian inputs can be integrated over weighted
  spherical polygons; the density and recurrence have independent algebra review.
- [measured] The bounded primitive check agrees with independent positive
  quadrature to 6.89e-15 absolute error, but unrestricted relative accuracy fails
  on thin near-equatorial regions.
- [speculative] Combining these primitives could be competitive with sampling.

## Alternatives not ruled out
Exact geometry may still cost more than sampling. Compact-state laws may acquire
too many singular pieces. Reset decompositions may only help special checkpoints.
No generalization or computational advantage follows from the current derivation.

## Failure modes
Absolute-magnitude conditioning breaks radial independence. Positive-volume logit
ties require first-index argmax; boundary measure-zero arguments do not remove
them. Floating-point cancellation can invalidate tiny-region probabilities.
Random outer prefixes produce a sampling baseline, not a deductive solution.

## Continuity
Keep this task active across successive cycles. Root owns backlog updates,
integration and communication with Claude; only this folder is agent-owned.

## Cycle 2 protocol (frozen before implementation)
Build a two-final-input conditional probability from normalized prefix hidden
state, its largest two inputs and their indices. Enumerate only the two tail
steps, then rank/readout polygons; integrate with the weighted sphere primitive.
No long-prefix estimator or trained-model performance claim in this cycle.

Alternatives: (1) correct moments but incorrect rank/first-index tie handling;
test H1 zero-recurrence/all-tied outputs and deterministic direct classifications;
(2) truncated geometry or normalization produces biased probabilities; compare
conditional outputs against independent random-direction validation, include
positive rescaling and partition-mass checks; (3) work per tail outweighs variance
reduction, so record geometric operations and CPU without asserting an FPE gain.

Use fixed synthetic H1/H2 cases, T3 and T5/T10 prefixes, at most 65,536 independent
validation draws per case, at most 12 cases initially. Single thread, 120 CPU
seconds total initial experiments across both cycles. Random draws are validation
only; the primitive itself is analytic. Preserve all failure cases. High-precision
fallbacks must be counted; no silent clipping of small negative probabilities.

## Cycle 3 protocol (frozen before implementation)
Build a positive simplex-quadrature fallback for weighted spherical moments with
an explicit analytic quadrature-error bound. It must expose the bound and node
count, rather than accepting agreement between two numerical approximations as
certification. The bound will be conditional on supplied geometry and real
arithmetic; floating-point clipping/rounding remains a separate qualification.

Two alternatives: (1) Duffy degree counting or the positive-weight error factor
is wrong; obtain an independent algebra review and compare against exact octant
moments. (2) The bound is correct but too expensive; test ordinary and thin
triangles from cycle1 at fixed absolute tolerances 1e-8 and1e-12, recording nodes
and CPU. The cheap falsifier is a bound smaller than observed high-precision
error, or inability to reach1e-12 within128 nodes per dimension.

All numerical runs acquire `/workspace/work/codex-compute.lock`, use one BLAS
thread and remain below2GB RSS. Cycle budget120CPU seconds. No trained T4 outer
integration: its assigned owners retain that work. Root receives checkpoint
commits and the standing investigation continues after each bounded cycle.

## Cycle 4 protocol (frozen before implementation)
Provide T50's backward collocation owner an exact-formula terminal kernel on
(a=h/R,b=u/R,K,j), with two fresh non-target inputs remaining. If j is already
processed, perform two random steps; if j is penultimate, insert bR between them;
if j is last, insert it after them. Radius dimension is always T-2. K=1 requires
both fresh inputs below u; K=0 requires exactly one above. Positive-volume
logit ties use first-index argmax, including dead states. Grid construction and
interpolation remain T50-owned, not part of this cycle.

Competing explanations: (1) wrong target insertion order/radius normalization;
test every target position of synthetic T3 against the independent top-two tail
and direct conditional simulation. (2) count boundaries or zero/axis states
silently lose mass; include a=0, b=0, counts0/1/2, identical output rows and
positive input/output rescaling. Up to12 fixed cases with65536 validation draws
each, one bounded locked run initially. Keep every failure. Analytic kernel
calls use no sampled model inputs; the check's random draws are validation only.

## Cycle 5 protocol (frozen before implementation)
Supply T48 with a stable vectorized formula for the inverse-sixth-power norm
integral along a segment whose endpoints share an orthant. This analytically
eliminates the last Duffy coordinate in T6 spherical-simplex integration.
Alternative 1: the angular expansion has a missing scale or Jacobian factor;
compare independent high-precision integration on equal, orthogonal and unequal
endpoint norms. Alternative 2: the formula cancels near parallel endpoints;
use a positive coefficient series with a geometric remainder bound, test
cosines near one and verify positive scaling. Preserve all discrepancies.

Initial tests use fixed endpoint pairs, no model samples, one locked numeric
run with at most120 CPU seconds and at most2GB RSS. Report scalar endpoint
evaluations, coefficient-series terms and actual CPU. The series bound excludes
floating arithmetic and the closed-form coefficient branch; do not label it a
full numerical certificate. T48 retains trained T6 integration ownership.

Cycle5 result: `segment.inverse_six_segment(A,B,diagnostics=False)` broadcasts
endpoint arrays of shape (...,dimension), returning one analytic integral per
pair. With diagnostics enabled it returns (values,stats); the stats include an
array of series-truncation bounds, NaN for the closed-form branch. Independent
80-digit direct quadrature of28 fixed pairs (norm ratios through10000, equal
and perpendicular directions, branch threshold, and nearly parallel inputs)
gave maximum relative error3.411e-15. Scale checks gave5.996e-15 and endpoint
swaps were identical. This is a validated primitive, not a solved T6 model.

Reproduce with single-thread environment variables and
`flock /workspace/work/codex-compute.lock python m16/approaches/feynman_fundamentals/check_segment.py`.
Measured author CPU1.5765 seconds for the entire check;84 analytic endpoint
evaluations including rescaling and swapping, no model passes or sampled
inputs. The main28 evaluations used56 norms,28 dot products,16 positive-series
evaluations of28 terms and12 closed-form evaluations. Independent numerical
reference quadrature is validation overhead, not part of the estimator.
Results and every tested endpoint are saved in `segment_results.json`.

## Cycle 6 protocol (frozen before the next validation)
Formalize the H2 support decomposition before proposing a density transport
implementation. Hypothesis: from zero initial state, its law has only an
interior density, two axis densities, an origin atom, and possibly one input
ray; after a fixed target insertion, two fresh inputs restore this support
under explicit nondegeneracy conditions. Alternative1: an omitted rank-one
gate cell creates another interior line; examine the final active gate matrix
and supply degenerate counterexamples. Alternative2: mathematically correct
supports still miss their weights numerically; independently integrate the
four-component toy law already derived and compare masses with exact angles.
This bounded cycle uses only scalar quadrature, no model-input samples. It
does not implement T50's transport driver or claim its interpolation is solved.

Cycle6 result: `strata_theorem.md` proves the support decomposition directly
from the rank of the final gate-cell map. T47 independently checked both the
unconditional theorem and the two-fresh-input conditional smoothing statement;
the latter requires both input weights nonzero and det(W*Wi,Wi) nonzero.
Explicit counterexamples show each hypothesis matters. This finite support
list is not a finite exact parameterization of its component densities.

`check_strata.py` independently integrates the four toy components with scipy
quadrature. Their masses are .32379180882521663, .25,
.051208191174783355, and .375; the sum is1 and maximum discrepancy from
the angle formulas is5.552e-17. Cost:870 scalar integrand evaluations,
.002456CPU seconds, zero model passes or sampled model inputs. Run under the
same single-thread flock wrapper as `check_segment.py`, substituting
`check_strata.py`. Saved output is `strata_results.json`. The first attempted
command used an incorrect file location and exited before computation; the
files were moved into the owned folder and the documented run then succeeded.

## Cycle 7 protocol (frozen before implementation)
Eliminate the last TWO simplex/Duffy coordinates via a weighted spherical
triangle moment, as proposed by T48. For ambient dimension d the exponent must
be d-3 and the spherical Jacobian must contain exactly one abs(det R), with no
factor2. Alternative1: a Jacobian/coordinate convention is wrong; compare
orthogonal unit endpoints against explicit sphere monomial moments for d3..6,
and other fixed triangles against independent nested quadrature. Alternative2:
QR/determinant or boundary recurrence is unstable near rank deficiency; expose
conditioning, reject suspect analytic values, and use positive quadrature on
the ORIGINAL affine triangle with a stated real-arithmetic error bound.

Include exactly collinear and near-collinear endpoints, positive scaling,
endpoint permutations, and count QR, condition estimates, solves, moment calls,
fallback nodes and rule construction. This is a primitive check only: no
trained integrations, no sampled model inputs, one bounded locked numeric run.
T48 and root independently checked the measure and coefficient exponent.

Cycle7 result: `triangle.py` exposes `triangle_inverse_norm(vertices, exponent,
absolute_tolerance=1e-11, known_plane_normal=None) -> (value,stats)` for three
endpoint rows in a common orthant. The scalar prototype uses weighted moments
on well-conditioned triangles and explicit positive original-domain quadrature
on suspect ones. For signed-L1-normalized Duffy endpoints, supply the orthant
sign vector as `known_plane_normal` to replace the plane solve by a projection.
The full measure proof and limitations are in `triangle_theory.md`.

Across16 fixed cases (ambient/exponent3..6, ordinary, orthogonal, nearly
collinear, exactly collinear), independent nested quadrature differed by at
most6.662e-16 relatively. Scaling and endpoint-permutation differences were
4.441e-16 and1.111e-15. Twelve further signed-L1-normalized cases using the
known normal differed by at most1.111e-15. The final validation run used60
primitive calls (30 analytic,30 fallback),12576 fallback nodes,51912 reference
integrand evaluations, and.242807CPU seconds, no model inputs. Full geometry,
special-function and rule-construction counters are in `triangle_results.json`.
Three primary fallback discrepancies exceeded their much smaller quadrature-
only bounds, exposing roundoff rather than a full certificate. No full-model
cost or accuracy claim follows from these primitive checks.

Reproduce with the same single-thread setup and
`flock /workspace/work/codex-review.lock python m16/approaches/feynman_fundamentals/check_triangle.py`.
The root approved this second lock for checks below20CPU seconds. Two earlier
runs before the final cost metadata and known-normal extension used.154219 and
.173212CPU seconds,48 primitive calls and28812 reference evaluations each;
these runs count toward author development cost rather than being hidden.

## Cycle 8 protocol (frozen before implementation)
The scalar triangle method is too costly for millions of outer nodes. For
signed-L1-normalized endpoints, construct a tangent basis from B-A,C-A and
the perpendicular component of A; its norm is at least1/sqrt(d). This avoids
per-triangle QR, SVD and plane solves. Vectorize spherical moments of degrees
1..3 using elementary edge integrals, and retain scalar guarded fallback for
suspect geometry. Alternative1: orientation or edge moments are wrong; compare
fixed batches against the reviewed scalar implementation for d4..6, including
endpoint permutations. Alternative2: tangent Gram-Schmidt loses thin geometry;
test a predeclared progression of widths and expose all fallback cases/errors.
Bound initial geometry benchmark to4096 deterministic triangles and at most20
CPU seconds through the approved review lock, no sampled model inputs.

Before adoption, extend the fixed width sweep to asymmetric triangles near a
simplex corner, whose plane-normal foot lies outside the triangle. These produce
mixed boundary signs and can expose cancellation hidden by the symmetric sweep.
Test widths .1 down to0 against positive original-domain quadrature, record all
heuristic-screen rejections, and retain the initial benchmark cost separately.

Cycle8 result: `triangle_batch.signed_l1_triangles(vertices, exponent=d)` takes
shape(...,3,d) signed-L1 endpoints and returns(values,stats). Degree1..3 moments
use vectorized tangent geometry and elementary arcs, without QR, SVD or solves.
The current4096-triangle d6 throughput check took.005956CPU seconds;32 slower
scalar comparisons differed by at most1.999e-15. The broader symmetric AND
asymmetric width sweep is more demanding: maximum relative differences from
positive original-domain integration are1.765e-11,2.871e-11,5.139e-11 for
d4,d5,d6. These are preserved, so no uniform near-machine-precision claim is
made. Every numerical screen and fallback contributes explicit counters.

`triangle_batch_results.json` records45 width cases, their endpoint-permutation
checks, the4096-triangle benchmark and32 scalar comparisons. Current full
check cost.033726CPU seconds, no model passes or sampled model inputs. Earlier
checks before adding asymmetric cases and Taylor-tail metadata cost.032893
and.032589CPU seconds. Reproduce under the approved single-thread review lock
with `check_triangle_batch.py`. The small-angle series exposes its real-
arithmetic truncation error; both floating geometry and the1e-10 roundoff screen
remain uncertified. This is ready for independently owned outer-driver pilots,
not a trained-model speed or challenge-solved claim.

## Cycle 9 protocol (frozen before implementation)
T50 identified redundant moment work: a final degree-p moment uses only its
matching-parity recurrence chain, whereas polygon_moments computes every
degree. Preserve polygon_moments as a reference and add a final-value evaluator.
Alternative1: wrong base case or parity indexing changes values; compare fixed
ordinary/thin polygons and all degrees0..9 against the unchanged full evaluator.
Alternative2: fewer Fourier terms do not save useful time; benchmark fixed
degree7 calls and count every boundary term, area evaluation and CPU. Keep
failures/guard behavior unchanged, do not introduce a new clipping rule.
This bounded check uses no sampled model inputs and the approved review lock.

Cycle9 result: `polygon_moment(vertices,degree,stats=None)` evaluates only the
needed recurrence parity, while the original `polygon_moments` stays unchanged
as reference. Tail mass now uses the single-value path and exposes edge-integral,
Fourier-term, area and recurrence counters through its existing optional stats.
All90 degree/case comparisons across ordinary, octant, thin-polar and near-
equatorial triangles were bit-identical, including numerically bad moments
that the existing guard must still reject. No new clipping rule was added.

For300 fixed degree7 calls, Fourier terms fell from25200 to14400, edge integrals
from6300 to3600, and unused area evaluations from300 to0. Measured CPU fell only
from.221137 to.204871 seconds in this small run (about7%); do not equate the43%
term reduction with a43% end-to-end speedup. The guarded tail still computes its
required geometric area. Total validation/benchmark CPU.467380 seconds, no model
passes or sampled inputs. Source hash and exact results are in `parity_results.json`.
Reproduce under the review lock using `check_parity.py`. T50 retains independent
cached-terminal comparisons and full estimator cost measurement.

## Cycle 10 protocol (frozen before validation)
Remove repeated target-coordinate interpolation by conditioning directly on
the target being second-largest. Its input CDF q has Beta(T-1,2) law; conditional
on no greater input yet among m remaining non-target positions, the next one
exceeds the target with probability1/m. Validate the combinatorial recursion
and resulting two-dimensional fixed-target transport before implementation.
Alternative1: observing hidden state changes the remaining-position law;
prove conditional independence from the full processed history, not merely
the unconditional1/m marginal. Alternative2: the outer rank normalization is
wrong; check constant-readout accuracy1/T and the exact H1/zero-recurrence
sign-classifier formulas for T3..10 using independent scalar quadrature.
No transport implementation or sampled inputs in this cycle; separate owners
must claim the follow-up. Truncating original inputs at a common symmetric
bound gives an explicit missing-mass interval rather than an assumed grid box.

## Priority regression: original-triangle radial bound
T48's full T6 partition stopped on a documented bound failure after2505 saved
cones; the isolated failed global cone is3123. Its64 endpoint triangles are
preserved in the supplied NPZ fixture. The centroid projection is valid but
can underestimate the available radial lower bound. For a common orthant sign
vector s, Cauchy gives ||sum y_i v_i||² >= min_i(s dot v_i)²/||s||².
Signed-L1 endpoints therefore have q>=1/d, including arbitrary thin triangles.
Test the saved failure before and after adding the maximum of these two valid
bounds; retain tolerance1e-11 and max-order128. Alternatives: the known-plane
assumption fails on supplied endpoints (verify residual), or the resulting
Taylor bound still fails (preserve exception and do not drop mass). This fix
takes priority over the pending cycle10 derivation.

Regression result: reproduced the original exception, verified signed-L1
residual2.221e-16, then added the valid orthant-plane lower bound. Minimum qlo
rose from.07913941145 to1/6. All64 formerly problematic triangles now complete
at the unchanged tolerance1e-11 and max-order128, using150384 positive nodes;
the largest individual quadrature-only error bound is9.406e-12. A separate
160-point-per-axis positive reference (1638400 nodes total) differs by at most
3.997e-14 absolute and9.659e-15 relative. Final regression CPU.038162 seconds,
no model passes or sampled inputs. Saved fixture and full source hashes are in
`triangle_failure.npz` and `triangle_bound_results.json`; reproduce via
`check_triangle_bound.py` under the review lock. Prior fixed-case result files
retain their old source-era counters rather than silently rewriting history.

Cycle10 derivation is now complete in `rank_conditioned_theory.md`, including
the conditional-on-full-history proof, Beta outer law, finite Gaussian-box
missing mass, and final scalar-interval kernel specification. T67 owns the
new transport implementation. Independent scalar integration for T3..10
recovered constant and sign-network toy accuracies with maximum discrepancy
2.776e-17. At T10 the rigorous omitted-box masses are6.33244e-4 for L4,
5.73302e-6 for L5 and1.97318e-8 for L6. Validation used1344 scalar integrand
evaluations,.000724CPU seconds, no model passes or sampled inputs. Reproduce
`check_rank_conditioned.py` under the review lock; results are saved in
`rank_conditioned_results.json`. These controls establish the measure algebra,
not trained-model collocation accuracy or an advantage over Monte Carlo.

## Cycle 11 protocol (frozen before validation)
Derive the law of a fixed linear map of a normalized Gaussian direction,
including singular ellipsoid surfaces when the map rank equals the number of
underlying Gaussian coordinates. Alternative1: image rank incorrectly predicts
volume dimension; compare the k=r surface Jacobian with a direct parameterized
area element. Alternative2: applying the unrestricted projection density to a
ReLU gate cell ignores hidden angular fibers; derive an explicit gate-weight
counterexample and check its mass/first moment by separate scalar integration.
Verify normalization and second moments for low ranks and k values using
deterministic quadrature only. Keep existing T48 primitives frozen; add theory
and checks in new files. T58 and T67 retain their respective engine ownership.

Cycle11 result: `projection_strata.md` derives both ellipsoid-ball densities
(k>rank) and ellipsoid-surface densities (k=rank), with the gate-cell conditional
fiber factor. A k2/rank1 gate example has mass1/2 but first submoment
-1/(pi*sqrt(2)); replacing its gate factor by an independent1/2 would silently
get that moment wrong. This sharpens the earlier finite-support theorem rather
than assuming unnormalized and normalized laws have the same strata.

`check_projection.py` verifies eight ball-density normalizations/radial moments
to4.441e-16, ellipse/ellipsoid surface masses and coordinate second moments to
3.4e-15, and the gate-fiber example to2.8e-17. Cost:2560 surface parameter nodes,
1176 scalar integrand evaluations,.002028CPU seconds; no model passes or sampled
model inputs. Run under the review lock; output is `projection_results.json`.
T58's current unnormalized transport and T67's fixed-target transport avoid
this normalized-sphere issue. Existing T48 primitives were not modified.

## Cycle 12 protocol (frozen before validation)
Derive a generic positive Chebyshev-tail bound for (m+delta*x)^(-nu), nu>=1,
including half-integer nu=2.5 used by T5. Factor the denominator into conjugate
linear Fourier factors, then bound the positive coefficient ratio and residual
hypergeometric factor. Alternative1: Fourier/Chebyshev normalization misses a
factor2; compare individual coefficients with independent high-precision cosine
integrals. Alternative2: the proposed geometric ratio is not an upper bound;
compare coefficient ratios and summed positive tails with the exact nu3
polynomial tail independently derived by T48. This is theory/bound validation
only, no estimator/source changes or new model integration. Use a short review
lock run, retain discrepancies, and report special-function evaluations.

Cycle12 result: `chebyshev_positive.md` proves the positive generic coefficient-
ratio bound, independently algebra-reviewed by T48. `chebyshev_positive.py`
provides a small bound evaluator; no estimator imports it automatically.
Twenty-four checks over nu1,2.5,3,5, interval lower bounds.2/.95 and first omitted
degrees1,4,12 matched independent70-digit cosine-integral coefficients within
5.804e-55 relatively. The exact nu3 summed tail matched the independent
positive endpoint remainder within7.787e-51. Generic bound/exact-tail ratios
ranged from.9999999999999994 to1.8963: the tiny lower-than-one value exposes
ordinary float rounding for the exact nu1 case, so this is not a certified
floating bound. Cost:.949231CPU seconds,32496 high-precision cosine integrand
evaluations,160 high-precision hypergeometric evaluations and24 float bounds;
zero model passes or sampled inputs. Run `check_chebyshev_positive.py` through
the review lock. Existing trained integrations and their source hashes remain
unchanged; T48 owns any later adoption.

## Cycle 13 protocol (frozen before validation)
T67 requested independent stability support for exact quadratic-Gaussian scalar
transitions. Verify the supplied positive Gauss fallback bound after centering
an interval, using the exp(-c*h*t-h²*t²/2) Taylor polynomial. Alternative1: the
degree count or positive-weight factor is wrong; compare high-precision Gauss
quadrature to independent high-precision scalar integration. Alternative2:
the real-arithmetic bound is mistaken for a float certificate; separately record
ordinary-float discrepancies on narrow intervals and do not claim the bound
covers them. Test fixed centers0,+/-6, half-widths1e-2/1e-4/1e-8, and three
nonnegative quadratic polynomials. Only theory/check files; T67 owns the engine.

Cycle13 result: `quadratic_gaussian.md` proves the centered four-point-positive-
quadrature fallback bound and records conditional normalization/error propagation
requirements. Twenty-seven fixed cases passed the100-digit quadrature check;
the largest actual high-precision quadrature error was.002903 times the stated
bound. Ordinary-float total relative errors reached3.484e-13, and12 cases
exceeded their tiny quadrature-only bounds due to arithmetic, explicitly retained.
Cost:.638798CPU seconds,17550 high-precision integrand evaluations and108 float
quadrature nodes, no model passes or sampled inputs. Reproduce
`check_quadratic_gaussian.py` through the review lock. This validates the
fallback theorem; T67's engine still needs its own integration/rounding controls.

## Cycle 14 protocol (frozen before validation)
Independently review T71's odd-dimensional tetrahedron reduction and T72's
outward T4 cubature enclosure without modifying either engine. Alternative 1:
thin-face cancellation or a measure normalization error invalidates the T71
formula; derive its cofactor signs, equal-endpoint limit and positive fallback
degree count, and specify an asymmetric thin-tetrahedron falsifier. Alternative
2: outward arithmetic encloses individual operations but misses quadrature
truncation or algebraic-rule uncertainty; derive T72's rational tail constant
and compare its raw GL20 interval with independently generated 70-digit GL20
cubature for the orthant and two fixed artifact simplices (indices 0 and 128).
The numerical check is at most 20 CPU seconds on the review lock, with no
model passes, full-artifact sweep or changes to frozen integration primitives.

T71 requested the asymmetric-coalescence numerical falsifier as well. Freeze
p=(.8,.05,.05,.05,.05), endpoints p+epsilon*(e_i-p), i=0..3, and epsilon
1e-2 through1e-8. Compare against independent70-digit original-domain tensor
GL12; retain the analytical branch/fallback choice, errors, permutations and
scale checks. These nearly constant denominators make GL12 truncation tiny;
the reference is still a numerical check rather than an interval certificate.

Cycle14 result: `t71_t72_review.md` independently derives T71's cofactor signs,
measure and fallback normalization, and T72's exact rational truncation tail.
Independent70-digit GL20 cubature lies inside all three raw T72 intervals:
orthant, artifact cones0/128. Cost1.530285CPU,24000 high-precision nodes, no
model inputs. The orthant finite-rule value .06250000000000148965 differs from
the exact probability1/16, correctly covered only after the truncation bound.
The asymmetric T71 test used12096 high-precision GL12 nodes and1.095377CPU.
Two analytic cases have maximum relative error8.947e-13 and permutation
difference2.324e-12; five narrower cases trigger fallback and agree within
3.416e-16 relative. This is useful evidence for the current rejection policy,
not a rounding certificate. Some fallback rounding discrepancies exceed their
tiny quadrature-only bounds, as already expected and explicitly qualified.
Reproduce `check_enclosed_review.py` and `check_tetra_adversarial.py` separately
under the review lock; result JSON files contain source hashes and inventories.

## Cycle 15: read-only mixed-measure kernel review
Question: does T58's conserved finite-volume mass also establish its singular
strata or prediction probabilities? Alternative1: a missing line/area Jacobian
causes discrepancies; independently derive each source transfer. Alternative2:
the kernel is correct but uniform cell projection leaks outside reachable
support; test the exact toy invariant h2>=h1 against saved fourth-step output,
without running a new numerical experiment. `mixed_measure_review.md` records
the derivation, a null-set pointwise API convention, and the retained numerical
counterexample. No primitive or estimator code changed; no additional model
or quadrature evaluations. This supports the assigned T58 owner, not a new
transport implementation or an accuracy certificate.

## Cycle 16 protocol (frozen before validation)
T71 requests independent checking of its fixed per-cone tangent frame. Derive
the cofactor scaling and inverse-Frobenius condition bound, then reuse the
asymmetric cone p=(.8,.05,.05,.05,.05) with remaining rays e0..e3. Freeze outer
nodes .99 and .999, the previously accepted analytic cases. Alternative1:
the cached tangent formula loses a power of1-u or delta; compare with freshly
generated70-digit original-domain GL12. Alternative2: row/column orientation
works only for one ordering; repeat after permuting the last four rays. No
trained sweep, one short review-lock check, no production-source edits.

Cycle16 result: fixed tangent cofactor/Jacobian and condition-screen derivations
are recorded in `t71_t72_review.md`. Both accepted analytic cases agree with
fresh70-digit original-domain GL12 within4.360e-14 relatively; permutation
difference<=3.931e-14. Cost3456 high-precision nodes,.410647CPU, no model inputs.
`check_fixed_tangent_review.py` reproduces the result; source hashes are saved.
This validates the algebra and two adversarial cases, not general roundoff.
