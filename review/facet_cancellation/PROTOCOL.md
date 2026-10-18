# T267: fixed H4T4 oriented-chain numerical discriminator

## Frozen scope and order

One process: eight exact synthetic chains, six rejecting guards, then one existing
H4T4 seed0 exact complex (all 5058 correct simplices, 3853 gate simplices used only
as a provenance count). No new geometry, model, label, RNG, checkpoint, or network
calls. The model in the archived JSON is decoded as metadata only. No H4T3 terminal
archive was found in the bounded source inventory; that rung remains outstanding.
No outcome-dependent subset, anchor, quadrature order, or stopping rule is used.

The synthetic controls must pass before the archived complex is decoded. The
combined source/theorem/resource gate covers this ordered single phase; independent
saved verification follows, with no author replay. The eight cases are a positive
orthant, its two-simplex split, a duplicate, reversed vertex order, positive rescale,
negative orthant, a deliberately nonconforming internal face, and a 3D orthant.
Exact constant, linear, and quadratic integrals must agree after reduction. Six
guards cover empty input, zero/duplicate rays, mixed orthants, floats, and a wrong
archived determinant. Expected facet counts are fixed in fixtures.py.

## Exact chain identity and limitations

For each simplex, primitive integer rays are sorted lexicographically. Normalize
each ray to L1 norm one. The same-sign coordinate vector s identifies a single
orthant, whose affine cross-section is s.x=1. Its anchor is a=s/d. A simplex has
positive Euclidean measure but is assigned orientation sign(det(sorted rays)).
Its oriented boundary facet omitting vertex i has coefficient orientation*(-1)^i.
Keys are (orthant, sorted primitive facet rays); sum integer coefficients exactly.
Only exact identical facets cancel. Coefficients with magnitude greater than one
are retained, not collapsed to membership.

The oriented affine simplex identity is [v0,...,v(d-1)] = sum_i (-1)^i
[a,v0,...,omit(vi),...,v(d-1)] as integration chains. It follows by the cone
homotopy formula, or by signed pyramids for an interior anchor and additivity as
the anchor crosses supporting hyperplanes. It holds for every integrable density
on the cross-section. For the Gaussian radial reduction the common density is
q^(-d/2); for d4 its coefficient is 1/(2*pi²). Each retained cone contributes
coefficient*sign(det(anchor,face)) times its unsigned positive-cone mass.
Degenerate anchor cones have determinant zero and contribute zero. Every nonzero
cone stays inside its original orthant, so q is between 1/4 and 1 in dimension4.
Per-original-simplex exact normalized determinant identities are checked and saved.
They are diagnostics, not the sole proof of the weighted identity.

This is a reduction of the archived oriented chain. Exact inherited cone coverage,
correctness, ties, and disjoint interiors are prerequisites, not recertified by
matching. Independent prior geometry reviews are pinned. Different triangulations
of the same geometric face need not match and remain in the chain. Orthants use
different anchors, so artificial coordinate-plane interfaces are not cancelled
across orthants. Remaining facets are therefore not asserted to be true decision
boundaries, and failure of this variant is not a lower bound for all cancellation
or coarser polyhedral-boundary schemes. No affine readout velocity/flux claim.

## Numerical integration and cost discriminator

Generate one NumPy GL20 tensor rule, shared by both methods. Evaluate every original
simplex once and every nonzero retained anchor cone once. Exact integer determinants
are converted to binary64, never recomputed through floating LU. Save every returned
integral, determinant, weight, q range and contribution before parity/range checks.
Accumulate by math.fsum. No adaptive order or retry. Difference of the two values
is descriptive; no tolerance waiver and no source floating-point certificate.

Existing real-arithmetic rule bound is |det|*(12n+9)/(6*pi²*3^(n-1)). Sum ABSOLUTE
coefficient-weighted determinants for the signed rule. With pi>3, use the rational
upper TV*249/(54*3^19). TV is upper-bounded by per-term dyadic ceiling at2^-128;
the signed determinant total is bracketed by floor/ceiling. This avoids an enormous
common denominator, and the ceiling work is charged. Exact signed totals must have
intersecting brackets. The bound excludes source node/weight/normalization/kernel
and accumulation errors. Historical T72 certified mass is not transferred to this
new floating evaluation. Counts and CPU are a decision test, not full native cost.

Record original/simplex, unique/cancelled/live facets, zero-anchor facets, integration
terms, coefficient multiplicities, determinant TV, fixed8000-node work, all exact
construction counters and observed integer sizes, rule/setup/chain/integration CPU.
The named node arithmetic is 16 multiply+12 add matmul, 4 square+3 sum norm,
1 square+1 reciprocal,1 weighted multiply+1 reduction-add allowance =39 categories
per node; it is not packed instruction/runtime work. Scalar conversion, determinant
setup, library rule construction, sorting/hashing/GCD cost, Python/control/memory,
logs and files are additionally visible through full process CPU and separate exact
counters, not silently free or claimed covered by39. Exact-moment validation work
is in CPU but excluded from the partial counters.

Inherited enumeration and certification are not free: pinned original results save
gate enumeration0.042940864CPU and geometry/exact audit5.465172562CPU, with the
original integer operation ledger. These historical timings are not a matched
speed benchmark. Cold boundary work includes all newly measured chain construction,
matching, rule setup and integration, with inherited discovery separately reported.
Original integrals evaluated here are validation/comparator work; report them too.

Alternative explanations and cheapest discriminators: (1) low cancellation may
reflect nonconforming subdivision, so retain exact facet incidence/multiplicity and
the nonconforming control rather than declare all boundary methods dead; (2) raw
facet savings may be offset by signed TV or setup, so compare weighted bound, fixed
node count and full observed construction/integration CPU; (3) finite numerical
agreement may conceal floating error, so preserve real-rule-only qualifications
and the inherited independent enclosure without upgrading this calculation.

## Resources, pins, failures

Publish protocol, all direct input/source hashes and completed independent reviewer
artifact, then immutable gate.json before one run. Required review pins are primitive,
fixtures, driver, this document and protocol; verify all protocol/gate/reviewer pins
separately. Explicit hexadecimal review/publication citations are required. Runtime
invoked executable, resolved binary, NumPy version/path and binary sources are pinned.
All inputs finite exact integers; reject unexpected values, never clamp or repair.

One primary lock, one BLAS/OpenMP thread, soft AND hard CPU110s, address-space2GiB,
100s internal CPU, timeout120s AFTER acquiring the lock. Hard stop may leave an
incomplete last phase, whose attempt events remain. No automatic resume or rerun.
All raw outputs are fresh; returned chains are saved atomically before cap checks,
mass rows are flushed immediately, and fatal records retain complete/pass_all=false.
Top-level imports/last serialization extend beyond internal timers; OS cap is whole
process. Counts do not pretend to price arbitrary integer bit complexity.

Command (after published gate only):

```
bash review/facet_cancellation/launch.sh
```

No final estimator/scaling/HP1 claim. The user-facing decision is the actual weighted
cost comparison for this fixed H4T4 chain; H4T3 and broader boundary constructions
remain separate, explicitly untried work.
