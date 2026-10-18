# T267 H4T3 combined source/theorem/resource gate

## Question
Is one new H4T3 exact capture followed by oriented-chain cancellation and closed trihedral integration ready for the bounded second-rung comparison?

## Method
Read angle.py, run.py, PROTOCOL.md, protocol.json, launch.sh, the unchanged parent reducer, and the generic exact.py capture implementation. Verify source/dependency hashes, prior independent control-review status and every input hash it binds. Do not decode weights, call geometry, import producer modules or replay controls during review.

For L1-normalized same-orthant nondegenerate rays a,b,c, all pairwise dots are nonnegative and z=|a||b||c|+(a.b)|c|+(b.c)|a|+(c.a)|b| is positive. The trihedral solid angle is2*atan2(|det|,z); dividing by4*pi gives the implemented probability atan2(|det|,z)/(2*pi). Positive determinant magnitude is obtained exactly before binary64 conversion. Only positive finite arguments are accepted, so there is no branch/wrap ambiguity in this domain. Multiplying by the reviewed signed anchor coefficient gives the integration-chain contribution. Degenerate anchor determinants are omitted exactly by the inherited reducer, not repaired by the floating kernel.

The six source fixtures have exact probabilities1/8 for positive/negative/permuted/scaled octants and1/16 for each symmetric bisecting triangle. Their fixed1e-14 binary64 diagnostic tolerance is not an outward certificate. Positive rescaling is removed by L1 normalization, and permutation leaves the absolute determinant formula unchanged.

The exact engine forms two initial half-lines, lifts along a neuron with nonzero input coefficient, and clips remaining affine gates with integer/Fraction arithmetic. For H4T3 its gate constraint identifiers do not collide with output/rank/orthant identifiers. All-OFF/identically-zero preactivations yield the same zero affine contribution regardless of internal active label. Output constant-margin ties reject higher-index candidates, while ordinary input/output equality boundaries are null under Gaussian measure. Correctness requires exactly one larger input, implemented by the rank count in outcomes(). Pulling triangulation uses represented face incidence and exact ranks/determinants. Per-ray certificates establish feasibility, tight incidences, extremality and full dimension, not by themselves completeness or disjointness. Those global properties remain inherited algorithm premises; gate determinant total8 is a consistency check, not a substitute proof.

## Status
The final source_gate.json binds75f1f361/shared96de1ba7, including the narrow dtype repair and aggregate serialization change. Execution is allowed only after that review and a pinned immutable producer gate are published. This is a new geometry capture, not reuse of a nonexistent terminal H4T3 archive.

## Results
[inferred] The solid-angle formula, signed orientation, fixed controls and generic H4T3 call paths are consistent. Old eight chain controls and six guards are reused through independent7fe3ac0b and its exact input hashes, without replay. The historical model identity is only the pinned m43_weights.npz source claim; no current checkpoint or historical runtime identity is established here.

Initial1d8b035d accepted any floating dtype although dyadic_model converts through Python binary64. The requested narrow repair limits the accepted item width to at most8bytes, ensuring float16/32/64 stored values embed exactly before Fraction conversion. Nonfinite values and wrong shapes are already rejected. The same pre-run repair keeps the correctness determinant sum exact for its assertion but serializes its exact floor/ceiling at2^-128 and integer bit lengths; every individual determinant remains saved. This avoids a large decimal aggregate without weakening the predicate.

## Compute accounting
Source/metadata only for this review. Producer limits are17 internal CPU seconds,20/20soft-hardCPU,2GiB,25wall after review lock,one numerical thread; exact engine additionally has12CPU from creation,4096-bit observed sites and2000-ray cap. All capture, certification, classification, orthant splitting, triangulation, record construction and integration are charged. Per-call45 kernel categories and nine rational coordinate conversions are partial conventions, not full native cost. Full CPU also contains prior-hash checks, imports, six controls and baseline validation; some final serialization lies outside internal timers but inside OS limits.

## Claims
[inferred] Gate covers exactly six fresh angle controls, one fixed H4T3 weight load/new capture, unchanged chain reduction, and baseline/reduced closed-form evaluations. Successful numerical completion does not certify floating accuracy, inherited global geometry, historical model identity or a speedup. Separate saved audit is required.

## Alternatives not ruled out
Different subdivisions or anchors may cancel additional facets. Construction costs may dominate saved integrals. H4T3 and H4T4 differences can be dimension/model specific. No noisy-model, H16T10, sealedHP1 or broad boundary-method result transfers from this phase.

## Failure modes
Fresh output/log guards and atomic snapshots preserve returned gate lists, each returned correctness-cone list, current capture state and each successful mass return. Internal capture work that fails before returning is represented by its attempt and partial work ledger, not a complete cone archive. Deadline/cap failure is retained without automatic extension or retry. Finite libm/norm/determinant conversions are not directed rounding, and matching determinant sums do not prove completeness.
