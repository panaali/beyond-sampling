# T267 combined chain/source/resource review

## Question
Does the fixed oriented-facet reduction preserve the archived integration chain, and is the single controls-plus-H4T4 comparison ready to execute within its frozen scope?

## Method
Independent written algebra and source inspection of primitive.py, fixtures.py, run.py, PROTOCOL.md, protocol.json and launch.sh. check_metadata.py verifies exact source/dependency hashes and inherited Git blobs without decompressing the scientific archive or importing producer code. The final source_gate.json binds the reviewed bytes.

For each same-orthant simplex, normalize each primitive ray to L1 one. The affine chart is s.x=1, with anchor a=s/d. The oriented chain cone identity gives the sum of anchored facets, with omitted-vertex signs (-1)^i. Multiplying the sorted simplex by sign(det) yields its positive mass. The implementation's coefficient times sign(anchor determinant) is therefore the correct signed coefficient on the unsigned anchored cone. The determinant identity is a useful check but does not replace the chain argument for a nonconstant density. Zero anchor determinants vanish; duplicate simplices retain multiplicity. Only exact common faces in one orthant cancel.

For dimension four, radial Gaussian integration gives |det|/(2*pi^2) times the simplex integral of q^-2. Every point in either normalized simplex lies in one signed L1 section, so 1/4 <= q <= 1. Independently, the Chebyshev coefficients for q^-2 on [1/4,1] have absolute values 2(5+4k)/3^k for k>=1. Truncating after degree18 gives uniform tail 249/3^19. Under the Duffy map, this polynomial times the Jacobian has degrees at most (38,37,36), hence tensor GL20 is exact. Both exact integration and the positive GL rule have Jacobian mass1/6. Their difference is bounded by twice that mass times the uniform tail, yielding |det|*249/(6*pi^2*3^19). Summing absolute coefficient-weighted determinants and using pi>3 gives the stated rational bound. Per-term dyadic ceilings are conservative even for signed coefficients; floor/ceiling signed sums remain outward.

The finite moment checks are determinant-weighted simplex averages: ordinary simplex-coordinate integrals have an additional common 1/(d-1)! factor. This does not affect their equality checks. The nonconforming control retains unmatched internal faces; it does not require geometric face-union matching.

## Status
Source/theorem/resource review passed after repair dbdc40d3/shared598b98c8 and the final metadata check. Execution requires the published completed source_gate.json and producer immutable gate; this README alone authorizes nothing.

## Results
[inferred] Chain cancellation and the ideal GL20 bound are valid under the stated same-orthant and exact-integer premises. No scientific controls or archive integrations were replayed during this review.

Initial b715c19a source had two narrow issues: int(x) could coerce malformed archived coordinates before the strict primitive saw them, and cold_boundary_cpu omitted archive conversion and dyadic accounting despite the prose's all-construction wording. Repair dbdc40d3 rejects noncanonical serialized integers and nonpositive determinant pairs, adds archive-conversion and dyadic-accounting intervals, and renames the cold component sum with explicit exclusions. These changes pass source inspection. The original source remains historical evidence.

## Compute accounting
Metadata hashing, Git-byte checks and source/AST reading only. Zero archive payload decompressions, producer imports, model/geometry/Gaussian/RNG calls. The producer's single process is limited to100 internal CPU seconds,110 soft and hard CPU seconds,2GiB address space and120 wall seconds after the primary lock, with one numerical library thread. Exact determinant counters are partial; bit maxima are observed at named determinant sites, not every integer operation. The39 named operations per quadrature node are a convention, not native instructions or a full cost model. Full process CPU includes validation, serialization and other setup; comparator integrations are disclosed.

## Claims
[inferred] Source clearance covers eight exact controls, six rejecting guards, and then the single pinned5058-simplex archived chain. pass_all denotes source/control/chain/range completion, not a certified floating comparison or success at the ARC task.
[inferred] The predecessor exact_artifact_h4_review establishes artifact consistency/containment within its own scope, not independent completeness or disjointness. This reduction preserves an integration chain regardless; interpreting that chain as total correctness probability still inherits those missing premises.

## Alternatives not ruled out
Nonconforming subdivisions may hide cancelable faces. Different orthant anchors retain artificial interfaces. Setup or signed determinant total variation may erase node savings. More efficient boundary constructions, other anchors, other rungs and actual native cost remain open. H4T3 has no pinned terminal archive in this phase.

## Failure modes
Real-arithmetic quadrature bounds exclude stored floating rays, generated nodes/weights, kernels, determinant conversion and accumulation errors. Numerical agreement is descriptive. This review does not establish the original union's coverage, first-index labels, checkpoint runtime identity, an estimator speedup, or a general failure of cancellation. A cap/failure is retained without automatic retry; all results need the separately assigned independent saved audit.
