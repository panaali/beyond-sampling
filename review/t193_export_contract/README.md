# T215 T5 simplex export acceptance prerequisite

## Question
What must the missing T192 arrays establish before lane34 can compare independent masses on their stored rays?

## Method
Source-only inspection of fixed T192 branch a5ab87572c0c633a10894bc822d9ff4b3c7d9593. manifest.json preserves ten original text files and Git blob/SHA256 identities; SOURCE.md traces the exporter, and ACCEPTANCE.md/schema.json specify data versus provenance prerequisites. No original code is imported. T55's distinct latest-evidence inventory d2c97d34/sharedf7d5e15e records293 tracked paths and zero NPZ/NPY artifacts.

## Status
Source/provenance review passes independently573a8cae/shared2d7dbb91. T193 numerical work remains blocked. Expected exports are results/simplices/H8T5_zoo0.npz and H8T5_zoo2.npz, neither supplied in the pinned tree. This is not authorization to run the exporter or enumerate new geometry.

## Results
[inferred, independently reviewed] Row-ray orientation and selected-index rule can be traced. However the NPZ schema alone omits model/source/runtime/rule bindings and local emission identifiers. An inherited T192_SLIVER value also changes the semantics of masses_each: with multiple thin rows, aggregate mass is placed in only the first thin output. This is a conditional source defect, not evidence that an actual export was generated incorrectly. See SOURCE.md.

## Compute accounting
Read/hash ten text blobs and one incidental m43_weights binary blob from Git; no binary decoding, source imports, model/geometry/mass/NumPy/Torch/Numba/RNG calls. No numerical mass or bound evaluations. No estimator cost claim.

## Claims
[inferred] Script availability does not establish data availability, independent-reference status or a numerical certificate. [inferred] A deterministic stride sample supports comparisons on those selected stored rays only. It does not certify unsampled cones or total model accuracy.

## Alternatives not ruled out
The actual exporter may have run with all intended environment defaults and complete external logs. Existing artifacts can be accompanied by a provenance sidecar without another capture. Missing metadata is unknown, not proof that values are wrong. Sliver refinement and higher-order rules may produce accurate values despite lacking rigorous error certificates.

## Failure modes
Importing estP executes geometry compilation/patching and incidental table loading; do not import it for a read-only inventory. Do not reinterpret rows as columns, renormalize stored rays, use the same-source refined mass as an independent truth, or silently extrapolate selected discrepancies to all simplices. Original T192 history is on its public author branch, not yet ancestor of the shared branch; pinned owned snapshots do not assert otherwise.

Availability update after this fixed-source review: T55 reports newer T192branch2930bc92 now tracks H8T5_zoo0.npz (1770680bytes) and H8T5_zoo2.npz (1704578bytes). The absence finding applies only to a5ab8757. No arrays decoded or source/provenance contract satisfied by this report alone; no numerical unblock, and new DRI T219 priority remains.
