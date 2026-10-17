# Source-only applicability and conservative numerical replacement

TASKS scope2d027d5f precedes implementation. No model, geometry, native kernel or quadrature evaluations. This phase is gated on independent review of the90 controls, the all-stride numeric upper, this source argument and the replace-once contraction.

## Stored model and arrays

The pinned noise_identity.json declares the104 parameters in Wi8,Wh8x8,Wo4x8 order as little-endian float64. Each serialized leaf is a finite Python float. Both original component runners and the cell-rescue runner use np.array on those lists; default NumPy inference produces base float64 ndarrays. A separate struct.pack reconstruction validates the stored model digest without a forward pass or NumPy import.

The model-independent family starts with default float64 np.eye(4); scalar arithmetic preserves float64. The full SVD basis has float64 shape4x3; the three-dimensional lune matrix and its inverse are float64. Dividing each inverse column by a scalar norm produces a float64 three-vector. This is the initial polygon-vertex invariant.

Inductively, clip either retains an existing three-vector or constructs p=abs(sj)*vi+abs(si)*vj and normalizes it, preserving float64 shape3. Subdivision retains vertices or normalizes pairwise sums. Means of such polygon lists have float64 shape3. Matrix rows_at starts from float64 zeros and float64 model arrays; the Boolean gates multiply those arrays without changing floating dtype. Its R/L arrays projected through the4x3 basis yield float64 three-vector rows. Clamp functions and differences/stacking of these rows preserve float64 and width3. Thus every clip normal, its interpolated p, midpoint ab/bc/ca, cell representative w/w0 and poly_int vertex/cross-product operand has the required base-array float64 shape3. The144-cell rescue explicitly checks every preparation vertex dtype/shape and reconstructs each stored cell with dtype=np.float64, so it preserves the same invariant. No claim that arbitrary user-provided model dtypes satisfy this contract.

The script checks the exact eight norm call sites, explicit poly_int np.dot call and clip n@v site against pinned ASTs. The existing source tracer retains numerical values; its independent parity and saved ledger reviews remain prerequisites. This argument establishes dtype and dimension from sources, not a new per-call native trace. Historical installed-library binding remains conditional on the pinned captured installation; it is not supplied by shape evidence.

## Numeric upper independent of byte strides

For default norm(x), real float64 shape3, the pinned Python implementation calls x.ravel(order='K'), then dot(x,x) and sqrt. The flattened three-vector is contiguous; the selected DOUBLE_dot/CBLAS/dot_compute path has three scalar FMAs (6 named arithmetic units), then the NumPy accumulator addition (1), plus one named sqrt. Upper8 is independent of the original layout, with possible copying/control unpriced.

For a valid float64 length3 dot, DOUBLE_dot checks positive strides divisible by8. If both are unit strides, three scalar FMAs plus the NumPy add cost7. Any other positive integral strides reach the strided n<4 tail: three FMAs plus accumulator merge plus NumPy add cost8. A nonpositive or nonmultiple8 stride takes three scalar products and three scalar additions, cost6. This finite source/binary argument bounds numerical work by8 without assuming historical contiguity. NumPy input copying can choose a cheaper branch but remains excluded as memory/control. An invalid out-of-bounds array is outside the valid-input contract.

Scalar-output float64 matmul with core dimensions1x3 by3x1 routes to DOUBLE_dot in pinned upstream source and installed parent at11a63f. Positive unit/stride2 events and reversed scalar fallback are independently tested in the90-control phase. The same upper8 applies to clip n@v once its shape/dtype invariant is accepted. This is not a general matrix/GEMV bound.

FMA counts as multiply+add. Bitmask initialization, indexing, branches, conversions/copy/allocation, dispatch/runtime, floating status and the implementation of the named sqrt outside the retained convention remain excluded. Values including zero, infinity and NaN do not alter the recorded scalar arithmetic loops; this is not a finiteness or numerical-accuracy guarantee.

## Replacement only

Contract only the eight norm sites (old6/call→upper8), the one explicit np.dot site (old5→upper8), and clip n@v (old5→upper8). Require each per-site multiply/add/sqrt count identity and reconcile counts with reviewed call census. Keep every other source, native and cold category unchanged. No sampling-ratio or full-native/hardware claim follows. If any source/path premise fails, retain the original ledger and report the proposed replacement unresolved.
