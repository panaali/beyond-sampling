# T180 fibre cold numerical closure

## Question
Can the remaining model-independent native numerical setup be bounded under the same numeric-category convention as the reviewed fibre census?

## Method
Claim c67d6544 precedes this folder. Pins preserve the released T180 source and installed NumPy/OpenBLAS. First capture only cold imports: six 1x4 full-vector SVDs, 48 3x3 inverses, six Legendre rules of orders 6,3,3,6,10,16. No model, mass, estimator, geometry clipping or Monte Carlo functions run. The unrelated m43 import is replaced by a stub whose clip raises; np.load is forbidden. This avoids unneeded stored-weight I/O while preserving numerical function statements and original source-order assignments. Reuse the same25 target names from the prior census, then compare newly reconstructed cold objects between an unwrapped baseline and transparent capture (prior serialized values were not retained); preserve all inputs, outputs, shapes and source hashes. Neither pass is an estimator timing claim.

Then inspect installed dispatch and derive narrowly scoped numerical envelopes. Existing 4x4 inverse and values-only SVD prices cannot transfer without shape/path verification. Coordinate the separate T156 rank and T11579 matmul owners. Cold replay is independently checked; no existing costly trained panel is repeated.

## Status
Frozen capture source awaiting independent gate. No numerical result yet.

## Results
Pending. Capture command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/fibre_cold_native/capture.py` under the review lock, 2GB cap and 20CPU limit.

## Compute accounting
Baseline and instrumented cold setup are research validation overhead, reported separately. Downstream use charges the setup once. Preserve the prior 1,838,399,658 visible numerical upper subtotal and component12 timeout. Remaining imports, interpreter, integer/control, memory/copy/allocation, library and floating exception costs stay explicit; this is not hardware work or full implementation cost.

## Claims
[measured] Prior T180 independently reviewed all48 components and cold call census. [speculative] The few remaining cold kernels may be small relative to the one-case numerical headroom. No full cost, rung pass or M16T10 solution inferred.

## Alternatives not ruled out
Hidden native branch work can invalidate a textbook count: capture dispatch before pricing. Import substitution can alter cold geometry: compare all assigned objects and fail if the clip stub is called. Iterative eigen/SVD paths may require actual finite iteration traces rather than a universal bound. Uncovered paths remain explicitly unresolved.

## Failure modes
Source or binary drift, unobserved calls, changed output bytes, unexpected model I/O, unsupported kernel paths or bounded CPU exhaustion stop the phase and are preserved. No target/cap relaxation or source replacement.
