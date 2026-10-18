# FullSVD internal BLAS discriminator

The parent helper trace identifies tiny public GEMV/AXPY/GER and GEMM(1,4,1), but does not establish their installed child paths. A separately frozen six-SVD-only table observer resolves that gap before pricing. Same saved inputs/default arguments/outputhex, no quadrature/model/mass calls. This is a call-shape discriminator, not a repeat outcome experiment or nativeprice itself.

Reuse the reviewed T11579 sixteen-entry process-local table wrapper and runtime target guards, extending only GEMV_N/GEMV_T/AXPY slots with their existing OpenBLAS ABIs. GEMV matches prior T150 adapter; AXPY uses the same ten-argument real vector-kernel ABI as SCAL. The wrapper logs true native strides/alpha, delegates unchanged, restores all19slots in finally and rejects event overflow. Parent/child nesting remains explicit. Build from pinnedsource; record binaryhash. One thread,2GB,20CPU reviewlock. Independent ABI/source gate before execution.
