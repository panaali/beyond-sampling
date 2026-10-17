# Fixed spectral child attribution

The first12-call spectral pass established parent dispatch and unchanged results but left scale/helper/BLAS children opaque. One separately frozen observer pass of exactly those same12 saved calls records the named children and original DSTERF D/E arrays (N<=16), with the same bitwise output gate. No source numerical substitution, model/mass/MC call, order change or larger panel. New costs are not inferred until source review.

Extend the reviewed transparent wrappers with DLASCL, DLAPY2, DLAE2, DLASRT, DLANST, DLARF1F, GEMV, AXPY, GER, SCAL, GEMM and NRM2. Every wrapper calls the original linked symbol and records only after return; DSTERF additionally copies its initial vectors. Symbol resolution stays in the explicitly pinned library. Parent/child nesting must not be double-counted as independent work. Captured absence of scale calls is a finite-input observation only. Event buffer overflow and N>16 reject; original captures remain untouched.

Build `gcc -shared -fPIC -O2 spy.c -ldl -o spy.so`; run capture.py with absolute LD_PRELOAD under the review lock, one thread,2GB/20CPU. Build/setup and observer work remain research overhead. Pins and output include source/binary hashes; raw failures preserved. Source gate required before execution.
