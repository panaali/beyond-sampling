# Installed AXPY ABI witness

The original claim by analogy to SCAL was insufficient evidence and was caught before execution. The frozen observer has not run. The pinned installed public wrapper gives a direct witness instead.

At scipy_daxpy_64_ 9eb86..9eba3, single-thread dispatch through gotoblas+0x388 passes RDI=N, RSI=0, RDX=0, XMM0=alpha, RCX=X, R8=incX, R9=Y. The three pushes place incY, zero buffer pointer and zero buffer stride in stack argument positions 7..9 of the integer/pointer register class. This is precisely the observer declaration (N,dummy0,dummy1,double alpha,X,incX,Y,incY,Z,incZ). The scalar double uses its independent SSE register and does not consume an integer argument register.

The selected daxpy_k_SKYLAKEX at a3cca0 reads incY from RBP+0x10 (first stack argument), X from RCX, incX from R8 and Y from R9, with alpha in XMM0. Caller and callee independently agree. The binary and runtime selected slot are already pinned; the disassembly is newly pinned too. This establishes the finite observer ABI, not arithmetic price or portable compatibility.
