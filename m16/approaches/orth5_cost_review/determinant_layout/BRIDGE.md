# Conditional category bridge, not a complete cost

The frozen controls establish the exact logical matrices, dimension/lda and installed GETRF entry/dispatch premises for the two batch strides used by the five-dimensional source. They include singular and nontrivial-pivot paths. NumPy's pinned source/disassembly copies each matrix before factorization, so the original stride is not an argument to the subsequent GETRF body. Historical T185 execution binding remains unrecovered; this applies to the pinned current implementation/premises, not automatically to an old runtime.

Under the inherited reviewed small-path and numeric-lane assumptions (determinant3 review26fd4e11, determinant4 reviewf53ea74a), the per-call named envelopes are104 and176. These include elementary-function invocations charged as one named call, not libm internals. For clarity split them before composing:

| Dimension | Arithmetic, absolute, sign and comparison category upper | Named log/exp calls upper | Original combined named upper |
|---|---:|---:|---:|
|3|100|4|104|
|4|171|5|176|

Dimension3 has24 LU arithmetic +63 abs/comparison +4 reconstruction add/multiply +6 sign negations +3 sign comparisons=100, then at most3 logs and1 exp. Dimension4 has56 LU arithmetic +40 abs +58 comparison +5 reconstruction add/multiply +8 sign negations +4 sign comparisons=171, then at most4 logs and1 exp. Singular cases can skip work. Binary sign/abs operations and inactive lanes use the existing declared convention; they are not exclusively physical floating arithmetic instructions.

For the default completed log branch,3,840 determinant3 and665 determinant4 instances therefore give a conditional category envelope497,715 plus at most14,180 log calls and4,505 exp calls (18,685 named calls). Their historical combined named sum is516,400. Do not report this as a full implementation upper/lower bound or infer speed/MSE from it. The exposed source expressions separately total32,262 output-element arithmetic/comparison units; adding them to installed-lane envelopes creates a mixed-category hybrid, not a newly validated common scalar cost, so this phase keeps them separate.

Unpriced work includes the single determinant5,5x5 and1296x5 matrix products, length5 norms/dots, reductions, array powers/sqrt/atan2 and other special-function bodies, setup, allocation/copying, integer/control, geometry and the sampling implementation. Thin-branch constituent prices remain open. No corpus branch frequencies, native-price multiplier or full comparison is inferred. Exact source call counts and a conditional upper for selected categories are different claims.
