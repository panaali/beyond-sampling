# Installed3x3 inverse numeric-category candidate

This is a source-derived candidate pending independent review. All48 saved cold matrices plus six analytic fixtures preserve baseline outputs/exceptions. Runtime table address,14 kernel targets, blocking parameters and one thread match reviewed T150 before any process-local wrapper installation. The wrapper is rebuilt from saved source and its binary hash retained; restoration is finally-protected.

The observed factorization uses IAMAX lengths3,2,1; GEMV(3,0),(2,1),(1,2); one strided DOT length1; SCAL lengths2,1. The independently reviewed T11579 determinant LU derivation applies to precisely these child shapes:24 floating expressions plus63 abs/comparison lanes. It does not supply the inverse solves.

The three-row solve splits into blocks2 and1, unlike the old four-row solve. Each block's scalar triangular solve has m diagonal multiplies and m(m-1) update operations per RHS, including multiplications by the unit-lower diagonal. Thus blocks2+1 cost5 per RHS,15 per lower or upper solve,30 total. Cross-block updates are separate packed GEMM calls, not included in30. Upper diagonal packing adds3 reciprocals. Copy/packing assignments and pivot swaps remain memory work, outside this numeric convention.

The four packed GEMM children are (m,n,k)=(1,2,2),(1,1,2),(2,2,1),(2,1,1), each once per inverse. Their actual SkylakeX tiny loops have mn floating lanes per K step, two operations per FMA, followed by mn lanes of alpha-times-accumulator-plus-C. Count2mn(k+1), respectively12,6,16,8:42 total. There are no numerical alpha/beta comparisons inside this packed body; dimension branches are integer/control.

Installed disassembly witnesses (same binary SHA432e1c17...):

* N2/M2: K-loop two2-lane FMAs a456e9/a456ee, final two2-lane FMAs a45707/a45714.
* N2/M1: K-loop one2-lane FMA a4574a, final one2-lane FMA a45768.
* N1/M2: K-loop one2-lane FMA a44942, final one2-lane FMA a44950.
* N1/M1: scalar K-loop FMA a44986, final scalar FMA a44994.

The N2 path skips M16/M8/M4 blocks; N1 similarly skips larger M blocks. Each source COMPUTE_SIMPLE loop uses the observed K count. Source macros and disassembly are pinned in inverse_price_pins.json; the upstream source is supporting evidence, installed binary structure supplies the path-specific correspondence.

Two triangular drivers retain the prior conservative four numeric beta comparisons (actual GESV beta1 skips scaling). Thus floating arithmetic24+30+42+3=99; other numeric lanes63+4=67; **candidate total166 per inverse**, or7968 for48. Integer/control/memory/packing/runtime and error-status handling remain excluded. This is not a full implementation upper bound or hardware cost, and cannot alone justify a matched-budget sampling claim. The bound concerns this fixed installed one-thread3x3 path; fixed controls do not establish arbitrary numerical accuracy.
