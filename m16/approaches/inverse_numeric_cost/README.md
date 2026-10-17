# T157 inverse numerical branch categories

## Question
Which numerical absolute-value, maximum/comparison and zero/subnormal guards remain outside T150's188 floating-expression envelope? TASKS-only claim8877acc was pushed before implementation. This is a separate category audit; h4 owns SVD,56 owns actual determinant, and the original inverse implementation remains unchanged.

## Method / frozen protocol
Pin T150 installed binary and saved disassemblies. Inspect the actual idamax short n1..4, inc1 path, LU zero/subnormal pivot tests, dscal zero tests and triangular driver beta tests. Charge all executed packed comparison/absolute-value lanes, including duplicated or inactive upper lanes, conservatively. FMA remains2, division1 as in T150. Exclude integer/control/index, register shuffles, pointer comparisons, copy/allocation/interpreter. This narrower numerical category total cannot close the full-cost gate alone.

First controls: independently emulate the assembly short idamax path and compare direct installed kernel for every vector of length1..4 over fixed alphabet[-2,-0,+0,1,+inf,-inf,NaN,2^-1074] (4680 vectors). IEEE MAX second-operand and unordered COMISD behavior must be preserved; do not assume NumPy argmax semantics for NaNs. Save first failure and stop before composition on mismatch. Then transparent same-process passthrough on fixed identity, row-pivot, zero, duplicate-row, subnormal-diagonal, NaN-diagonal, infinity-diagonal and finite-overflow controls; outputs/exceptions must match uninstrumented baseline. These are analytic controls, not trained-model evaluations.

Provisional derivation before tests: idamax abs lane upper totals32 and comparison/max lanes44 for lengths4,3,2,1; four LU guards ≤8 abs lanes+8 numeric comparisons; up to3 dscal calls ≤2 zero comparisons each; each triangular driver ≤2 beta comparisons. Thus ≤102 added numerical units and ≤290 combined FP+numerical guard units. Source/binary review and controls can reject or refine this proposal; it is not yet a result. Numeric NaN handling may terminate comparison scans early, but pointer clamp/failure behavior is separately retained.

Alternatives and cheap discriminators: a one-pass textbook idamax would undercount this installed two-pass vector implementation (compare actual short disassembly); NaN MAX/COMISD semantics differ from a mathematical max/argmax (fixed exhaustive alphabet); binary padding or scale guards may add numeric work (scan every reached kernel disassembly and inspect source constants).

## Status
Running; no completed category bound or challenge result asserted.

## Results
Pending fixed controls and independent review.

## Compute accounting
One thread,2GB, <=20CPU review lock. Count setup and fixture CPU separately; no mass or geometry replay. Native integer/memory categories remain unresolved, and source-expression conventions are not hardware cycles.

## Claims
[measured] T150 pins installed NumPy/OpenBLAS and short call shapes. [inferred, pending review]102 additional numeric units may bound the omitted categories. [speculative] A complete implementation count may remain below the matched-sampling threshold.

## Alternatives not ruled out / failure modes
Other backends, shapes, BLAS block sizes and thread counts require new analysis. NaNs/subnormal/overflow fixtures diagnose branch semantics, not inverse accuracy. Failures remain evidence of this finite implementation/test, not proof of a broad mathematical negative. A complete integer/control bound and runtime convention are still open.

## Bounded results

All4680 fixed IAMAX vectors match the actual pinned kernel, including signed zero, infinities, quiet NaNs and the smallest positive subnormal. Observed emulator lane maxima equal the frozen bounds. Eight fixed inverse controls preserve baseline outputs/exceptions under transparent own-process kernel capture, with IAMAX shapes4,3,2,1 and no GEMM. Controls CPU0.066751; saved-data count composition0.000639CPU. No trained-model input was used.

The conservative added numerical subset is102 units; with T150 arithmetic the inverse numeric envelope is290. Exact/recipe inverse contributions771,372,740 /620,175,440; visible-plus-numeric subtotals1,324,773,508 /1,076,152,155. These are not full implementation cost bounds. `DERIVATION.md` and `summary.json` preserve category/source pins and explicit exclusions. Independent T55 source/count algebra agrees; final saved-control audit pending.

Reproduce controls (if needed): `flock /workspace/work/codex-review.lock bash -c 'ulimit -v 2500000; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/inverse_numeric_cost/controls.py'`. Existing reviewed T150 `kernel_spy.so` must be built from its saved C source first; no installed file is changed. Recount only: `/workspace/clean-neurons/.venv/bin/python m16/approaches/inverse_numeric_cost/summarize.py`.

[measured] Fixed controls and passthrough parity. [inferred, source algebra independently checked]290 numerical-lane upper envelope under the stated convention. Signaling-NaN trap semantics, integer/control, memory and IEEE side effects remain outside scope. No broad negative or full-cost/MSE claim follows.

Final independent T55 review **a24db934** reproduces all4680 saved IAMAX entries with a separate scalar-register transcription, audits the eight inverse event paths and verifies290 numerical-subset composition. Bounded task complete; integer/control/memory and full-cost gate remain open.
