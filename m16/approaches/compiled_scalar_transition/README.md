# T66: native scalar Gaussian transition kernel

## Question
Can a native implementation reduce repeated scalar Gaussian interval cost
without changing the already diagnosed finite-grid transport approximation?

## Method
Port only the four bilinear interpolation basis integrals on each precomputed
scalar interval. Keep centered quadratic moments, stable tail CDF/expm1,
identical narrow/invalid-moment/value screens and positive GL12 fallback.
Reuse StagePieces geometry, complete/partial interval allocation, sparse
assembly, q-panel rule and original recurrence. A private function-global
adapter substitutes the primitive only in StagePieces; no canonical edits.
The native entry accepts four basis weights, not arbitrary continuations.

Build local C++ shared library without fast-math. All GL12 nodes/weights are
constructed and charged at runtime. Account build wall/child CPU separately
from each cold estimate's load/setup; report the cold first-use sum, not only
kernel time. Preserve named opaque erfc/exp/expm1 calls and partial arithmetic
counts; native speed alone is not a portable FLOP or sampling win.

## Frozen alternatives and falsifiers
1. Python vector/temporary overhead dominates: native same-math kernel should
   reduce primitive and end-to-end CPU while preserving probabilities.
2. Sparse assembly/outer loop dominates: primitive gain may not matter; retain
   full benchmark and cold compilation penalty, no broader port on hope.
3. CDF/rounding differences alter safeguards or probabilities: adversarial
   center/width cases, independent high-precision positive integration,
   fixed cached-kernel parity and row-mass/nonnegativity must pass first.
4. Apparent gain hides omitted setup or changes accuracy: same h9 H2T4
   complete q-panel GL2 only baseline/native, full new process setup and
   source hashes; constants/duplicates/scale controls. No h/noise/rule sweep.

Primitive gate: absolute native/Python disagreement<=3e-12, independent
positive integral check<=3e-12, no probability-guard failures. Include
centers0,+-1,+-5, tiny widths1e-10/1e-6, ordinary .1/1 widths within[-6,6],
and normalized bilinear fractions/slopes of both signs. Record fallback
classification differences rather than asserting identical floating paths.
Then fixed trained StagePieces matrices at signed/central/tail target values,
constant/tie and1e-16 positive scaling. Full prototype benchmark only if all
pass; no attempt to improve spatial accuracy or reinterpret T67 negatives.

## Status
Claim63bf600 pushed. Protocol frozen before implementation/measurements.
T48 confirms no duplicate native Gaussian kernel in completed DD/T104 lanes.

## Results
Pending. Classification is deterministic numerical transport of a fixed
interpolant; spatial and floating errors are not certified.

## Compute accounting
One thread,2GB,<=110CPU cycles; project locks. Count compilation, library
load, rule construction, geometry/arrays, all kernel moments/fallback nodes,
sparse construction and recurrence. Retain invalid-input/fallback failures.
No Monte Carlo/training, cached advice, or model-specific parameter tuning.

## Claims
[inferred] Same real-arithmetic basis integrals preserve the interpolant.
[speculative] Native evaluation helps total runtime after build/setup costs.

## Alternatives not ruled out
Streaming sparse assembly, prefix-sum operator reuse, other native kernels,
and exact boundary representations. No total-error or total-cost certificate.

## Failure modes
Platform erfc rounding, narrow tail cancellation, coefficient ordering,
overflow/nonfinite inputs, changed fallback thresholds, sparse guard failure,
and benchmark warm-cache/cold-build mismatch.

## Control checkpoint
Build via `python -c 'from native import build; build()'` in this folder,
then `controls.py` and `full_controls.py`, one thread/review lock. Compiler
child CPU .189867 and wall .190244 seconds, source/command in build.json.
Forty signed/tail/narrow interval cases agree with Python within6.94e-18;
160 independent60-digit positive integrals agree within4.97e-17. Floating
errors can exceed the tiny quadrature-only bound and are explicitly counted.
Fifteen trained stage/target sparse-matrix comparisons agree within1.13e-13;
row guards pass. Invalid zero-width/NaN inputs reject explicitly.
A single40,000-interval batch costs .043900CPU Python versus .007993native.
Primitive/control cycle1.191CPU; no end-to-end inference from this ratio.
Full h3 constant/duplicate ties and1e-16 scaling pass, with base/scale parity
exact to displayed double precision; full-control cycle .859CPU. Each native
estimate constructs its own rule/library setup and baseline clears GL12.

## Frozen full-estimator pilot
`pilot.py` runs the unchanged h9 H2T4 complete q-panel GL2 estimator once per
backend. Both return .3077232488426774 (displayed double results identical).
Runtime ratio Python/native is1.412; including compiler child CPU in native
first use gives1.363. Combined estimator cycle12.780CPU seconds. The measured
5.49x primitive gain shrinks to1.41x end-to-end because geometry, sparse
assembly, recurrence and target work remain. Exact raw times/counts and
source/checkpoint hashes are in pilot.json; no timing extrapolation.

The estimate retains all T67 caveats: its small total error on this base is
known between-label cancellation, not component convergence; no accuracy
improvement, noisy-model win, portable FLOP win or full-error certification
is claimed. Compiler/setup costs and named opaque library calls are explicit.
The finite same-rule CPU benefit is a scoped implementation result awaiting
independent source review. No larger mesh or additional model sweep follows.
