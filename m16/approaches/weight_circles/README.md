# T191: weight-derived circles

## Question
Can network structure justify selected great circles and their measure,
rather than merely rotate a fixed quadrature grid?

## Method
Claim6d1f9cc9 precedes implementation. `THEORY.md` distinguishes orthogonal
measure invariance from a sufficient exact rank2 correctness reduction.
The identity must hold on the selected circle outside an angular-null set;
ambient-almost-everywhere equality alone is insufficient. The correction is
preserved in4f307b4c and `SOURCE_CORRECTION.md`.

`CONTROL_PROTOCOL.md` freezes14 wedge controls,2 exact-collinearity rejections,
180 deterministic integer-input recurrent state/prediction checks, and16
circle cases with plain/traced parity. `controls_gate.json` pins the independent
source gate8dfaf3de and27 input/dependency files. The original Hopf source is
used with only the prior reviewed tile-mask correction; no shared edits.

## Status
Bounded analytic phase complete21e3f9dc, independently reviewed3b9aa608.
No trained estimator, model-reference draw or new geometry capture has run.
No trained accuracy or scalable mechanistic reduction is established.

## Results
All frozen controls pass. Maximum wedge error5.56e-17. At four fixed phases,
H4T3 difference memory has circle accuracy1/3 (maximum discrepancy2.22e-16),
H10T3 median memory has accuracy1, and its cyclic readout has accuracy0.
The T4 zero-readout example has circle accuracy1/2 versus Gaussian accuracy1/4:
prediction-only projection reasoning fails because it neglects target rank.
This is the designed exact insufficiency witness, not a failed circle method.
All16 plain/traced values, piece counts and legacy counters agree exactly.

Reproduction of the immutable one-shot run (refuses overwrite):
```
ulimit -v 2500000
ulimit -t 20
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock timeout 25 /workspace/clean-neurons/.venv/bin/python m16/approaches/weight_circles/controls.py
```
Use saved `controls.json` for review; no repeated estimator calls are needed.

## Compute accounting
Recorded0.520463486 process CPU seconds;32 fibre calls and180 deterministic
forward checks,14 successful frames and2 exact-collinear rejections. Zero
trained calls/reference draws. The16 observed fibre ledgers total306396 named
numeric expressions plus16 startup expressions (306412 total); integer-control category0, with100 sort/search/native-opaque
and memory records retained separately. Plain calls, control checks, startup,
frame construction and discovery are separate from that observed subtotal.
A successful explicit4D frame kernel has39 arithmetic/sqrt/division expressions
plus2 positive-norm comparisons, excluding input validation and control work.
These source counts are not complete binary/runtime cost bounds.

## Claims
[inferred, independently source/algebra reviewed8dfaf3de] An appropriate
rank2 representation of the entire correctness predicate gives an exact
circle law; an orthogonal rotation alone leaves an outer base integral.
[measured, independently saved-reviewed3b9aa608] The finite controls satisfy their frozen
expectations/parity checks. No generic fibre correctness follows.
[speculative] A weight-derived certificate or conditional decomposition may
reduce the trained network's integration dimension or branching.

## Alternatives not ruled out
Weight-derived orientation may improve finite quadrature without providing a
mechanistic reduction. Conditional event decompositions and error bounds may
work when a global rank2 condition fails. The rank-wall spanT-1 is only a limit
of the sufficient all-walls test, not an impossibility theorem. The exact
H10T3 synthetic median model is a control, not a trained ladder result.

## Failure modes
Dropping rank walls; invalid measure changes; numerical rank thresholds passed
off as identities; incomplete first-index ties; ambient-null exceptions on the
chosen circle; the known mixed-zero root mask; hidden discovery/branch costs.
Frame controls only reject two exactly collinear fixtures and do not establish
robust rank detection near degeneracy. No claim about every floating input.

Independent audit checks all saved model/output/pin/site records and einsum shapes.
Hidden state vectors were not serialized; hidden-state equality therefore rests
on source algebra and author checks, not independent raw-state reconstruction.
