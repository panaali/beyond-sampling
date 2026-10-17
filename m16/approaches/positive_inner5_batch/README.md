# T11589 batch primitive parity

## Question
Can the scalar positive-inner arithmetic be batched without altering the fixed primitive results, preparing a bounded dispatch-cost pilot?

## Method
Claim2557fa91, author34 yields. Float64 Nx5 array API, nonempty matching shapes. Finite inputs are scaled and processed by constant/same-sign/cross branches; all dot and wedge reductions retain explicit scalar left-fold ordering. Zero divisors reproduce the scalar arithmetic-failure status. No collinear fallback or tolerance change. Original26 primitive inputs/results are reused, plus4 new invalid-array guards. Supported outputs require identical float bits; all statuses/branches and original exception text retained.

## Status
Frozen30-control phase complete6b902b30 after independent source gate and pre-run coverage repair9c85365a. Independent saved audit34891b4c passes all30/35progress records without candidate replay. Two-case outer pilot claimed072262d8/source4c6506fa passed independent source gate and completed19639b0c; saved audit e99ed577 passes. Full66 split continuationc6dbf588 completed; independent saved audit63f21254 passes all116864 construction/assembly records without replay. No trained run.

## Results
All30 controls pass:16 supported values bit-identical, all26 statuses/branches and inherited exception text match,4 invalid-array guards reject, input bytes unchanged. CPU .004010943 total; candidate timer returned0, so no timing inference. Reproduction command (fresh output location required): `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/positive_inner5_batch/controls.py` with20CPU/2GB limits. Existing outputs refuse overwrite. Any future finite mismatch remains a failed control, not silently waived to a tolerance. API shape handling differs explicitly from original list-only scalar API; it is not an API drop-in replacement.

## Compute accounting
One thread2GB20CPU review. Candidate primitive and all control overhead recorded. Vector masks, indexing/copies/ufunc and reduction dispatch remain unpriced; no source-model FLOP saving or matched runtime speedup asserted. Prior scalar ledgers remain immutable.

## Claims
[measured] All30 frozen finite controls pass, including16 supported bit-parity checks. [inferred] Operation trees and left-fold reductions match scalar real arithmetic. [speculative] Batching may reduce Python dispatch cost; no matched runtime result exists.

## Alternatives not ruled out
NumPy versus scalar sqrt/divide, signed-zero, exception semantics and branching may differ outside these controls. Full outer construction/determinant/accumulation requires its own frozen pilot and current binary pins. No generic or full FP certificate.

## Failure modes
Pairwise sums replacing ordered additions, eager unsupported division, zero-scale underflow, changing branch inequalities, treating negative/nonfinite inputs as valid, or claiming native performance from a source operation count.

## Frozen outer pilot
Original identity and first nonsymmetric split identity child, n6 only:432 newly evaluated inner nodes and2 opaque determinant calls. Every P/D/J/weight construction byte must match independent scalar construction, then sequential assembly/determinant/value/branch counts must match the archived scalar result. This executes2 new mass evaluations, no old scalar inner/reference/geometry/network/MC replay. All masks/copies/dispatch and validation overhead remain explicit unpriced work; no timing gain follows from the prior source-expression model.

Pilot19639b0c: both n6 saved output/determinant/branch bits match; all432 construction P/D/J/weight rows match scalar construction exactly. Total CPU .011992039 including construction validation/archive overhead. Reproduce once in a fresh output directory with `/workspace/clean-neurons/.venv/bin/python m16/approaches/positive_inner5_batch/outer.py` under the same review lock/limits; existing results refuse overwrite. No source-reference or old inner calls.

## Full fixed split results
All66 split rows match saved scalar determinant, final-value bits and branch counts; all116864 split-node construction records match.65 new mass calls/116648 new inner nodes, plus one split pilot reused once; original pilot remains216 extra control nodes. Full phase .555931302CPU plus .011992039 pilot, including validation/archive work. These are descriptive execution times, not matched speedups.

At orders6/10/16, inherited absolute-threshold passes are3/4/11 of11 parent groups and relative passes0/6/11. The order10 loss versus old forced-thin rule remains4vs5 absolute and6vs7 relative. No tuning or output changes occurred. Actual array mask/copy/control/native cost is unpriced; scalar expression-model totals must not be transplanted as batch cost.

Exact command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/positive_inner5_batch/full.py`, under20CPU/2GB limits in a fresh output directory. No old inner/reference integration, geometry, network or MC replay. Canonical constant mapping96 operations, actually executed staged114, prior source generation and certificate construction retained as inherited setup.
