# T66: compiled geometry kernels with checked semantics

## Question
Can native kernels reduce geometry iteration time while preserving the mathematical integral and exposing failures? Wall time is the target; compilation alone does not improve a FLOP-normalized accuracy score.

## Method
T66 claim pushed788a682 before implementation. First phase owns only this folder. T20 owns the corrected Feynman native/interval adapter; T30 owns legacy/shared correctness fixes. Do not duplicate either. Great-circle legacy gcirc.py was read and has fixed angular lookahead and silent max_pieces termination. It is an ordinary-case comparison, not the sole correctness oracle. No legacy/shared files are modified.

Use two projective charts for an orthonormal plane(U,V): x=sign*(U+tV)/sqrt(1+t²), t inR, sign=±1. Positive homogeneity removes the radial factor. Each chart is an affine-input ReLU recurrence with piecewise affine hidden states/logits. Integrate a correct affine interval by (atan(hi)-atan(lo))/(2pi); the two charts cover the whole circle. This avoids a finite angular gate probe. At each current left root, a nonconstant preactivation a+b*t has right-side sign determined by b and whether current>=-a/b; a constant has sign(a). Ignore identically zero preactivations. Recompute the full affine recurrence at each gate breakpoint initially; do not claim incremental savings without implementation.

Within a gate interval, split at every nonconstant input-pair and logit-pair root. Evaluate the constant target/output ordering in each subinterval, using long-double interior representatives for finite close endpoints and slope/intercept lexicographic limits at infinities. Network output ties choose the first index. Persistent input ties are assigned stable index ordering explicitly; they have probability zero for a generic Gaussian plane but must not rely on an unspecified sorting implementation in controls. Input/output arrays and finite arithmetic are validated; overflow, nonprogress or cap exhaustion raises, never returns a silently truncated fraction.

## Frozen alternatives and falsifiers
1. Python loop/root bookkeeping may dominate, or dense numerical kernels may already dominate and a scalar native port may not help. Compare identical fixed planes, same weights, complete circles, cold build/loading and warm execution separately. Report pieces/root/interval counts and native arithmetic inventory; no speedup inferred from algorithm labels.
2. An apparent discrepancy may be a legacy lookahead/tie/cap defect or a new port error. Independently enumerate small rational affine gate cells with Fraction arithmetic and integrate their angular lengths at high precision; test zero/duplicate readouts, signed scaling and close roots. Legacy trained agreement alone is insufficient.
3. Native floating roots may still be unstable near coincident crossings. No epsilon root deletion or lookahead is permitted. Retain failures and maximum reference differences; exact real-arithmetic chart derivation is not an interval-arithmetic certificate of floating execution.

## Frozen first cycle
Build C++17 with-O3-ffp-contract=off into ignoreddata; Pythonctypes wrapper owns validation and exception conversion. Controls: deterministic H1/H2,T3/4 rational fixtures, including all-zero/duplicate readouts, identical gates, tiny-positive logit gaps, close recurrent gates, and cap0/1. Verify chart total mass1 and first-index ties, scale input/readout weights by positive1e-12/1e12. Use independent exact small-cell reference, not random sampling, for acceptance tolerance1e-12.

Then compare a frozen small pilot of4 orthonormal planes (QR of RNG660001 normals) for originalH16T10 and its standard relative-RMS .0025/.02 copies, plus publicH2T10 seed0; no truth values used. If controls pass and the pilot completes within20CPU seconds, benchmark32 fixed planes per case (same seed/prefix) with one warm repeat. Retain every case and any legacy cap failure/discrepancy. Report numerical agreement at1e-12, absolute differences, cold compilation overhead and native/PythonCPU. No higher-N sweep or speed threshold tuning. A later DD profile is a separate frozen phase after this checkpoint, not permission to run an unbounded enumerator.

## Status
Design and protocol frozen; native implementation and independent controls next. T55 math review requested. T37 prior diagnostic remains separately handed off.

## Results
Pending. No speed or accuracy claim yet.

## Compute accounting
One thread, <=2GB, ulimit-v2500000. Review/short cycles<=20CPU under codex-review.lock; longer checkpoints<=110CPU under codex-compute.lock. Include compilation, wrapper loading, plane generation, weight preparation and validation separately from warm kernelCPU. Actual coefficient matrix products, root work, sorting and readout checks remain necessary computation; a smaller wall time does not make them free. Great-circle accuracy estimation remains sampling-based when the planes are random; this task does not relabel it deductive or reuse its fitted control variates.

## Claims
[inferred] The two projective charts give exactly the same real-arithmetic angular integral as the homogeneous great circle. [speculative] Native execution can achieve the task's10x wall-time target on ordinary full models. No scoped result is confirmed without independent checks.

## Alternatives not ruled out
Incremental affine-state caching, native DD adjacency/bitsets, different batching, exact predicates and other compilers. The existing Feynman native work belongs toT20 and can supply coordinated evidence only after its own review.

## Failure modes
Near-coincident floating roots, overflow/underflow, target ties on nongeneric planes, compiler arithmetic differences, insufficient warmup, compiler/build overhead dominating small workloads, or measuring a silently truncated integral as a speedup.

## Full-recomputation checkpoint
The21 frozen small fixtures match the independent Fraction/70-digit oracle at identical final double values; cap0/1 raise. Build .502CPU, controls .032CPU. The4-plane pilot agrees with legacy within1.23e-15 and completes within its20CPU gate.

On the frozen32-plane cases, maximum absolute native/legacy difference is2.78e-15; both repeat exactly. Warm speedups are3.9568x originalH16,3.9569x small-noiseH16,4.1306x large-noiseH16, and12.0168xH2T10. NativeCPU is .092/.092/.092/.00399 versus legacy .364/.364/.380/.0480. The entire benchmark including preparation and one repeat uses3.675CPU,202MB. Thus the larger network has not met10x; the small one has. This first variant remains unchanged as the correctness/performance reference. Every model and source/build hash is retained.

## Frozen prefix-cache follow-up
The larger network uses8064/8056/8436 gate pieces over32 circles. Recomputing all10 recurrent coefficient steps at every piece is avoidable: cache both affine hidden coefficient vectors at each step and the next gate root per step. At the globally earliest next root, invalidate only its step and all later steps. Ties choose the earliest step. Without a finite lookahead, no earlier prefix gate can change before the chosen root; cached earlier roots remain candidates. Root==current is resolved by the right-side rule and excluded from future roots. Recompute readout each piece. This is a real arithmetic/work reduction in addition to native execution and must be labeled as such, not attributed solely to compilation.

Freeze separate incremental source/library, retain full-recomputation source. Re-run the same21 exact controls, cap tests and32 fixed planes; require agreement with the full native outputs and matching gate-piece counts, then report actual coefficient-step savings and cold/warm cost. Do not increase batch size or choose favorable cases to meet10x. Alternatives: prefix reuse may dominate, or scalar readout/root/sort work may become the bottleneck; count the components before any further optimization. T55 independently reviews invalidation logic. DD profiling remains deferred until this bounded comparison closes.
