# T278 fixed H4T4 application source/resource review

## Question
Is the separately frozen single-model, two-rule residual-band discriminator ready after independent synthetic control verification?

## Method
Independent source/algebra inspection and hash/Git metadata checks. No target archive decompression, numerical face/CDF/rule evaluation, model calls or control replay in this review. Read run_application.py, application_math.py, APPLICATION.md, application_protocol.json, application_launch.sh, inherited exact interval/Gaussian modules and the frozen face core. Reuse the independent18-control review e5ecc281 and all its bound input files.

## Face-law and interval reasoning
Write the two selected coordinates as x_a=u+delta and x_b=u-delta. The two-coordinate Jacobian is2, giving joint density exp(-u^2)/sqrt(pi) times exp(-delta^2)/sqrt(pi). For R23, conditional top and low coordinates supply factors (1-Phi(u))*Phi(u)^(T-3); for R12 the factor is Phi(u)^(T-2). Thus the unordered pair/top configurations have equal weights Z23/12 and Z12/6 at T4, with no missing factor two: both signs of delta are integrated explicitly. The two R12 lower positions remain separately sampled in index order; their ordering is not restricted.

R23 has top coordinate o1, pair u+/-delta and lower maximum o2. Ordering and the nearest-gap partition require |delta|<min((o1-u)/3,u-o2). R12 requires |delta|<(u-o2)/3. R23's second-ranked label is b on negative delta and a on positive; R12 reverses these labels. Source implementation and eight kind/true-index buckets agree. Boundaries are null under the continuous ideal law, while identically-zero readout margins preserve first-index ownership over nonzero-length intervals.

The implementation freezes a finite160001-point[-8,8] cumulative table with interpolation/clipping, exact stored binary face coordinates, and a binary cap8.5/sqrt(2). Its windows are recomputed rationally from those stored coordinates; source rounded-tau differences are retained. This is an explicitly approximate numerical pushforward, not exact ideal coarea integration. Integrated window mass is kept as a diagnostic. Accuracy1-minus-computed-error does not separately enclose omitted/table/cap error.

Inherited exact partition and frozen-mask continuation operate in delta, and signed symmetric differences are formed separately on each half-window. The standard-normal coordinate is z=sqrt(2)*delta. The separately timed first/second/direct formulas are algebraically the reviewed helpers and have twelve fixed synthetic formula-parity checks before model loading. Ideal Taylor remainders sum with positive rule/configuration weights; floating bound arithmetic is not directed. Direct signed-band parity within3e-13 checks only inner consistency, not face-law or outer integration accuracy.

## Status
The final source_gate.json identifies the final reviewed repair commit. Original application fadd5157 was amended before execution: atomic actual/frozen partition returns, per-half intermediate returns, formula-parity values before assertions, final event-count guard, explicit heavy-lock naming, reference archive/dimension/hex endpoint binding, and the raw face-return snapshot. Numerical rules, controls and tolerances are unchanged. Publish completed review and immutable application gate before one launch; the execution scope must be published separately as promised by the owner.

## Fixed numerical scope
GL orders3 and5 over all18 configurations give486 and2250 instances. Their18 shared centres occur once, so2718 unique points and5436 actual/frozen partitions. The central18 points are processed first. The frozen elapsed-CPU projection stops above75CPU; this is a scheduling test, not an unbiased complexity estimate or authorization to retry. Successful completion requires5439 events. Exactly64 predetermined indices retain full completed affine cells; all other completed points retain errors/bands/values/ledgers, and active snapshots preserve the most recent full returns. A stopped run cannot claim full-rule coverage.

The model and inherited reference load from fixed byte-pinned artifacts. Reference H/T, exact artifact SHA and binary endpoint hex values must agree. No fresh reference, checkpoint, training or RNG is used. Historical reference/geometry correctness and real-dyadic versus rounded-forward semantics remain inherited qualifications.

## Compute accounting
One heavy-work lock /workspace/work/codex-compute.lock, one numerical-library thread,90 internal CPU seconds,105soft/110hard CPU,2GiB address space,120wall after the lock. Raw completed-point stream cap128MiB; inherited per-layer10000-piece cap. Full process cost includes imports/verification/observation/aggregation; composed method timers omit those costs explicitly.

Each corrected method pays actual plus frozen partition/error discovery, matching, frozen mass and its own correction evaluation. Zero pays frozen work; actual pays actual work. Face/table/rule/model/reference setup is included in every composed method. Shared centre work is charged once per standalone rule even though reused in the research run. The CPU-times-squared-reference-error diagnostic is not equal-FPE performance or a statistical timing interval. The floating robust-halving flag refers only to the retained reference interval and observed component times; it is not an outward numerical certificate.

## Claims
[inferred] Source law, coordinate conversion, side labels, exact interval wiring and accounting conventions are consistent for this bounded discriminator. Finite face/rule computations are classified as inductive under the project's directive even without RNG. No ARC/HP1/rung pass or general negative follows.

## Alternatives not ruled out
Outer quadrature, table shift/clipping and finite caps may dominate the inner Taylor correction. Exact band discovery may be too expensive. Other residual discovery methods, rules, models, and representations remain open. A scheduling stop is not an accuracy result.

## Failure modes
Fresh outputs and noclobber launcher prevent overwrite. Atomic active snapshots preserve returned face/partition/half stages before later assertions/calls; completed points append once. Failures preserve counters, active stage, partial data and incomplete status, with no automatic retry. Stored full-cell coverage is a sample, not exhaustive independent recurrence verification. A separately source-gated saved audit must check completed coverage or the exact stopped prefix without rerunning the producer.
