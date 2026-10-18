# SRB per-call draw diagnostics

## Question
Can the actual completed candidate count be exposed without changing the
estimator result, its reported charge or its RNG-call order?

## Method
The local `adapter.py` preserves the pinned numerical AST and wraps only call
boundaries. `DESIGN.md` specifies separate requested/completed counts, quotas,
kept rows, original source drawn value and reported cost. Immutable snapshots
reset per call while preserving the estimator's original precompute cache.

## Status
Bounded controls complete, raw79910a7a. Independent saved audit is pending;
source/resource review0aef2c76 (label correctionb9f05786) was publishede9a4db86,
and execution manifestd93a0cbf was published2cac96b0 before the sole run.
Task scope and independent saved reviewer were published13593efa.

## Results
[measured] All four fixed deterministic cases preserve prior estimate bits,
reported charge, request sequence and warnings, without replaying the old
baseline. The same instance retains its precompute cache and immutable snapshots.

| Fixture | Kept inputs | Completed candidates | Original raw charge |
|---|---:|---:|---:|
| Minimum quota | 28 | 50,000 | 48 |
| One batch, budget1000 | 584 | 50,000 | 993 |
| Delayed second batch, budget1000 | 584 | 100,000 | 993 |
| Binary floor, budget17 | 28 | 50,000 | 48 |

Three deliberate failures pass: first RNG call fails before returning; second
RNG call fails after one returned batch; synthetic Q fails after complete draw.
Requested/completed pairs are50,000/0,100,000/50,000 and50,000/50,000. Unknown
kept/charge values remain null. A nested invocation is rejected while its outer
call completes; immutable snapshot mutation is rejected. All raw failures remain.

The sole command used the pinned venv, one thread, `ulimit -v 2097152`,
`ulimit -t 20` and `flock /workspace/work/codex-review.lock`, executing
`/workspace/clean-neurons/.venv/bin/python review/srb_draw_diagnostics/controls.py`.
Outputs are `controls/results.json`, `controls/progress.jsonl` and `run1.log`.
The harness refuses existing outputs; do not overwrite or rerun this attempt.

## Compute accounting
Main-scope0.343629783 CPU seconds,78,996KiB peak RSS; external cap covers imports
and final serialization excluded from that timer. Eight outer invocations plus
one nested attempt rejected before original code; original Q executes4 times,
synthetic Q twice (one raises). Across all outer calls500,000 candidates were
requested and400,000 returned by deterministic API streams. Zero Gaussian-law,
trained-forward, fitting or baseline-replay calls. Nineteen progress records.
Metadata, wrappers, validation, cache initialization and serialization are real
research overhead; unchanged returned charges are not validated full costs.

## Claims
[measured] Fixed deterministic interface and failure/lifecycle controls pass;
independent saved review is pending. These controls measure no accuracy advantage.
[inferred] Unmodified numerical AST and forwarding wrappers preserve the fixed
ordinary ndarray RNG API under successful diagnostic metadata operations.
No pricing correction, historical draw-count reconstruction, production adoption
or complete native-cost claim follows. Prior T214 policy-conditional cost gap
remains a separate result with its explicit assumptions.

## Alternatives not ruled out
An exception may occur after an RNG internally advances state but before it
returns; report only completed batches and keep internal work unknown. A stale
per-call counter or mutable snapshot could misattribute later work; lifecycle
controls must test two calls and retained prior snapshots. Different allocations,
CV modes, concurrency and arbitrary stream implementations are outside scope.

## Failure modes
Source/API drift, observer failure, unsupported nested calls, cache corruption,
mutation of prior snapshots, extra RNG requests or changed outputs must remain
visible. No failed control can be dropped or treated as a pricing-policy result.
