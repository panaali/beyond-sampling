# T188 exact-mode prefix census

## Question
How many exact-mode depth3 pieces at H8T4 share the same global future margin map?

## Method
Fixed four-parent pilot; two exact flags with unchanged geometry, source review first. PROTOCOL.md freezes scope; exact_common.py records stop observation, pilot.py groups exact histories/states using reviewed weight predicates.

## Status
Full exact-mode census completed1027142e after shared independent source gate36cbf0c3/ff313d14. Pilot independently32b8689c passed. Full saved audit pending; no new geometry running.

## Results
All96 initial charts per model, stopped at depth3 before final input/readout:

| Model | Prefix pieces | Global histories | Global future maps | Same-truth histories | Same-truth future maps |
|---|---:|---:|---:|---:|---:|
| H4T4 seed0 |868|66|65|220|217|
| H8T4 seed0 |2203|210|210|668|668|

Every repeated history crosses initial charts; there are no same-parent repeated-history groups. H4 has one global different-history all-OFF reset (three truth-specific groups); H8 has none. Stored-map group counts match exact-map group counts, but individual stored maps equal exact only21/868 H4 (zero states),0/2203 H8. Thus floating evaluation and exact represented maps are not interchangeable.

Global history sets equal the prior recipe census, although exact mode keeps36/181 additional pieces and1/4 additional same-truth histories. All66 H4 prefixes also equal the first3 masks of the independently archived450 full gate histories; no mismatch. This is saved set agreement, not a new exact partition proof.

The original pilot is retained:84 pieces/42 global/48 same-truth states. Its recipe comparator had81/42/47. All pilot/chunk lookahead, mass and final callbacks are zero.

Commands (outputs refuse overwrite): one-thread `flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/future_equivalence/exact_count/chunk.py H batch`, under `ulimit -v 2500000` and `ulimit -t 20`, H=4 then8, batch0..7 serially, stop first failure. Same envelope for `aggregate.py` and `compare.py`. Source/runtime/model/protocol pins and durable per-state/progress records accompany every phase. The H8 four-parent pilot is reused once; all H4 geometry is new.

## Compute accounting
New chunk in-script CPU1.396013064 H4/3.783344414 H8; inherited H8pilot.231858944 separately; saved aggregation1.071916962. Initial Python imports and final serialization excluded, so no matched runtime claim. Combined inherited source proxies938169/3433633, finish entries1252/2779. New exact state formation H4 28800mul+28800add; H8 383616+383616, plus inherited pilot27216+27216. Weight decodes288/832 plus H8pilot104. Initial96 charts/model and exact predicates reused; source imports, wrapper formation, Fraction/hash/GCD/control/native implementation remain distinct/unpriced. summary.json contracts recorded ledgers only.

## Claims
[measured, full independent audit pending] Exact-mode prefix counts above. [inferred under reviewed fixed-weight predicates] These exact state counts are counts of global/common-positive-projective future-margin maps. [measured] Beyond repeated chart representations, only the H4 reset merges different histories. No measured geometric union, mass reduction, cost-MSE or asymptotic gain.

## Alternatives not ruled out
Domain-only or label-only equivalence, approximate certified pruning, changes of chart representation, and reuse of shared classifier planes may still help. The Claude child owns truth-chart reduction; this phase does not duplicate it. A finite absence of extra global matches does not exclude these alternatives.

## Failure modes
Exact mode uses floating TOL/HV clipping; no certified real partition or region feasibility. Prefix pieces are not terminal cells. No mass or final accuracy follows from these counts. All caps passed; no geometry retry or silent failure removal.

## Publication chronology
The h4 source review JSON existed locally before the pilot; its hash and completed gate were published in512f24c0 before execution. The review artifact/history itself reached shared5c5f3896 after pilot608013fa/e7b0084e. This is a publication-order deviation, not a numerical rerun or a missing pre-execution source review. The next full-scope gate must be shared before execution. Original records remain unchanged.
