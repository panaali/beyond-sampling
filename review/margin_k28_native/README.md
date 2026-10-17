# T115: frozen k28 provenance before native cost

## Question
Can existing k28 training-axis evidence support a direct native-source cost
audit without substituting models, references or costs from other levels?

## Method
TASKS-only claim4dbeebbc. `inventory.py` verifies pinned original margin/rung
files, recipe AST identity, noise generator and all5120 saved truths/errors.
This initial phase performs no estimator, model-forward, reference or download.
T90's reviewed checkpoint/truth panel is k29, not k28, and is not relabeled.

## Status
All five current canonical models,5,120 original noisy specifications and both
recipes are complete. Full saved summary8d7a52c0 reconciles all80 shards and
five clean controls. Allfive raw seed audits and final global/clean reviewfb8a2443 pass. See `HANDOFF.md` for all values and qualifications.
Original historical checkpoint/per-copy byte hashes remain absent; current
canonical provenance and exact historical-error/mean-counter reproduction do
not retroactively prove historical identity.

## Results
Inventory3196b26e: all5120 original truths and both recipe error arrays present;
ASTs identical, historical hashes/counters missing. Five current canonical
2149-byte objects01963e68 validate generation/MD5/SHA/length, metadataDec12 2025.
Initial repeated-header parser failure9e51143a retained; get_all repair reuses
seed0 payload and fetches only remainingfour. All5120 noise arrays match exact
original generator84b15611, current hashes only.

Fixed15 controls013b4ba2 independently05cd1e39: all30 recipe outputs reproduce
old value-minus-truth errors exactly; baseline/native bits,STATS and grouped
errors agree. Allten clean legacy/cell counters match.60 calls,8.67636547CPU,
peak212556KiB. Numeric totals1,675,867/2,353,523 for85B/112, no multiplier transfer.
First64 scope7ab84a99 independently38378124: all254 raw records/128 pairs pass,
26.550111182CPU. Numeric costs8,423,528/10,518,429; matched111 ratios
.00026659167459924/1.2081022056782487e-6, no pilot individual losses. This is
not full-panel evidence. The completed79 fixed64 continuation shards preserve both recipes,
all ten noisy/five clean control reuses, whole-seed saved counter means and
every individual loss. See `FULL_PROTOCOL.md`; final raw auditfb8a2443 passes.

Continuation command per fixed seed/noise/block:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-compute.lock timeout 120 /workspace/clean-neurons/.venv/bin/python review/margin_k28_native/shard.py SEED NOISE BLOCK`
with external `ulimit -v 2500000; ulimit -t 110`. Pilot(0,20,0) is excluded.
Saved-only `full_summary.py` refuses incomplete coverage and completed overwrite.

Independent saved seed audits ac75ff27/5663c685 pass all site sums, original
subtraction errors and exact mean counters. Seed0 numeric111 ratios85B/112
.0003436985830/.00001354385190, zero individual losses. Seed1 ratios
.04952494424/3.068269852e-7, with TWO85B individual losses (worst1.2840496);
112 has none. Historical72 has no seed1 losses and stays separate. The complete summary additionally retains23/1 seed4 losses, for25/1 total
85B/112 losses; all ten seed averages remain favorable. Final global review
fb8a2443 passes; no uniform individual or complete native-cost success is claimed.

Reproduce source inventory with
`python review/margin_k28_native/inventory.py` (refuses completed overwrite).

## Compute accounting
One thread2GB, provenance review<=20CPU and controlled primary105soft/110hard
plus90CPU graceful. Inventory.023990698CPU; reconstruction.115632985CPU;60
fixed-control estimator calls above, no reference/model-input MC. Network
provenance used five payload downloads andsix HEAD requests including parser
rescue, all failures/bytes retained; no seed0 redownload. Numeric111 and
historical72 costs remain separate. Pilot full80-shard projection2124CPU is
not a runtime guarantee. Continuation reserves110CPU before each process under
a3000 recorded-research CPU envelope, including controls/pilot once. Actual
recorded CPU2081.222020532, maximum31.417634027CPU, peak212556KiB;20,500
plain/traced calls total, no evaluator replay during saved-summary/review.
no k29/k30 multiplier transfer, complete native/runtime claim or new training.

## Claims
[measured] Original errors/truths and mean counters exist for allfive seeds.
[measured] Every noisy error subtraction and original seed mean counter agrees.
[measured] Both recipe averages favor the candidate on allfive seeds;25/1
individual noisy losses remain.
[inferred] Current canonical checkpoints and unchanged error-subtraction parity
support a qualified recovery, but do not prove original byte identity.

## Alternatives not ruled out
Alternative1: missing original weights change reproduced estimates; canonical
object metadata and fixed all-source output/error controls could discriminate.
Alternative2: source drift changes the method; AST comparison checks complete
frozen evaluator semantics before any native measurement. Noise ordering or
dtype drift needs explicit generator/conversion controls. Saved errors alone
do not recover original output bits or individual counter values.

## Failure modes
Missing provenance is an evidence gap, not estimator failure. No silent new
reference, checkpoint choice or fitted counter; preserve any future mismatch.
