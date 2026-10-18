# Fixed retained-summary protocol

T258 scope66bc9df5, following source review of e9f02fc9 through b526dcf9. This is a saved-only consistency test, not the requested independent-seed replacement reproduction.

Inputs are exactly the four JSON files and three logs in ../retained/inventory.json, plus immutable source/manifest/protocol and completed written-review and source-review artifacts. Source bytes are not imported. No NumPy, archive/checkpoint/model, network, training/fitting, MC/RNG, original estimator or fresh data calls.

The checker records exact integer compatibility of each tested Boolean mean with its known denominator using neighboring-binary64 midpoint intervals, clamped to [0,n], then filters candidates by exact Fraction division rounded to binary64. This assumes the historical Boolean-mean final division has that rounding behavior; it does not prove historical execution. Unique counts are required for subsequent contractions. Empty/ambiguous candidates are preserved and stop the run, not silently rounded to a preferred answer. Bin endpoints may include Infinity as source metadata, but tested rates and R-squared entries must be finite; empty binned metrics use source None. Literal conditional table supports use the full N denominator.

Fixed checks: C/C2/D sample dimensions; all12x11x16 validation/evaluation shapes; first-max validation-selected families; cumulative full-neuron group and final-step-only identities; per-neuron evaluation ordering; unconditional replacement metrics; five D confusion counts; disagreement, correctness conditional rates and endpoint linkage; exact paired task-gain variance (no new CI); three exclusive rank cases plus separate overlapping blind case; near/far union; gap/pattern partition supports and errors; far correct/error decomposition; two length10 rank-picked histograms; seven noise rows and direct-fit rates; literal ten fit-step log records. The v1 record is only used for sample-size consistency; other v1 scores remain unverified. Log timing/metric reconstruction and printed README rounding are not asserted by this phase.

Every rate candidate is saved before uniqueness assertion; checks save raw evidence before failure assertions. Same-directory atomic fsync+replace prevents truncation of the previous result. Progress is append-only; fatal resets completed flags. Refuse existing output/result.json or progress.jsonl. No automatic retry or threshold relaxation. Preserve absent raw archives and every discrepancy.

Resources: one process, review lock, 17CPU internal measured-main timer, soft AND hard RLIMIT_CPU<=20 and RLIMIT_AS<=2GiB, one-thread environment. The external cap covers imports; the internal timer starts after stdlib imports, and final serialization follows its reported CPU snapshot. No timing or native-cost claim. Gate must directly pin checker, protocol, this document, inventory, source-review artifact and completed written review. Both protocol pins and gate pins are validated separately; source-review direct pins checked too. Source review commit/publication must be nonempty hexadecimal identifiers. Execution scope/gate published before sole run. Independent final saved audit remains mandatory.

Expected progress is begin, selection_complete, partition_complete, complete on success; fatal is appended on any failure. The number of saved count/check records follows the fixed input schema and is not itself a raw-trial count. Unknown integer supports are an evidence limitation, not model failure.

Command after published gate only:

```
set -e
set -o noclobber
ulimit -v 2097152
ulimit -t 20
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /usr/bin/python3 /workspace/work/recover-small_h2_56/review/replacement_claims_recheck/saved/check.py > /workspace/work/recover-small_h2_56/review/replacement_claims_recheck/saved/run1.log 2>&1
```

No execution has occurred while preparing this protocol.
