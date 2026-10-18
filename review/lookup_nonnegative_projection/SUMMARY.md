# T224 fixed saved-panel outcome

Author raw45ab5489, shared32e3aa0f. Independent full saved audit pending. Original T201 results remain primary; this is a post-original-result descriptive comparison on reused data, with no fresh validation or new generalization CI.

| Original panel | Negative rows | Hidden SSE reduction | Network agreement before -> after | Task accuracy before -> after | Agreement improved / worsened | Task improved / worsened |
|---|---:|---:|---|---|---|---|
| train | 16516 /65536 | 0.220952% | 95.353699% -> 95.436096% | 97.967529% -> 97.911072% | 145 / 91 | 104 / 141 |
| holdout1 | 16695 /65536 | 0.263096% | 95.309448% -> 95.390320% | 97.991943% -> 97.906494% | 157 / 104 | 104 / 160 |
| holdout2 | 16570 /65536 | 0.239776% | 95.393372% -> 95.507812% | 97.976685% -> 97.950745% | 179 / 104 | 132 / 149 |

All196608 rows and192 chunks completed once. All146827 rows without negative predictions retained exactly equal stored readout values (numeric equality permits signed zero); no floating SSE increase or projected readout tie was recorded. Negative coordinate counts are23892/24099/24183, and every per-neuron magnitude/context table is retained. No neuron or panel was selected.

The fixed projection is a small fidelity improvement on these reused panels, with task regressions. It does not remove most of the reconstruction discrepancy or prove a causal mechanism. The original unconstrained fit, a separately constrained fit, richer features and exact mechanisms remain distinct possibilities. This finite result is not a family-level negative and does not authorize predictor adoption.

Named primitive expression counts: comparisons3145728, subtractions6291456, multiplications37748736, additions35389440. These exclude validation/aggregation and use2H-1 mathematical dot counts; they are not native/T196/full-cost totals. Author main-scope CPU10.535777677, peakRSS55504KiB; external20CPU/2GiB cap covers timing exclusions. Prior controls consumed.038630243CPU; inherited fits/forwards/reference construction remain separate.

No new inputs, recurrent forward, lookup reconstruction, fit or random/reference calls. Raw arrays are under `saved/`; the exact command and runtime/source prerequisites are in `SAVED_PROTOCOL.md`. Independent reviewer may reconstruct saved algebra but must not repeat the author application.
