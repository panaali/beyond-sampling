# Fixed six Q controls: bounded results

All21 original/observed pairs (42 Q evaluations) preserve output bits and warning category/message multisets. Array arithmetic and both dense-operation conventions match the source formulas on every pair. Original syntax is restored exactly by stripping observer wrappers. Independent saved audit by recover_910 passes all21 pairs,9102 events/3385 call records and56 progress records; review40ea5646 is shared at2270bf54.

| Fixture | Batch rows | Active interval positions | Visible 2-MAC + array arithmetic per row |
|---|---:|---:|---:|
| zero_readout_preserved_tie | 2 | 4 | 8449 |
| zero_input_weights | 2 | 3 | 7699 |
| zero_recurrent | 3 | 4 | 8449 |
| repeated_gate_breakpoints | 2 | 7 | 10699 |
| heterogeneous_intervals | 3 | 12 | 14449 |
| tiny_slope_cutoff | 3 | 7 | 10699 |

These are incomplete numerical categories. Scalar arithmetic, comparisons/Boolean operations, library calls, sorting, selection/copies, control and native implementation costs remain separate. The observed J range3..12 does not test the full19-position upper envelope. No historical/trained cost factor follows.

All six batch outputs match the corresponding singleton outputs in this fixed panel; this is not a BLAS-invariance guarantee. The zero-input-weight fixture preserves its original invalid-division warning. The zero-readout fixture preserves the original all-zero Q result, an analytic tie-ownership defect tracked separately inT210; no repaired output is used in this audit.

The sole attempt used7.69401231 process CPU seconds within the external20CPU cap and98,188KiB peak RSS. The launcher used the project-prescribed virtual-memory limit2500000KiB, which is not a strict2GiB hard limit; observed RSS remained below the2GB process budget. This includes durable JSON serialization, tracer and fixture overhead; it is not estimator timing. There were no trained checkpoints, stochastic/model input draws, Monte Carlo, author-module/advice/precompute imports, or native-kernel traces. The prior source-only observer name collision was corrected before execution, with original source history retained.

Reproduction requires a fresh output location because the frozen runner refuses prior artifacts. Do not rerun completed controls. Original command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock -n /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/srb_cost_review/q_controls/run.py`, preceded by `ulimit -v 2500000` and `ulimit -t 20`.

All raw events and function trees are preserved in results.json/progress.jsonl/trees.json. A later cv allocation/rejection/returned-charge phase requires its own published frozen scope and gate.
