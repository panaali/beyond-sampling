# Saved T3 reconciliation

After source/theorem gate `c22e80a1` published `1683823d`, immutable execution gate `3931010d` was published `2c8173c1` before the sole saved-only run. `check_saved.py` completed in 0.008000665 process CPU seconds under the declared review lock, 20 CPU-second and 2 GiB limits. No author cap, geometry, mass, network or random-input evaluation occurred. `saved_run1.log` is empty because the checker completed without an exception.

All 73 payload hashes, 51 T3 row identities, counters and ten five-seed groups pass. The printed 834/zero-violation text has 135 retained summary lines; it remains author-reported control evidence, not an independent numerical replay. The sum of saved author `sec` fields is 11.2733824253 wall seconds, not process CPU.

The table below reproduces the report's maximum per-model point ratio at each threshold, using exact Fractions of stored floats and **continuous** canonical draw budgets FLOPs/511. The source counter convention is FLOPs=2×ops. Point reference accuracy and incomplete implementation cost qualifications remain.

| Test | Threshold | Mean retained cells | Maximum point error ratio | Maximum reported B² ratio |
|---|---:|---:|---:|---:|
| cap | 1e-8 | 225.6 | 0.0001213483 | 9.36861e-9 |
| cap | 1e-7 | 207.4 | 0.0001216393 | 6.39353e-6 |
| cap | 1e-6 | 184.2 | 0.0001212728 | 0.0007083140 |
| cap | 1e-5 | 168.0 | 0.0001289484 | 0.04496614 |
| cap | 1e-4 | 159.0 | 0.0001474756 | 2.67752317 |
| mass | 1e-8 | 211.2 | 0.0001339305 | 6.09234e-8 |
| mass | 1e-7 | 184.4 | 0.0001327934 | 6.90123e-6 |
| mass | 1e-6 | 165.4 | 0.0001402066 | 0.0005107037 |
| mass | 1e-5 | 141.8 | 0.0001640945 | 0.04813946 |
| mass | 1e-4 | 110.2 | 0.0018361859 | 12.61513091 |

All thresholds and all individual rows are retained in `saved_results.json`; no winner is selected. Historical JSON `N` agrees with FLOPs/500 to one stored ulp and is explicitly separate from the corrected report convention. The maximum absolute residual between global B and summed step bounds plus the declared mass padding is 4.15595e-18. This is a saved-data consistency diagnostic, not an outward-summation proof, because individual summands are not archived.

There is no T4/T5 replay or rendered pruning report in the frozen author tree. A reported B² ratio below one does not establish a certified estimator: the cap padding proof, floating geometry, wrong-centroid numerical masses, retained recipe error and total cost remain qualified. Independent saved-output review is required before final handoff.

Reproduction command (fresh isolated outputs only, never overwrite the completed attempt): `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock timeout 25 /workspace/clean-neurons/.venv/bin/python review/split_attribution_review/revised_cap/check_saved.py`, with external `ulimit -t 20` and `ulimit -v 2097152`. The checked-in execution gate and source pins are part of the contract.
