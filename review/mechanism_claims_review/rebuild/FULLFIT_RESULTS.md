# Fixed reconstruction experiment — preliminary saved results

Independent saved audit is pending. This is a new frozen reconstruction experiment on the original checkpoint, not validation of the provenance of historical percentages and not an ARC estimator or cost result.

| Panel | Original task accuracy | Lookup task accuracy | Lookup network agreement | Case2 separable task accuracy | Case2 separable network agreement |
|---|---:|---:|---:|---:|---:|
| train | 0.9526977539 | 0.9796752930 | 0.9535369873 | 0.9004210526 | 0.8754066986 |
| holdout1 | 0.9525909424 | 0.9799194336 | 0.9530944824 | 0.9012735822 | 0.8750785281 |
| holdout2 | 0.9536285400 | 0.9797668457 | 0.9539337158 | 0.8987821844 | 0.8732823846 |

The lookup predicts task labels more accurately than it reproduces the original network. The case2 separable model is scored only on its declared case2 subset. Neither is a deductive accuracy estimator. Individual numerical Clopper–Pearson intervals are in the raw results; they condition on this fitted model and do not include training variability or simultaneous coverage.

## Compute accounting

Exactly 191 new 1024-input chunks, 195584 new vectors, 1955840 scalar normal draws, 217 homogeneous least-squares calls. The 1024-input pilot prefix and its hidden/logit arrays were reused once without another forward. Total distinct inputs are196608; no new library forwards or reference calls. Script CPU7.467979396 seconds; external Bash reports7.64 wall,7.27 user,0.36 system seconds. Maximum RSS183504KiB. Internal CPU excludes interpreter/top-level imports and final serialization; external process limit covered the whole run. The T1965948 sampling benchmark is a planning reference, not an as-implemented charge.

All raw arrays, coefficients, rank/count/singular-value metadata, labels, disjoint error tables, RNG chains and partial accounting are retained. No failure, resampling, coefficient adaptation or repeated run occurred.

## Reproduction

The single run used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `ulimit -v 2500000`, the primary compute lock, and `bash -c 'ulimit -t 110; time -p /workspace/clean-neurons/.venv/bin/python fullfit.py'`. The immutable output guard prevents repeating it in this folder. `fullfit_gate.json` was published as36db1a21 before execution and binds independently reviewed source0fc2abe8/shared e4b0ff4d.

## Limits and alternatives

[measured, pending saved review] The two heldouts reproduce the approximate98% task score for this newly fitted lookup. They do not establish98% network agreement, explain all network errors, validate historical splits or provide an estimator of network accuracy. Other feature sets, distributions, checkpoints and fit variability were not tested. Original network errors remain in every disjoint bucket; no method family is ruled out.
