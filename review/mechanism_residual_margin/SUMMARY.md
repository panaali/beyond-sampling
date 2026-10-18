# T201 saved residual-margin results (independent full audit pending)

The fixed lookup frequently removes errors made by the original network. All original labels were reproduced during the diagnostic; no fit, recurrent forward or input was regenerated. These are measurements on the already fitted original checkpoint, not a deductive estimator or causal intervention.

| Panel | Both correct | Network error erased | Error introduced | Both wrong, same class | Both wrong, different class |
|---|---:|---:|---:|---:|---:|
| train | 61999 | 2205 | 437 | 492 | 403 |
| holdout1 | 61980 | 2240 | 449 | 482 | 385 |
| holdout2 | 62027 | 2183 | 470 | 490 | 366 |

Every row sums to65536. Network errors are columns2+4+5; lookup errors are columns3+4+5. Agreement is columns1+4. All five counts independently reconcile the previously published task/representation totals.
- holdout1: 2240/3107 (72.095269%) of original errors are erased; 449 previously correct rows become lookup errors.
- holdout2: 2183/3039 (71.832840%) of original errors are erased; 470 previously correct rows become lookup errors.

## All sixteen contributions on error-erased rows

Each entry is the aggregate signed contribution `(hhat_i-h_i)*(Wo_truth_i-Wo_original_competitor_i)`. Neuron indices are zero-based and every neuron is reported. No residual intervention or top-k selection was performed. A signed contribution is descriptive algebra, not causal input attribution.

| Neuron | Train signed sum | Holdout1 signed sum | Holdout2 signed sum |
|---|---:|---:|---:|
| 0 | 2765.33878 | 2511.91263 | 2607.27263 |
| 1 | 2808.32252 | 2712.4396 | 2623.50604 |
| 2 | -816.448193 | -830.061405 | -788.160495 |
| 3 | 4270.49673 | 4046.05654 | 4281.78717 |
| 4 | 396.950433 | 329.769341 | 427.466687 |
| 5 | 3698.52485 | 4307.25591 | 4154.88522 |
| 6 | 2307.65196 | 2033.71573 | 2239.62843 |
| 7 | 2097.32261 | 2247.30315 | 2240.83973 |
| 8 | 7588.26157 | 7766.45638 | 7041.80701 |
| 9 | 124.067892 | 147.65598 | 125.57218 |
| 10 | 3687.17177 | 3734.27286 | 4079.57859 |
| 11 | 3068.90853 | 2930.44378 | 2431.97712 |
| 12 | 1813.79892 | 3059.18008 | 1888.53309 |
| 13 | 2425.09743 | 2648.99847 | 1945.71231 |
| 14 | 4271.21384 | 3800.77433 | 4331.53284 |
| 15 | 5039.79376 | 5030.45264 | 5290.22244 |

All signed and absolute aggregates for every transition, the complete positive-bin/case/picked-rank context tables and all per-row terms are retained in saved/results.json and its192 immutable archives. A few-neuron correction was not tested; its usefulness cannot be inferred from these totals.

## Numerical and resource scope

The run completed in9.660255768 scriptCPU seconds with peak RSS55516KiB. It made192 diagnostic calls and reconstructed196608 lookup hidden states/readouts from the old coefficients. All389 expected input/label/coefficient/weight archive reads occurred. The maximum observed floating closure discrepancy was2.2137847111e-13; this is not a certified forward-error bound. No output ties or zero term sums occurred on these panels. Large cancellation ratios remain in the raw statistics.

Partial expression counts: lookup/readout40,894,464 multiplications +35,782,656 additions; residual/margin7,077,888 subtractions +3,145,728 multiplications +5,898,240 additions +3,145,728 absolute values. Selection, validation, decoding, BLAS implementation, reductions/assembly and serialization have no complete native charge here. This is diagnostic work, not a new estimator cost. Inherited fitting/input/forward construction remains separate.

Controls used0.180262936 recorded authorCPU across the preserved zero-call environment failure and completed fourteen-control recovery. Their independent saved audit4459a768 passed. Full source gatee706d182 was published and bound inbd0e9e3a/sharedf2d339e1 before the sole trained-data diagnostic. Author raw88d44ff9 is published/shared31698ad9. T55 full saved review is assigned by9082bb7e and pending.

Command: one-thread `flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/mechanism_residual_margin/saved.py`, under external `ulimit -v 2097152; ulimit -t 20`, internal19CPU hard policy and17CPU soft stop. Existing output prevents replay. No failure or cap extension occurred in this phase.

## Interpretation and remaining alternatives

The roughly98% task score does not by itself demonstrate faithful reproduction of the network: this fitted method changes thousands of its decisions. The same-state feature summary can solve the task exactly in principle, yet need not determine original-network correctness. Prior T138 supplies a scoped exact non-sufficiency witness; this phase measures a different finite reconstruction discrepancy.

The margins may reflect omitted information, fitting error or their interaction. This diagnostic does not separate those causal explanations, establish an unavoidable population error, transfer to noisy weights, certify the floating network, or show a cost advantage. Conditional residual laws, retained lower-input information and different fitted families remain open; no approach is declared dead.
