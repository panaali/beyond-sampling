# T268 saved sign-sector result — independently audited

The sole fixed contraction completed all 5,058 inherited correct simplices, 17 controls, 20 chunks and 76 progress events. Raw commit `e893ef82` was published as shared `b9c1f28f`. Independent saved audit `27db195b` / shared `7026e40e` checked all 46 pins, seventeen controls, 5,058 rows, twenty chunks, sixteen sectors, five blind sectors and 76 events. Its rational Gaussian elimination and midpoint-distance loss reconstruction passed in 0.771083525 CPU seconds with 79,756 KiB peak RSS, without producer calls or failures.

Each sign orthant has Gaussian mass 1/16. The five designated blind sectors therefore have total mass 5/16. Their conditional correctness probabilities are approximately:

| Positive input positions (zero-based) | Correct simplices | Conditional correctness |
|---|---:|---:|
| None | 1,207 | 0.37948404–0.37948408 |
| 0 only | 312 | 0.54672729–0.54672736 |
| 1 only | 196 | 0.70169822–0.70169830 |
| 2 only | 125 | 0.35266899–0.35266903 |
| 3 only | 548 | 0.85813014–0.85813023 |

Displayed decimals are approximate summaries, not outward-rounded bounds. The exact rational intervals are retained in `output/results.json`, together with all sixteen sectors and every component record.

The five-sector correct mass is approximately 0.177419293–0.177419313, and conditional accuracy is 0.567741737–0.567741801. Every one of the sixteen sector intervals lies strictly inside (0,1) after conditioning on its sector. Conditional on the inherited coverage and enclosure premises, correctness is therefore nonconstant within every sector.

For a deterministic binary predictor of whether the network is correct, allowed to choose a separate answer for each of the five blind sectors, the minimum disagreement mass is approximately 0.101599821–0.101599841 of the whole input distribution. This is an oracle binary-correctness diagnostic. It is neither network-versus-truth error nor an accuracy-estimation MSE bound.

The true second-rank label varies inside a sign sector. Consequently, even a constant predicted class could have nonconstant correctness. These results **do not establish a predicted-class quotient obstruction** and do not complete the richer T268 conditional class-law objective. Compact conditional probabilities, finer summaries, other models and cheaper constructions remain open. No additional fixed-output or class-law test was run.

## Accounting and reproducibility

The immutable execution gate `7aca5771` was published as `4c606d21` before the separate sole `bash launch.sh` invocation. Independent source/theorem/resource gate: `a4c7046c` / shared `7c7bf23e`. The output directory deliberately prevents rerunning into the same records.

Measured internal CPU was 3.848091687 seconds, with 78,484 KiB peak RSS; external real/user/system times were 4.22/2.06/1.92 seconds. There was no failure or retry. The run made no new model, geometry-generation, Gaussian-mass, fit or RNG calls. Its selected arithmetic counters are not a native FLOP measurement.

The reused reference reports 38.605260022 seconds of composed successful cold construction/enclosure CPU, including setup and controls. This remains charged, disclosed advice; adding the new contraction does not make it a cheap deployed estimator. The reference's exact coverage/disjointness and numerical-platform enclosure premises remain inherited rather than re-proved here. No FPE, ARC-budget, held-out panel, noisy-model or T10 claim follows.

## Variants and scope

Tested: the fixed seventeen controls, every archived correct simplex, all sixteen sign sectors, and the predeclared five-sector binary loss. Untested: predicted-class distributions, finer rank or hidden-state summaries, new models, new quadrature, alternative estimators and any target-scale transfer. None is ruled out by this bounded result.
