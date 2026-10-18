# T258 fixed reproduction resource pilot

## Question

Can the independently implemented final-step replacement pipeline bind the original model, retain its complete fitted dependencies, and execute a small fixed pilot within the declared resources?

## Method

`run.py` uses the previously reviewed feature and regression helpers. After exact converted-weight identity and a 1,024-input comparison with retained library outputs, it generates three fixed streams: 4,096 training, 1,024 validation and 1,024 evaluation inputs. Nine comparator-bank fits precede twelve final candidate fits. Validation selects each neuron's family; both all-twelve and the fixed no-hinge-eight selection are evaluated. All inputs, model coefficients, cell routing, pooled fallbacks, predictions and selection scores are retained.

## Status

One numerical run completed. Independent saved audit is pending; no further fits or full campaign are authorized by this result. Source review `104fa975`, published `d58d76b8`, preceded immutable execution gate `b802b79f`, published `31ab2d3c`. Only then was `bash launch.sh` invoked. Raw commit `8f3eb7fb` preserves the sole run.

## Results

[measured, pending saved audit] All 21 fitted objects and 13,370 linear solves returned. Exact converted model weights matched the independently reviewed canonical payload. On the 1,024 old wiring inputs, maximum absolute hidden/logit differences were approximately 4.98e-14/2.97e-13; all first-index output labels agreed.

The independently held-out evaluation panel has 1,024 inputs. Original network correctness is 980/1,024. The all-twelve replacement agrees with the original on 924 and is correct on 945. The fixed no-hinge-eight replacement agrees on 975 and is correct on 1,003. These are separate targets: agreement with the original network and correctness against the input-order task. No confidence interval or historical full-reproduction criterion is applied to this small resource pilot.

Exact reproduction command for the once-only run was `bash launch.sh` in this folder. It refuses existing outputs; do not rerun it over retained evidence.

## Compute accounting

73.060523548 internal CPU seconds; Bash timing 73.74 wall, 64.98 user and 8.20 system seconds. Peak RSS 149,128 KiB. The process stayed within 105/110 CPU seconds and 2 GiB; no retry occurred. The fit-stage total, approximately 70.04 CPU seconds, includes per-solve observer persistence and is not bare solver time.

Seven forward calls cover 7,168 input vectors, including the retained-reference wiring check. Three new random-generation calls produce 61,440 Gaussian scalar values. There are 63 feature batches, 21 fits, 63 prediction batches, 12 validation-score calls, one selection, six endpoint assemblies and six metric calls. All 13,370 attempted linear solves returned. There are 168 permanent NPZ archives totaling 128,398,818 bytes; metadata, logs and last-solve crash witnesses are additional. No new Torch/library reference call occurred.

## Claims

- [measured, pending saved audit] This fixed pilot completed and retained deployable dependency data for all 21 fit objects.
- [inferred] The small pilot supports feasibility of this bounded implementation, not a runtime projection to one million training inputs. Cell counts, conditioning, observer overhead and retained-array sizes change with sample size.
- [speculative] Larger training data may improve agreement; this pilot neither establishes nor rules out that outcome.

## Alternatives not ruled out

Sampling and fit variability, sparse-cell overfitting, validation selection, numerical conditioning and dependency differences can explain disagreement with historical scalar summaries. The no-hinge result does not justify dropping the other families after observing this panel. A separately frozen full reproduction, if the DRI chooses it, must reuse these stream prefixes once and retain its independent evaluation policy.

## Failure modes

There was no numerical failure in this pilot. The earlier regression-control publication-order violation remains preserved in its own phase and is not retroactively cleared. The source-publication automatic-review rejection and its exact same-repository blob resolution are preserved in `PUBLICATION_REVIEW.md`. No method failure, ARC cost win, confidence claim, architecture ceiling or full reproduction follows from this resource run.
