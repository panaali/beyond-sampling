# Independent fixed reconstruction saved audit

## Question
Do the fixed train/two-holdout archives and coefficients support the reported scores without refitting or rerunning the RNN?

## Method
Delegation4226647e. Checker frozen before outcomes. Verify191 new chunks, pilot reuse once, complete source/archive/RNG/progress chains; independently construct features, three-term lookup and sparse two-term separable predictions, label/disjoint-rank tables, scalar-fsum hidden metrics and beta-CDF bisection intervals. Structural fit metadata and a numerical normal-equation residual diagnostic do not re-fit or factorize the design.

## Status
Complete independent saved audit results_v2.json. Author experimentcd272a57 ran once; initial checker30ff6469/05a34918 source-reviewed2d269c77. Nearzero-R2 reviewer criterion failure preserved9ad3ae47; recovery97d65169 source-reviewed e8b80edc and boundfa5c1c6f before saved-only recovery. No author replay.

## Results
All191 new chunks, one pilot reuse,196608 observations,393 progress records,217 source-recorded fits and34 beta calls pass. Independent predictions reproduce all saved labels and disjoint tables. SSE/SST/MSE retain1e-10 relative tolerance with1e-12 scale floor; CI endpoint tolerance2e-12. Maximum scaled normal-equation residual7.244820596790121e-15. Stored R2 and median must exactly equal their stored-total formulas; independent-fsum differences up to1.5216716775512396e-12 are retained. Original relative-to-R2 criterion failures remain recorded for separable neuron4 in train and holdout2. These are reviewer-derived-metric conditioning limitations, not author refits or experiment failures.

## Compute accounting
One19/20CPU review-lock process,2GiB virtual cap and one thread. Predictions, linear readouts, scoring, hashing and numerical CDF inversions are reviewer research work. Zero refits/SVD/RNN/library-reference/model-checkpoint loads/normal draws. CPU excludes imports; external cap includes them. No full-cost comparison.

## Claims
[inferred] A passing saved audit supports this new fixed-fit experiment only; historical98%/90–91% and summary sufficiency remain separate.

## Alternatives not ruled out
Prediction operation ordering may affect low bits/argmax; exact label mismatches must be retained. Numerical fit rank and terminal RNG advances remain source-gated. No simultaneous intervals or training-variation inference.

## Failure modes
This does not verify neural forwards beyond the independent1024 wiring pilot, or prove exact least-squares optimality. It cannot certify the general reconstruction family or an ARC rung.

## Checked heldout interpretation

| Panel | Network task accuracy | Lookup task accuracy | Lookup network agreement | Lookup median hidden R2 | Case2 separable task accuracy |
|---|---:|---:|---:|---:|---:|
| holdout1 | .9525909424 | .9799194336 | .9530944824 | .9448801722 | .9012735822 |
| holdout2 | .9536285400 | .9797668457 | .9539337158 | .9454614487 | .8987821844 |

All denominators and individual numerical CP intervals were independently contracted. The lookup's high task accuracy is distinct from reproducing the network. The source summary already determines the correct task label, so this result does not establish network reconstruction or a deductive accuracy estimator. Original historical splits remain unvalidated.

Author internalCPU7.467979396; reviewer failed phase.807868393 and recovery2.251893928, combined3.059762321CPU excluding imports. Both reviewer attempts and exact source versions remain published. Zero reviewer refits/SVD/RNN/reference/normal draws.

Audit command after published recovery gate: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 bash -c 'ulimit -v 2097152; ulimit -t 20; flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/t201_checkpoint_pilot_review/fullfit_saved/check.py'`. Outputs immutable; do not rerun.
