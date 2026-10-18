# Source and inference contract, owner a9b09a14

This review reads text only. All numerical headline values below remain owner-reported. T244's intervention contract, T251's oblique-projection qualification and T253's retained A-phase inventory are predecessors; this package does not repeat their numerical work or extend their provenance to new C/D outputs.

## Replacement semantics

`nacommon.run` stores float64 hidden and preactivation arrays for the original recurrence. `phaseC` and `phaseC2` fit original preactivations using input-derived feature/cell tables. HingeBank learns comparator formulas from original training preactivations; held-out hinges are predictions of these formulas, not held-out true gates. Thus there is no source-level held-out true-comparator leakage through HingeBank. Hinge-bank training outputs are reused to train the second-stage formulas, rather than cross-fitted.

Validation chooses a family separately for every step and neuron. Evaluation predictions are cast to float32, then inserted into a float64 recurrent state after ReLU. Partial replacements affect subsequent un-replaced neurons. For S containing all16 neurons, however, the final assignment is exactly `h[:,j]=maximum(F[T][:,j],0)` for every j. It erases all incoming recurrent values. Consequently the all16 endpoint equals a final-step-only full replacement with the same F[T], Wo, dtype and readout operation. It tests a learned final-state formula, not independent recurrence reconstruction. This does not invalidate its measured agreement or the partial-intervention results.

v1 and v2 reuse train/validation/evaluation seeds 241031/241032/241033. v2 additionally sorts its per-neuron cumulative ordering by single-neuron agreement on the evaluation set itself. Those intermediate curves are evaluation-selected. The all16 endpoint does not depend on that order. `reportC` also uses evaluation R-squared to name the first family crossing .99, while separately reporting the validation-selected family's R-squared. These are different selection statements.

## Noise and direct-output diagnostics

`phaseC_diag` perturbs true final preactivations with independent Gaussian coordinates scaled by empirical per-neuron SD times sqrt(eps). The printed `1-R2=eps` is a nominal expected squared-error ratio under this chosen additive law; realized noise R-squared is not computed. Each eps receives a new draw. The finite curve cannot establish a universal requirement that every neuron attain R-squared .99999, and does not characterize formula errors with different covariance, gate dependence or margin dependence. The owner's narrower wording 'unstructured errors' must retain this distributional limitation.

The direct-logit diagnostic uses a separate 600k/200k train/evaluation panel (seeds241041/241042), a common Hall or A12K-Hall feature map, and a single final-step fit. It is not a controlled change of only the loss relative to the validation-selected, neuron-specific C2 composite trained on1M rows. Its post-ReLU-target alternative fits h directly but does not project fitted h back to the nonnegative orthant before readout.

For a fixed common feature map, training rows and output-independent ridge matrices, coefficient fitting and prediction are linear in the target matrix. In exact arithmetic fitting h then multiplying by Wo.T commutes with fitting h@Wo.T. The same is true of the pooled fallback because it depends only on features/cells. Numerical normal equations, separate target-product rounding, mixed per-neuron families and float32 casts require separate qualification. T259 may review this identity and alternatives; no redundant fit is proposed here.

A finite .966 direct-fit score therefore does not prove that the variable set itself imposes a .966 ceiling. It describes these feature maps, ridge rules, data and precision. The strongest untried variants remain open rather than being declared exhausted.

## D1 partition and inference

`phaseD` reads saved x/ref/p_all. Its five disjoint outcomes are: both correct; network correct/replacement wrong; network wrong/replacement correct; both wrong/same label; both wrong/different labels. Their counts must total N. Disagreement is the sum of the second, third and fifth outcomes. Accuracy gain is (third minus second)/N, not agreement gain. Network-error coverage divides (third+fifth) by all network errors; correct-answer disagreement divides the second by all network-correct rows. Both denominators and integer support must be retained.

The near/blind union uses global input gaps g12<.05 or g23<.05 or at most one positive input. These overlapping conditions are unioned once. It is a descriptive input event, not a certified neural-margin event. The source additionally reruns the original network to make all160 gate bits, then counts full-pattern frequency within the same evaluation sample (self-inclusive). Such frequency is not population cell mass or independently estimated support. The gap/pattern associations do not prove causal structure, randomness, or failure of every richer account. Argmax uses first-index output ties; task/rank labels use argsort conventions, which should be pinned in any finite replay.

Paired task-gain uncertainty should use the disjoint correctness changes, rather than independent-binomial errors. Neither a noisy estimate near the .1pp target nor a one-sided no-improvement result is two-sided equivalence. The stated0.5pp reproduction criterion must distinguish a fixed retained-model evaluation from refitting on new seeds; the latter also changes fitted coefficients and selected families.

## Retention and next evidence

The pinned owner-folder Git tree contains aggregate JSON/log/table artifacts and m16_weights.npz. The inspected C/C2 writers do not serialize fitted CellLinear/HingeBank coefficients. They write `../../../data/t241/phaseC2_eval.npz` (v1 analogous), relative to working directory, with x/ref/p_all only. This NPZ is not present in the tracked owner-folder tree. No blanket claim about untracked or other-worktree files is made. D1 pattern attribution cannot be reconstructed from those three arrays alone without retained gates or a new forward pass.

The result writers do not pin historical executable/NumPy/BLAS, checkpoint/weight digest, exact source commit or evaluation archive hash. Current source/model assertions cannot retrospectively establish these bindings. Source text advertises default ~30CPU-minute fitting and ~1.5GB peak; these are unverified reports, not permission or a safe resource estimate.

Next, separately publish a retained-JSON/log schema and arithmetic audit, plus a filesystem-metadata-only search for the named evaluation/coefficient artifacts. Preserve original records. If raw x/ref/p_all is recoverable, freeze its provenance, dtype/shape expectations and a saved-only label/partition checker before decoding. If not, report that limitation and design an independent, bounded, resource-gated reproduction; do not silently replace missing evidence with a default full script. Analytic controls must cover first-index ties, unseen-cell pooled fallback, missing age values, float32 staging, family selection and the all16 final-step identity. No new data or fit is authorized by this written package.
