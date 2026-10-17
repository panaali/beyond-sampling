# T37: hidden-reference gate-remainder ceiling

## Question
Can an exact handcrafted hidden-state reference support a first-order approximation of the trained recurrence, despite the previously rejected independent OUTPUT-residual law?

## Method
T37 was claimed and pushed8a44a93, TASKS.md only, before implementation. Preserve the old `reference_error` results. Use the actual22-state handcrafted second-argmax reference in alg_zoo/handcrafted.py, and the original16-state trainedT10 checkpoint with the same two DEV relative-RMS perturbations .0025/.02, RNG1 copy0. All sources are local and already cited by the previous T37 memo; no new external ARC source is read.

Fit one22-to16 linear map B on a separate calibration pool, minimizing hidden squared error over all10 recurrent steps. This is an optimistic calibration-only diagnostic, explicitly not weight-derived and not a production estimator. Use the unregularized minimum-norm solution from numpy.linalg.lstsq with rcond=None, recording rank/singular values and cost. B may produce signed reference states; the exact error identity does not require their positivity. No positive alignment assumption, model-specific ridge tuning or validation-selected alignment variant is permitted.

Let r_t be the exact reference state, hbar_t=B*r_t, z_t=W*hbar_t+w*x_t, D_t=diag(z_t>0), and hlin_0=0. Define hlin_(t+1)=relu(z_t)+D_t*W*(hlin_t-hbar_t). For the actual model h, let v_t=W*(h_t-hbar_t), kappa_t=relu(z_t+v_t)-relu(z_t)-D_t*v_t. Then kappa>=0, kappa<=abs(v)*1[abs(z)<=abs(v)], and h_(t+1)-hlin_(t+1)=D_t*W*(h_t-hlin_t)+kappa_t. Readout differences use the actual Wo; no assumption that Wo*B equals the reference readout.

## Frozen alternatives and falsifiers
1. The earlier output-residual independence failure might hide a useful pathwise linear hidden approximation. Compare actual, aligned-reference and first-order labels/accuracy on exactly the same held-out inputs. This variant retains full reference/input dependence and never shuffles residuals.
2. A large hidden error might be harmless after readout, or small errors might reverse tiny margins. Report hidden RMS, centered-logit RMS, label disagreement, target-vs-best-rival margin errors, and the propagated gate-crossing remainder identity separately.
3. A final nonlinear gate may dominate an otherwise useful approximation. Predeclare one rescue: keep the same first-order state through step9, then perform the actual nonlinear final update. This changes no fit and is scored regardless of first-order outcome. It cannot repair all earlier omitted gate crossings.
4. Alignment capacity may be the bottleneck. Report calibration/held-out residual RMS and singular values. Do not claim all hidden references fail if this one map fails; time-dependent/nonlinear alignment and explicit gate mixtures remain open, with different representation and cost.

## Frozen numerical protocol
Use4096 iidN(0,1) calibration sequences (seed370201) and32768 disjoint validation sequences (seed370202), length10. Reuse the two input pools across the two DEV models, but fit B separately on calibration only. Stream recurrent calculations in chunks<=4096; no arrays larger than2GB. Exact reference states are shared between models where possible; count actual evaluations and duplicate work honestly.

For each approximate rule report paired bias and SE relative to actual correctness. The diagnostic tolerance is .001: |bias|-3SE>.001 rejects this rule; |bias|+3SE<=.001 advances this finite ceiling only; otherwise inconclusive, with no automatic sample increase or new tuning. This is a descriptive paired normal interval, conditional on the fitted B, not an exact simultaneous guarantee or a leaderboard MSE ratio. Both rules, both DEV models and every failure are retained.

Before trained evaluation, independently verify the gate-remainder algebra and exact first-index ties on deterministic controls. An identity alignment of a network to itself must give kappa=0 and exact first-order trajectories. A trained recurrence with W=0 must also be exact for any reference B. Check the reference final readout against its independently known closed form g_i=x_i^+ +2max(0,x_j:j!=i), including all-negative/one-positive ties; source comments contain a sign typo, so actual assignments govern. Verify the propagated kappa sum equals actual-minus-linear state/logit differences within explicit floating tolerance. No full-model success claim until these controls pass and an independent reviewer checks the result.

## Status
Protocol frozen; controls and bounded diagnostic next. T55 theory/independent review requested. No production estimator or shared code changes.

## Results
Pending. Exact handcrafted reference accuracy is .9904405381944444; previous output-independent residual laws missed trained accuracy by22.56–48.68 percentage points. Those failures do not settle this distinct pathwise hidden-reference approximation.

## Compute accounting
One thread, <=2GB, ulimit-v2500000; shared numerical lock, <=110CPU cycles (review lane<=20CPU). Count actual trained/reference recurrences, alignment matrix products/least-squares calls, first-order/rescue recurrences and validation arithmetic. Report CPU/RSS and explicit operation inventory; do not call training/calibration or residual decomposition free. Initial intended cycle<=20CPU. These sampled oracle diagnostics are neither deductive estimates nor new high-precision truth calculations.

## Claims
[inferred] The gate-remainder identity is exact for arbitrary fixedB with first-index output ties. [speculative] A useful low-error first-order hidden-reference rule exists. No evidence yet that a scalable weight-derived alignment or probability calculation exists.

## Alternatives not ruled out
Time-dependent/nonlinear alignment, a different reference network, conditionally analytic crossing corrections, explicit gate mixtures and richer joint state laws. T80 owns cumulant variants and is not duplicated here. No follow-up is silently implemented or fitted after observing validation.

## Failure modes
Reference capacity/alignment mismatch; large accumulated gate crossings; unstable recurrent products; target-margin sensitivity; sampling uncertainty; least-squares ill-conditioning; an empirical oracle diagnostic being misrepresented as mechanistic success; ignoring full fitting/propagation costs.

## First frozen result
Both fitted alignments have rank22. On held-out sequences, the pathwise first-order rule is substantially more faithful than the aligned reference alone, but all rules reject the .001 bias ceiling:

| DEV noise | Actual accuracy | Aligned reference | First order | Exact final-step rescue |
|---|---:|---:|---:|---:|
| .0025 | .9399414 | .5220947 | .8674622 | .8774414 |
| .02 | .6241760 | .4451599 | .6016846 | .6033325 |

First-order paired biases are -.0724792(SE.0016890) and -.0224915(SE.0017843). The predeclared final-step rescue leaves -.0625000(SE.0015633) and -.0208435(SE.0016306). No approximate rule advances. The held-out hidden RMS itself is12.8605/12.7122; first-order state-error RMS is only .5821/.6217, yet label disagreements are10.81%/16.03%. Centered-logit error RMS4.607/5.102 and target-margin error RMS6.217/6.836 show why hidden MSE alone is misleading.

The accumulated gate-crossing identity holds within2.51e-13 in hidden coordinates and1.09e-12 at readout; kappa nonnegativity/upper-bound violations are only7.11e-15 floating residuals. Crossings occur in roughly2.8–5.3% of neuron-input pairs per later step, with kappa RMS rising to .43/.45 in middle steps. These facts distinguish the omitted nonlinear remainder from an implementation identity error. T55 independently checked the algebra and is checking the recurrence via a different residual-state formulation.

Total first cycle1.5345CPU seconds, peak338,988KB RSS, including torch and model construction. Calibration least-squares uses .0320/.0240CPU. It uses73,728 trained and36,864 reference trajectory evaluations, plus deterministic controls; these are trajectory counts, not an end-to-end FPE score. The held-out diagnostic also performs five16x16 dense products and two22-to16 products per step, one extra final-step16x16 product, five terminal16-to10 products, and an extra terminal22-to16 product. Calibration adds two40960x22 least-squares solves with16 right-hand sides and their residual products. Elementwise gate/remainder checks, sorting, RNG, statistics and reference construction are included in CPU; no portable complete scalar-operation total or estimator efficiency claim is asserted. The full JSON retains both B matrices, singular values, seeds and every rule.

## Frozen support-preserving rescue (separate second phase)
A first-order expansion can produce negative hidden coordinates, violating the trained ReLU state's exact support. Before evaluating this new variant, freeze hclip_(t+1)=max(relu(z_t)+D_t W(hclip_t-hbar_t),0), hclip_0=0, using the already calibrated B. It repairs surrogate deactivations when a reference-active coordinate crosses below zero; it still cannot activate a reference-inactive coordinate and therefore is not the exact full recurrence. This cheap structural correction is distinct from fitting a residual law or simply rerunning the full model. Compare original first-order and clipped first-order on a fresh32768-sequence pool, seed370203, both DEV copies. Record pre-clipping negative fractions and paired accuracy/error; no new alignment or tolerance selection. Apply the same .001 descriptive bias gate; no automatic larger sample or new fitted parameter. Preserve the first phase unchanged. A zero result is not failure of all support-aware or nonlinear reference methods.
