# T114: independent training-axis review

## Question
Does E44 survive fresh weight noise and an independent reference/cost check at its weakest reported point and at the least-trained rung? Claim1e3e8a1 was pushed before implementation. Reviewer previously authored T90 at k29; that is a different rung and is not reused as independent evidence here. E44/T101 was implemented by Claude.

## Method
Frozen before new outputs:
- Absolute sigma .05: k24 seed4, noise seeds120/121,128 copies each; k20 seeds0..4, noise seeds120/121,32 copies each. Clean counterparts included separately.
- Relative noise: k20..24, allfive training seeds, noise seeds122/123,32 copies each. Use one scalar sigma(k,s)=.05*RMS(all32 weights at k,s)/RMS(all32 weights at k30,s), so relative perturbation is matched to the original fully-trained absolute panel. This is global RMS, not per-array RMS; do not tune after outcomes. Clean counterparts separately.
- T75 frozen E37 recipe on every copy. F2L3 frozen L3 with independent R4 cache fix on first8 copies of each noise seed per cell (16/cell), all clean cells. This bounded smaller fibre panel cannot independently confirm every E44 fibre tail.
- Each phase checkpoints per-model estimates, errors, sampling variance, counters and failures. Fixed panels may be divided into <=110CPU blocks; no favorable block selection. One numerical thread,2GB, primary compute lock; short controls use review lock<=20CPU.
- Write a separate exact T3 geometry reference using ordered-pair charts, local spherical polygon clipping and explicit first-index zero-margin ownership. Cross-check clean and first-copy outputs against the author's exact.py before panel scoring. No MC model inputs; noisy model weights are validation only.
- Reexecute the preexisting independent R3 scalar recipe and counting type on the new models, checking agreement with frozen T75. Disclose reuse of R3 implementation; this is independent of E44's estimator/source count but not a wholly new recipe implementation. Count model-dependent scalar arithmetic/comparisons on both estimator and deterministic sample reference; audit unary negation, integer powers, setup and output correctness comparison. Preserve author's counter alongside independent recount; do not silently multiply all costs by a fitted factor. F2L3 uses R4 strict counters with source-path inspection and independent controls; any missing audit remains qualified.
- Aggregate separately by k, seed, noise kind, recipe. Primary sampling denominator is mean[p_i(1-p_i)*cost_sample/cost_i], using per-model own cost. Also report E44's mean-cost convention to isolate differences. Bootstrap over fixed noisy copies with fixed seed991, report clean and individual failures separately, and retain covariance across common-noise rungs. No DRI confirmation by this worker.

Competing explanations: (1) E44's weakest point passes only because of reused noise or mean-cost denominator; fresh independent samples and both denominators discriminate. (2) Low-training panels look easy only because absolute noise dominates small weights; anchored relative noise tests this without adapting estimator. (3) A numerical tie/topology defect masquerades as bias; independent reference and constant/duplicate readout controls precede conclusions.

## Status
Protocol frozen; source/control implementation next. No results yet.

## Results
None yet.

## Compute accounting
All source setup/import, geometry, estimator/reference and validation CPU recorded; model downloads hashed separately. Prior R3/R4 cost frameworks explicitly attributed. Per-model results preserve both claimed and independently executed counts. Shared evaluator and estimator sources remain unchanged. No claim of equal-wall-time advantage.

## Claims
[speculative] E44's scoped average-operation advantage persists on this fresh fixed panel.

## Alternatives not ruled out
Other training seeds, per-layer relative perturbations, adversarial weight changes, smaller sigma, much larger fibre panels and certified floating geometry. Failure of one panel/implementation does not retire all mechanism-derived estimators.

## Failure modes
Low-noise tails, small positive-volume output ties, thin cones, model-specific cost correlation and underpowered fibre subsets may change apparent advantage. Positive results apply only to tested cells and predeclared conventions.
