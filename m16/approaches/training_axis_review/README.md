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

### Initial controls and retained defects
`controls.py` under the20CPU review lock took2.607CPU excluding initial imports. Twelve clean/first-copy trained controls (k24seed4 and k20seeds0..4) match author's truth within2.22e-16, and R3 scalar reimplementation agrees with frozen T75 within4.44e-16. Local counter adds unary negation and the sampling correctness comparison; nonnegative integer powers use repeated-multiplication cost. No estimator is altered. Model-independent cached geometry remains separate from these model-dependent counts and does not become magically free in a cold executable.

Three deterministic fixtures expose inherited failures: zero input weights yield exact accuracy1/3, but both T75 and author's exact.py return1.0; zero/identical output rows give1/3 while T75 returns.9976141 and author truth.9983942. New independent reference and F2L3 return1/3 within5e-16 on all three. Preserve these failures. They are outside the predeclared E44 trained/noisy panel and do not by themselves refute its scoped claim. Frozen estimators remain unchanged; all panel references come from the independent implementation, with author-reference discrepancies recorded.

Independent T55 semantic review: mean[p_i(1-p_i)/n_i] compares equal per-model costs, while mean[p_i(1-p_i)]/mean[n_i] is a distinct uniform-allocation comparison at mean cost. Both are useful but not interchangeable. Anchored global RMS perturbations preserve relative parameter noise across training rungs; they are not function-invariant under neuron rescaling. Absolute and relative results will remain separate.

### Frozen cap telemetry follow-up
Read-only F2L3 source inspection found two64-step tracing loops and a400000-trace emergency branch. The completed absolute panels reached at most552 traces/model, but that does not prove individual trace loops never exhaust. Freeze a separate telemetry replay of every previously scored F2L3 clean/noisy model: shadow only the module's `range(64)` with an identical sequence whose natural exhaustion increments a counter; explicit breaks retain their semantics. Compare every estimate and strict counter with its saved original row, and report all natural exhaustions. No estimator threshold/cap changes or favorable model selection. This tests a possible silent-truncation failure; it does not constitute a fresh full scalar arithmetic audit of F2L3.

### Initial panel completion and independent confirmation freeze
All31 initial cells completed:2,176 noisy rows plus31 clean rows,160.324 recorded phase CPU. Maximum trained reference disagreement5.56e-16; maximum scalar-reimplementation disagreement2.45e-15; no failed trained checks. Relative k24seed4 has ratio.75974 on64 copies with pointwise CI[.38564,1.17149], so its direction is unresolved statistically. Other relative cells have T75 ratios<=.03282. All fixed fibre subsets and clean cases have ratios<1, with small-subset and inherited-counter qualifications.

TASKS scope6c34141 was pushed before new runs: ONE independent confirmation panel for relative k24seed4, noise seeds124/125 with256 copies each, unchanged RMS scale and frozen estimators/reference/counters. Fibre uses first8 copies of each seed. Report separately from the initial cell selected for uncertainty; do not pool for a prespecified headline or expand again based on the outcome. This is a finite confirmation design, not an optional stopping rule.

All527 original fibre evaluations were replayed with identical output and strict counts. No64-step trace loop exhausted; no400000-trace emergency threshold approached. Telemetry replay42.421CPU. This verifies that silent loop caps did not cause the reported finite-panel estimates; it is not a proof about all models or a fresh full arithmetic audit.
