# T90: training-amount neighbor of the passing M4T3 recipe

## Question
Does the unchanged T75 recipe pass the next training-amount rung, M4T3 trained on2^29 sequences, after its established2^30 result? Claimcecab6e was pushed before implementation. Only the training amount changes; H4,T3, five training seeds and the noise protocol stay fixed.

## Method
Use the frozen `est75.RECIPE = {K:(4,4), rule:'adaptive', eps:0.1}` from the shared source snapshot9dec2e5. Import it read-only from the isolated admin-t90-claim checkout, record source hashes, and assert the recipe. Do not use the later margin-aware T85 rule or fit any constants. Download the exact public zoo checkpoints for training seeds0..4 at2^29 and2^30 into ignoreddata, recording URLs and hashes. No new training or input sampling.

Truth is the original independent spherical-polygon enumerator `arc_m43_repro/exact.py`, with its floating-geometry qualifications. Compare clean and first-noisy-copy references to the separately implemented native DD T3 engine from T66; verify partition mass and fully resolved boundary recipe (`rule='impact', K=(4,4)`) on these controls. All shared source stays unchanged. This is a deductive candidate, with random weight perturbations solely for validation; no random model inputs are used.

## Frozen alternatives and falsifiers
1. The impact rule reflects robust computation and should transfer to the adjacent training amount; alternatively earlier checkpoints may have important small/cancelling gates that its scale proxy freezes incorrectly. Evaluate every training seed, preserve bias, worst errors and resolved/cutting-gate counts.
2. An apparent failure may be a reference/integration or implementation mismatch. Before interpreting it, compare polygon truth with independent DD geometry and the fully resolved recipe; use fixed refinement/identity controls, not extra Monte Carlo.
3. A favorable average could hide seed-specific failures or cost variation. Require each seed's clean ratio<1 and each seed's noisy aggregate ratio and upper95% bootstrap endpoint<1. Report per-noise-seed panels and a per-model-matched sampling variant separately. No averaging across training seeds to hide a failure.
4. Operation conventions or setup exclusions may create an apparent win. Preserve T75's published72-op forward convention and strict primitive counter, report exact-region/candidate-cell counts and CPU alongside it, and label these inherited units rather than a newly audited portable FLOP certificate. Preparation, truth and independent-check work are validation costs, never free advice to the estimator.

## Frozen execution
First controls/pilot: all five clean checkpoints at both training amounts; for the new2^29 amount, first4 copies from absolute-weight-noise sigma.05 RNG20 per training seed. Independent DD and fully resolved recipe checks on each clean and each first noisy copy. Compare old2^30 clean outputs to saved T75 regression results. These resource/control pilots cannot establish a passed rung.

Before seeing pilot accuracy, freeze the full new-rung panel: noise seeds20,21,22,23, each256 copies for each of five training seeds, with the same Wi/Wh/Wo draw order as zoo43.noisy_copies. If the pilot is numerically sound and projects under110CPU seconds per256-copy block, run all20 blocks, checkpointing each. If slower, subdivide the same panel; never select fewer/favorable copies. Bootstrap500 resamples per training seed with RNG0, exactly the T75 report convention. Also report mean(err²)/mean(72*p*(1-p)/ops), preserving the published ratio using mean ops separately. Full panel planning is fixed; no lower amount2^28 is authorized by a successful pilot alone. If any complete rung criterion fails, stop the ladder, independently diagnose the new behavior and record plausible rescues before changing the recipe. Re-running every earlier rung is mandatory after a future recipe change.

## Status
Protocol frozen. Checkpoint acquisition and dependency controls next. T66 scoped speedup handoff is separate.

## Results
Pending. No passed-rung or challenge-solved claim.

## Compute accounting
One thread,2GB RSS, <=110CPU cycles under codex-compute.lock; <=20CPU review/control lane uses codex-review.lock. Record every model hash, estimator operation counter, geometry/reference operations, candidate cells, reference leaves, full CPU, source setup and failures. Random weight generation and bootstrap are validation work; no validation truth or per-model fitted advice enters the frozen estimator.

## Claims
[inferred] This changes exactly one ladder axis by one training checkpoint step. [speculative] The unchanged recipe remains competitive for all five seeds at2^29. Only the DRI confirms a rung after independent checks.

## Alternatives not ruled out
Margin-aware T85, different representation/re-merging, stricter impact thresholds and exact-small-support mechanisms. They are not silently substituted for the T75 recipe. Earlier model amounts remain untested until this neighbor passes.

## Failure modes
Positive-mass output ties outside the current generic trained checks, floating polygon/DD topology, uncertain full operation coverage, calibration noise dependence, bootstrap interpretation and accidental source drift. Exact/fully-resolved comparators share some primitives but use distinct geometry arrangements; native DD supplies an independent arrangement. No broad impossibility follows from a finite failed rung.

## Pilot and predecessor regression
All five2^30 clean models exactly reproduce the saved T75 outputs and operation counts. The next2^29 clean ratios are approximately0, .0061, .0018, .0293 and .0182; this is only a pilot, not a passed noisy rung. Original/strict polygon truth, the fully resolved boundary recipe and independent native DD agree within the frozen1e−12 tolerance on every specified clean/first-noisy control, with unit total DD mass. Exact public checkpoint hashes and frozen source hashes are saved in `manifest.json` and `pilot.json`. No recipe changed.

The full panel remains the previously frozen20 blocks of256 noisy copies. The first4 RNG20 copies overlap the resource pilot and enter the final statistical panel only once; repeated pilot computation is validation overhead, not independent evidence. Block files preserve every estimate, exact truth, operation/cell counts, model hash and CPU. Blocks checkpoint independently; no lower training amount is tested before complete review.

Pilot accounting:1.835CPU/199MB; predicted candidate+truth time5.53CPU per256-copy block. Maximum independent-reference difference4.45e−16. One individual pilot copy has ratio1.228, so the pilot is not described as every-copy success. The predeclared criterion concerns each training seed's aggregate noisy MSE/CI plus its clean case; all individual failures remain in the raw panel and worst-case summary. This distinction is fixed before full evaluation and matches the predecessor's aggregate protocol.
