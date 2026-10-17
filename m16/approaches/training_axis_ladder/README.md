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
First2^29 neighbor phase complete with independent review; handed to root/Claude for finite-rung classification. No lower amount evaluated. T66 scoped speedup handoff is separate.

## Results
The complete fixed2^29 panel passes the author-defined finite per-seed criterion, and independent arithmetic/geometry checks agree. Detailed results and qualifications below; no challenge-solved or unseen-model guarantee.

## Compute accounting
One thread,2GB RSS, <=110CPU cycles under codex-compute.lock; <=20CPU review/control lane uses codex-review.lock. Record every model hash, estimator operation counter, geometry/reference operations, candidate cells, reference leaves, full CPU, source setup and failures. Random weight generation and bootstrap are validation work; no validation truth or per-model fitted advice enters the frozen estimator.

## Claims
[inferred] This changes exactly one ladder axis by one training checkpoint step. [measured] The unchanged recipe passes the frozen finite-panel per-seed criterion at2^29 under inherited T75 units. Independent review checks arithmetic and a preselected extra geometry case. Only the DRI confirms a rung; population and other-training-amount claims remain open.

## Alternatives not ruled out
Margin-aware T85, different representation/re-merging, stricter impact thresholds and exact-small-support mechanisms. They are not silently substituted for the T75 recipe. Earlier model amounts remain untested until this neighbor passes.

## Failure modes
Positive-mass output ties outside the current generic trained checks, floating polygon/DD topology, uncertain full operation coverage, calibration noise dependence, bootstrap interpretation and accidental source drift. Exact/fully-resolved comparators share some primitives but use distinct geometry arrangements; native DD supplies an independent arrangement. No broad impossibility follows from a finite failed rung.

## Pilot and predecessor regression
All five2^30 clean models exactly reproduce the saved T75 outputs and operation counts. The next2^29 clean ratios are approximately0, .0061, .0018, .0293 and .0182; this is only a pilot, not a passed noisy rung. Original/strict polygon truth, the fully resolved boundary recipe and independent native DD agree within the frozen1e−12 tolerance on every specified clean/first-noisy control, with unit total DD mass. Exact public checkpoint hashes and frozen source hashes are saved in `manifest.json` and `pilot.json`. No recipe changed.

The full panel remains the previously frozen20 blocks of256 noisy copies. The first4 RNG20 copies overlap the resource pilot and enter the final statistical panel only once; repeated pilot computation is validation overhead, not independent evidence. Block files preserve every estimate, exact truth, operation/cell counts, model hash and CPU. Blocks checkpoint independently; no lower training amount is tested before complete review.

Pilot accounting:1.835CPU/199MB; predicted candidate+truth time5.53CPU per256-copy block. Maximum independent-reference difference4.45e−16. One individual pilot copy has ratio1.228, so the pilot is not described as every-copy success. The predeclared criterion concerns each training seed's aggregate noisy MSE/CI plus its clean case; all individual failures remain in the raw panel and worst-case summary. This distinction is fixed before full evaluation and matches the predecessor's aggregate protocol.

## Complete2^29 panel: author result, independent review pending
All5,120 predeclared noisy models were evaluated,1024 per training seed, with no omitted cases. The frozen per-seed aggregate criterion passes for all five seeds under the inherited T75 operation convention. This is a finite-panel result awaiting DRI confirmation, not a claim that every individual model wins.

| training seed | noisy RMSE | published MSE ratio | conditional95% bootstrap CI | mean candidate ops | candidate cells |
|---|---:|---:|---:|---:|---:|
|0|0.0001123|0.0000592|[0.0000519,0.0000674]|34481.7|170.0|
|1|0.0051411|0.04771|[0.03464,0.06311]|25207.3|103.0|
|2|0.0012678|0.003648|[0.002868,0.004731]|31708.6|157.2|
|3|0.0018642|0.009184|[0.004518,0.01555]|30507.4|144.8|
|4|0.0148706|0.30358|[0.25365,0.35643]|23262.0|109.8|

The per-model-matched ratios are0.0000591,.04742,.003641,.009137,.30222, consistent with the primary conclusion. Clean ratios remain below.030. Individual failures are explicit:0/11/0/2/97 noisy copies have ratio>1; the worst copy has8.084pp absolute bias and ratio8.57. Thus this is not uniform accuracy dominance over the weight distribution.

Independent T55 statistical review distinguishes the published ratio, which uses one observed ensemble-average budget, from the per-model matched comparison. The primary bootstrap holds that mean budget fixed, so its CI is conditional on it; the model-matched bootstrap resamples individual costs. Five separate95% intervals are not a simultaneous95% family certificate or a theorem over unseen training amounts. Noise draws are paired across training seeds by design; no pooled-seed confidence argument is made.

Reference correct-piece counts are75–90 on average and are labeled accordingly. Independent global DD gate counts on the five clean models are70,52,53,70,72; they are not interchangeable with rank-conditioned correct leaves or candidate half-chamber cells. The method still resolves much of a small model and does not establish favorable asymptotic scaling.

Validation block CPU totals112.917 seconds across five bounded cycles, each below26CPU; pilot1.835CPU, summary.060CPU. These block totals exclude Python imports before each timer and network acquisition, which are not per-model estimator advice and are not silently presented as complete campaign cold cost. Candidate/reference timings and all operation counters are retained per model. Reproduce with the pinned source checkout and downloaded hashes: `python pilot.py`, `python block.py --training-seed s` for s0..4 under the resource lock, then `python summarize.py`. No2^28 model has been fetched or evaluated; advancement awaits independent review and a separate rung claim.

## Independent handoff
T55 reviewe49308c independently recomputes the primary, model-matched and integer-budget ratios from all5120 raw rows using Decimal arithmetic, agreeing within4.89e−15 relative. It verifies unique model hashes and independently preserves the individual-loss counts. Before opening the selected author row, it reconstructed seed4/noise23/copy255 and summed separate native-DD triangles with60-digit areas: p=.60181227642865760011 and total mass=.99999999999999999572, agreeing with polygon truth within1.00e−16 and the exact same model hash. This supplements, not replaces, the source/statistical review and15 original reference controls. See [independent review](../feynman_fundamentals/training_neighbor_review.md). Rung advancement remains a DRI/board action; no next checkpoint is inferred safe from this finite result.
