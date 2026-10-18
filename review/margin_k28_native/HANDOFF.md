# T115 k28 direct cost audit

The fixed H4,T3 training level n_seqs=2^28 has complete direct source-cost
records for five clean models and 5,120 original noisy specifications, for both
unchanged RECIPE85B and RECIPE112. All original error subtractions and saved
whole-seed mean counters reproduce exactly. Five current canonical checkpoint
objects are verified; original historical checkpoint byte identity remains
unproved because those hashes were never saved.

## Results

Matched integer budgets use each candidate's own numeric count and111 units per
sampling evaluation. Ratios below1 favor the candidate. Bootstrap intervals
are pointwise, conditional on the original saved truths and cases.

| Seed | 85B ratio | 85B individual losses | 112 ratio | 112 individual losses |
|---|---:|---:|---:|---:|
| 0 | .000343698583 | 0 | .0000135438519 | 0 |
| 1 | .049524944236 | 2 | .000000306827 | 0 |
| 2 | .000160620391 | 0 | 4.18788696e-13 | 0 |
| 3 | .021632332766 | 0 | .007337726845 | 0 |
| 4 | .157332412408 | 23 | .002530982626 | 1 |

All ten seed-average bootstrap uppers are below1, maximum .174976031224.
All forty noise-block averages and ten clean comparisons are below1. Worst
noise-block ratios are .172192638400/.008944324488 for85B/112; worst clean
ratios .164730588475/.009198795375. This is not uniform per-copy success:
85B has25 noisy losses, worst2.844071868295;112 hasone, worst1.749361779078.
Historical72 counters give6/1 noisy losses and remain separate. The summary
also preserves continuous own-budget and uniform mean-budget comparisons.

## Coverage and cost

All80 shards,5,120 unique current model hashes,10,240 paired records and10,220
new baseline records reconcile. The ten noisy controls, first64 pilot and five
clean controls enter once. Every raw site/category sum is retained. Exact
original mean legacy counts and cell counts match for every seed/recipe;
individual historical noisy counts never existed and are not invented.

Noisy numerical totals are574,097,097/796,386,386 (85B/112), with
17,966,722/27,007,147 integer expressions separate. Clean numerical totals
551,596/785,894 and integer totals17,285/26,872 are separate.

Recorded controls+pilot+continuation CPU:2,081.222020532, below the frozen3,000
resource envelope; maximum process31.417634027CPU, peak212,556KiB. Exactly
20,500 baseline/traced estimator calls across5,125 models/two recipes are
research work; no new reference draws, model-input MC, training or noise law.
Saved summary8.155768824CPU; provenance, network and independent review work
remain separately documented. Five checkpoint payloads/10,745bytes andsix
HEAD requests include the preserved repeated-header parser failure/repair.

## Verification and integration

Owned folder: `review/margin_k28_native/`. Source gate9235759d; all numerical
records completec9abf7a8; full saved summary8d7a52c0. Canonical/current-model
reviewf51a6d34, controls05cd1e39, pilot38378124, seeds0..3
ac75ff27/5663c685/84e0040c/4f7f070c. Final independent seed4/global reviewfb8a2443 passes all raw coverage, clean
ledgers, counters and metrics.
Only the owned folder is to be integrated; never merge this old research
branch wholesale. Claude remains responsible for INDEX/rung classification.

## Limits and remaining possibilities

This is a declared numerical-source convention, not a complete native,
binary, control, memory, setup or runtime cost certificate. Pointwise
bootstrap does not cover saved-truth numerical error, historical checkpoint
identity or population generalization. It is H4,T3 training-axis evidence,
not H4,T4 Hopf evidence or an M16,T10 solution.

The observed individual losses remain fixed-case limitations, not a proof the
recipe family cannot work. No accuracy-driven variant, precision change or
allocation was tried in this replay. Formal/native cost closure and any
mechanistic repair require separately frozen controls and a new scope.
Historical byte identity cannot be recovered from matching errors alone.
