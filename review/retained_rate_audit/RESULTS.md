# T283 numerical saved contraction, pending independent saved review

Sole execution completed after immutable60e1935e/shared077d7969, independently source-gated7f3aab62/sharedeb81643c. No failure/retry. All289 T271 and720 T277 unique rows,22 T271 groups and90 T277 model/budget cells meet frozen coverage. All1,186 required numerical field comparisons pass. Reviewer CPU0.439285972s, maxRSS61820KiB; external0.58wall/0.50user/0.07sys. No author/model/geometry/MC/RNG calls. All saved outputs and comparisons are in output/results.json.

## What reproduces

All original fit.json quantities reproduce under exact binary-input moments and independent60-digit OLS/Student-beta inversion. T277 pooled alpha by T3/4/5/6/8/10 is0.63861665/0.63506059/0.48929046/0.61409914/0.55803340/0.66701222. Spearman rho=0.02857142857; fixed-intercept s=0.32499950843, interval[0.04511388109,0.60488513577]. All three seeds/eight replicates exist at every cell; no nonpositive corrected MSE is dropped.

The T277 preregistered finite rule returns refuted because rho>−0.2. Its other branch, all alpha within0.5±0.15, is FALSE. That finite rule result does not prove dimension independence, equal/asymptotic rates, or that a prefactor is the sole cause.

## Concrete discrepancies and limits

T271's claim “CIs include0.5 on every model” is false for s0.005_1 and s0.01_0. The other three include0.5. Reproducing the historical interval formula does not certify its nominal coverage under cross-budget shared scrambles/reference errors. Nor does including0.5 establish equivalence to0.5.

All raw T277 prefactor fields are preserved. Eight of12 match the median ratios reconstructed from fit.py's MC-denominator convention; the four T3/T4 entries do not. At16k/256k:

| T | saved prefactor | reconstructed fit convention |
|---|---|---|
|3|0.0009809086334051 /0.0009932736356575|0.0009786221546860 /0.0009863155444008|
|4|0.02995517869834 /0.01900217924374|0.03003415916737 /0.01905228079861|

No producing script for prefactor.json is retained in this package. The source uses exact-summary accuracies for T3/T4 errors but MC accuracies for fit.py ratio denominators; this is a possible convention explanation, not yet an audited reconstruction of the separate file. The fixed audit deliberately records unmatched numeric fields instead of silently changing the denominator or tolerance. Its pass_all flag applies to the frozen required comparisons; it is not a statement that every prose/aggregate claim passed.

T277 enum.json is aggregate-only provenance, not an independently reproduced slope. T271 enumeration scalar fits reproduce, but free exact-mass ordering/average geometry and saved integration remain inherited premises. Source cost factor2.23 and FPE/scalar conventions are not new native/runtime measurements or current leaderboard passes.

T271 target measured ratio range is0.4333455–1.0546753. Two noisy models have all measured points above1. Results remain individual descriptive ratios, not a new six-group confidence-qualified score. No sealed HP1 inputs were read.

## Current scoring separation

The 07:01 DRI reframing introduces retrain-averaged MSE with a 1k-FPE cost floor and sealed HP2 evaluation. This audit checks historical T271/T277 conventions only. It supplies no HP1/HP2 score, retrain-population claim or new cost-floor comparison; neither sealed panel was read.
