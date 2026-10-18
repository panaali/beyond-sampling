# T151: held-out exact error-cone semantics

## Question
Does the independently reviewed T139 semantic point check transfer unchanged to T148's held-out H8T4 seed1 exact-error geometry? Claim d3068e6 published before implementation. Approximate recipe cells are excluded: they need not describe the original network's true errors.

## Method
Apply the SAME T139 deterministic method and six analytic controls: sum all four stored rays with coefficient1, direct independent NumPy recurrence and stable second-argmax, then64 smallest absolute truth-minus-best-other margins with any failures first and original index ties. Verify exact selected inputs from sums of binary-ray Fractions, exact recurrence/logits/gate signs/first-index predictions in16-case chunks. No new selection rule, tolerance, geometry, Gaussian mass, model noise or random input. All30726 fixed exact-mode simplices are used.

Freeze T148 commit a90dd3b archive directory `/workspace/work/recover-small_h2_910/m16/approaches/frozen_recipe_holdout/exact/`: metadata SHA9d44297aded8825a6ada3daf60e1ed4905bca3d876cb144949aa49925c202b27; error_captured.npz SHA215ce37b68c283e29224a2c720ae30c24aaf741b1269bf6bc5d0f2044b6ec9db. Seed1 public checkpoint SHA34fbb32d51cfae0d77b6cba73747506a93a72077657c4f02e243074178b75761. Copy the reviewed T139 script with only path/hash/expected-count configuration changes; save an explicit source-diff manifest before runs. No outcome-driven code change.

Alternative explanations/falsifiers: geometry semantics could fail on another seed; exact Fraction recurrence distinguishes thin-boundary double errors from true witness errors; passing witnesses do not exclude wrong behavior elsewhere inside a simplex or incomplete/overlapping error geometry. This is complementary to T148 mass/reference and T146 cost reviews.

## Status
Bounded floating and exact held-out point checks complete; independent T55 saved-record/source review add47fb passes.

## Results
All30726 fixed witnesses have the captured parent truth and an incorrect independent direct network prediction; no input ties. All64 smallest-margin witnesses pass exact Fraction recurrence with unrounded binary-ray sums, no floating/exact label changes and no exact gate zeros. Exact margin range[-.002442762399879239,-.000011686991552445548]. All six unchanged analytic controls pass. These are finite witness results, not a whole-cone or accuracy certificate.

Reproduce from repository root using `/workspace/clean-neurons/.venv/bin/python m16/approaches/heldout_error_semantics/check.py controls`, then `floating`, then `exact 0` through `exact 3`, under one-thread OMP/OPENBLAS/MKL, review lock and `ulimit -v 2500000`. source_manifest.json records all six configuration substitutions and verifies their inverse reconstructs the original reviewed source byte-for-byte. Raw inputs/logits/margins in floating.npz, exact inputs/logits/signs in four exact JSON files; summary.json is compact.

Independent reviewer add47fb checked all30726 saved labels/selection and independently reproduced all64 selected exact recurrences using integer arithmetic (.048CPU excluding load). This independently checks the finite panel; it does not expand the scope to whole cones.

## Compute accounting
One thread,2GB, review lock<=20CPU per process.30726 floating and64 exact rational trained forward passes, plus fixed analytic controls. CPU includes imports and checkpoints except final serialization/return. Measured total5.417221427CPU including controls and repeated imports, peakRSS213128KiB. All review work is separate from estimator runtime; no cost-competitive claim.

## Claims
[measured] The same frozen semantic point check passes30726 floating and64 exact held-out witnesses. No self-confirmed, whole-cone or full-rung claim.

## Alternatives not ruled out
Whole-cone defects, overlap/completeness, floating source geometry errors, other models and approximate-recipe interpretation. Passing finite witnesses cannot certify total accuracy.

## Failure modes
Rank/output ties and thin boundaries require explicit ownership and exact checking; stored binary rays themselves approximate mathematical geometry. Exact certification remains limited to the64 chosen points.
