# T151: held-out exact error-cone semantics

## Question
Does the independently reviewed T139 semantic point check transfer unchanged to T148's held-out H8T4 seed1 exact-error geometry? Claim d3068e6 published before implementation. Approximate recipe cells are excluded: they need not describe the original network's true errors.

## Method
Apply the SAME T139 deterministic method and six analytic controls: sum all four stored rays with coefficient1, direct independent NumPy recurrence and stable second-argmax, then64 smallest absolute truth-minus-best-other margins with any failures first and original index ties. Verify exact selected inputs from sums of binary-ray Fractions, exact recurrence/logits/gate signs/first-index predictions in16-case chunks. No new selection rule, tolerance, geometry, Gaussian mass, model noise or random input. All30726 fixed exact-mode simplices are used.

Freeze T148 commit a90dd3b archive directory `/workspace/work/recover-small_h2_910/m16/approaches/frozen_recipe_holdout/exact/`: metadata SHA9d44297aded8825a6ada3daf60e1ed4905bca3d876cb144949aa49925c202b27; error_captured.npz SHA215ce37b68c283e29224a2c720ae30c24aaf741b1269bf6bc5d0f2044b6ec9db. Seed1 public checkpoint SHA34fbb32d51cfae0d77b6cba73747506a93a72077657c4f02e243074178b75761. Copy the reviewed T139 script with only path/hash/expected-count configuration changes; save an explicit source-diff manifest before runs. No outcome-driven code change.

Alternative explanations/falsifiers: geometry semantics could fail on another seed; exact Fraction recurrence distinguishes thin-boundary double errors from true witness errors; passing witnesses do not exclude wrong behavior elsewhere inside a simplex or incomplete/overlapping error geometry. This is complementary to T148 mass/reference and T146 cost reviews.

## Status
Protocol frozen; configured script and controls next.

## Results
None yet.

## Compute accounting
One thread,2GB, review lock<=20CPU per process.30726 floating and64 exact rational trained forward passes, plus fixed analytic controls. CPU includes imports and checkpoints except final serialization/return. All review work is separate from estimator runtime; no cost-competitive claim.

## Claims
[speculative] The fixed held-out point semantics will match the captured exact-error interpretation. No self-confirmed or whole-cone claim.

## Alternatives not ruled out
Whole-cone defects, overlap/completeness, floating source geometry errors, other models and approximate-recipe interpretation. Passing finite witnesses cannot certify total accuracy.

## Failure modes
Rank/output ties and thin boundaries require explicit ownership and exact checking; stored binary rays themselves approximate mathematical geometry. Exact certification remains limited to the64 chosen points.
