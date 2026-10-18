# T266 exact pairing controls

## Question
Do finite exact numerical controls distinguish opposite-sign cancellation from
unchanged bias, one-sided/symmetric residuals, non-invariant laws and cost effects?

## Method
`primitive.py` contracts finite rational reversal tables and first-index labels.
`fixtures.json` contains 13 tables, six label fixtures and 12 guards with fixed
expected values. `controls.py` records every return before checking it; the
published source/resource gate and immutable execution binding precede the sole
bounded launch. `PROTOCOL.md` defines finite-law versus Gaussian qualifications.

## Status
Fresh claim `213f4b30`; sole numerical run `81149d17` complete after independently
published source and execution gates. Recover_78's saved audit remains pending.

## Results
All 13 finite-law tables, six label cases and 12 guard rejections pass; 64 progress
events retained, 0.05839 CPU seconds. See `RESULTS.md`. No empirical performance,
trained cancellation or Gaussian-law conclusion.

## Compute accounting
Planned command: `bash review/residual_pairing_controls/launch.sh`, once after gate.
Exactly 25 table attempts/13 returns, six label calls and 12 rejected tables;
37 valid table rows, 64 progress events. Whole-process 20/20 CPU seconds and 2 GiB,
17 internal CPU, 25 wall seconds after lock. No model/CDF/geometry/RNG calls.
Hypothetical single/paired prices test a formula; they are not measured runtimes.

## Claims
[inferred, prior independently reviewed] Pairing preserves bias only under the
appropriate law and reduces absolute envelopes through opposite signs.
[measured, pending independent saved audit] The fixed exact implementation matches
the declared finite controls. No claim about a trained surrogate follows.

## Alternatives not ruled out
Other surrogates, involutions and paired partitions may help. Constant optimistic
surrogates test a narrow limitation; their failure would not reject pairing.

## Failure modes
Ties, conditioning, fixed-point atoms, uncharged paired work, zero-variance ratios,
floor budgets and incorrect exact schema are explicitly exposed by controls.
Missing deployable surrogate/integral remains the trained-application prerequisite.
