# Independent T255 saved scalar audit

## Question
Do the saved 300-block MC summary and capped-prefix descriptive calculations reproduce the fixed raw scalar records?

## Method
Explicit delegation 74c79d2f. `check.py` independently computes integer hit totals and the raw sum of squared hits, derives exact rational binomial and block-mean variances, and compares 100-digit Decimal roots with the saved 80-digit values. Fixed source floating-output tolerances remain unchanged. It verifies all block IDs/seeds/counts, the sole float32 block, six progress events, source/review gates and scalar k5/k6 contractions. No author code is imported or called.

## Status
Checker and scope frozen before reading the completed numerical results or block records; pending independent source/resource gate. No numerical reviewer execution yet.

## Results
None yet. This follows the completed written-only review in the parent folder, without duplicating it or upgrading the capped sampling design.

## Compute accounting
Proposed single 19 soft / 20 hard CPU second, 2 GiB, 25 post-lock wall second review. All hash, parsing and integer/Decimal computation is reviewer cost. No array, checkpoint, model, geometry, quadrature, MC or RNG calls. No new truth or confidence interval is computed. Earlier metadata/source reading is separate from this planned saved calculation.

## Claims
- [inferred] Integer raw moments provide a distinct check of the author's centered Fraction variance calculation.
- [inferred] Correct scalar arithmetic cannot establish unbiased capped-subtree selection or historical model/runtime identity.

## Alternatives not ruled out
Historical execution may match the frozen source despite incomplete records; MC pseudorandom stream diagnostics may be informative without proving mathematical independence. Neither possibility is rejected by this audit.

## Failure modes
Missing/duplicate blocks, incorrect field types, resume/source mismatch, reported variance drift and wrong denominator can invalidate the arithmetic. Every failed reviewer attempt will be retained; no automatic retries. Prefix mass and coverage remain numerical scalar metadata, not independently integrated or decoded arrays.
