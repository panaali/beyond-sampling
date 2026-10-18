# T245 rank-adaptive next-history upper bound

## Question
Can the per-parent affine-arrangement upper bound depend on exact hidden-map rank instead of elapsed input dimension?

## Method
Existing Ready proof-only task claimed in `050ebb65` after independently complete T244. The planning idea was supplied/yielded by recover_34, who independently reviews the written proof in `review/t245_rank_bound_review/`. `PROOF.md` works with exact full-dimensional parent cones and their nonnegative hidden images. `EDGE_CASES.md` preserves eight analytic domain/sharpness examples. `manifest.json` freezes four inherited proof/review texts, including T236's dimension-only bound. No executable candidate or control suite is introduced.

## Status
Frozen written proposal awaiting independent review. No numerical or trained-rank claim.

## Results
Proposed upper bound: for parent map rank r>0, at most sum_(j<=min(r,H)) binom(H,j) positive-volume next gate words. For rank zero, at most two (one if all input weights vanish). The positive hidden-image section has dimension r−1; adjoining the new scalar makes an r-dimensional affine arrangement. Boundary-only words are excluded.

There is no experiment reproduction command: all results are written deductions. The provenance operations are `git show <manifest.shared_snapshot>:<path>` and Git blob/SHA256 comparisons, without importing source or decoding data.

## Compute accounting
Text/Git reads, byte copies/hashes and written algebra only. Zero exact or floating rank calculations, matrices or arrays decoded, network calls, clipping/enumeration, samples, mass integrals, fits or numerical controls. An eventual parent-rank census and its preprocessing would require a separately claimed scope and complete source accounting.

## Claims
- [inferred, pending review] The section argument yields the proposed per-parent and aggregate upper bounds under exact domain/rank assumptions.
- [inferred] Positive state/time-coordinate reparameterizations preserve corresponding ranks, without a general Gaussian-mass invariance.
- [speculative] Actual low-rank trained parent images might sharpen a count explanation. No rank distribution or trained/random difference was measured.

## Alternatives not ruled out
Ranks may usually be full; geometry rather than rank may determine realized counts; labels or states may aggregate across many histories; probability integration can dominate. None is tested or excluded. New actionable owner-priority evidence takes precedence over any optional extension.

## Failure modes
Counting boundary faces as open histories, replacing exact rank by numerical thresholds without qualification, treating incomplete or approximate domains as exhaustive, using signed states without a positive section, or inferring an algorithmic lower bound from an upper bound would invalidate the claimed interpretation. Relative-domain restrictions require their own image dimension and probability scope.
