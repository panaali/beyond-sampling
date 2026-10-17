# Fixed recipe approximation versus numerical mass (T145)

## Question
After repairing the demonstrated finite Gaussian mass bias, how much error remains from the unchanged recipe108 gate approximation on H8,T4 seed0?

## Method
Pin T131 model and estT snapshot. Run its original recipe108 with eps=.1, replacing only terminal Gaussian mass calls by an error-simplex recording stub. Neural gate decisions and sequential output-error clipping remain unchanged. Capture all96 initial halves; cap55CPU/200M named geometry operations and retain completed-parent checkpoints. Approximate recipe cells need not represent true-network error events: that difference is the approximation under test, not automatically a geometry defect. Record all-off cells and coefficient-identical readout rows; unresolved positive-volume tie concerns block interpretation.

On the captured rays, replay (a) unchanged wide12/power2 and thin48/power3 mass, (b) the single frozen T140 uniform doubling to24/96, and (c) independent positive radial Duffy integration, selecting the first order20/32/48 with conditional per-simplex bound≤1e-9 divided by error-simplex count. Same determinant cutoff/threshold in(a,b), no clipping/renormalization. Eight fixed12-parent batches, each≤55CPU,1thread/2GB/primary lock; preserve every negative, exception, unmet bound and partial. Source original total must reproduce final3 recipe .9752542405196474; otherwise investigate provenance before attribution.

Compare positive recipe accuracy against T131 corrected exact-reference .9761608075466027; compare original/doubled masses against this same recipe geometry's positive value. Report signed and per-cone absolute numerical errors, preventing cancellation from masquerading as accuracy. Orthant and coordinate-order Gaussian controls precede replay; all-off first-index controls precede capture. No parameter tuning, new MC, other checkpoints, sharedsource edits or passed-rung claim.

Alternative1: numerical quadrature dominates the old recipe discrepancy (same-ray replacement separates it). Alternative2: freezing gates creates genuine approximation bias (positive recipe versus positive exact). Alternative3: stale source/geometry/tie semantics explain discrepancies (hashes, cell counts, original reproduction and tie audit). Independent reviewers examine source and saved-array attribution; direct true-network membership is not presumed for approximate cells.

## Status
Claim a99881b pushed before implementation; protocol frozen.

## Results
Pending. Reproduce capture.py, replay.py BATCH for0..7, summarize.py using project interpreter and one-thread resource locks.

## Compute accounting
Capture, import/rule setup, all original/doubled/positive work, exact radial certificates and checkpoints count. Existing T131 exact-reference and T140 research costs remain dependencies, never free production effort. Inherited named counters are partial proxies; T146 separately owns completecost inventory. No sampling win follows solely from these counts.

## Claims
[inferred] Same-ray independent integration separates numerical mass bias from the captured recipe approximation's discrepancy against exact geometry. [speculative] The corrected recipe may retain useful accuracy at lower cost; this is not established.

## Alternatives not ruled out
Other eps values, gate certificates, coarser families, conditional targets, improved integration and new representations. This tests exactly one prior recipe on one checkpoint, not an exhaustive negative or general success.

## Failure modes
Floating geometry, approximate gate ownership, output ties, endpoint quadrature, cancellation, failed source reproduction, determinant cutoff, unknown cost and conditional—not full-arithmetic—error bounds. Retain failures and incomplete coverage.
