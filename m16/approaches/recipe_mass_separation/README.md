# Fixed recipe approximation versus numerical mass (T145)

## Question
After repairing the demonstrated finite Gaussian mass bias, how much error remains from the unchanged recipe108 gate approximation on H8,T4 seed0?

## Method
Pin T131 model and estT snapshot. Run its original recipe108 with eps=.1, replacing only terminal Gaussian mass calls by an error-simplex recording stub. Neural gate decisions and sequential output-error clipping remain unchanged. Capture all96 initial halves; cap55CPU/200M named geometry operations and retain completed-parent checkpoints. Approximate recipe cells need not represent true-network error events: that difference is the approximation under test, not automatically a geometry defect. Record all-off cells and coefficient-identical readout rows; unresolved positive-volume tie concerns block interpretation.

On the captured rays, replay (a) unchanged wide12/power2 and thin48/power3 mass, (b) the single frozen T140 uniform doubling to24/96, and (c) independent positive radial Duffy integration, selecting the first order20/32/48 with conditional per-simplex bound≤1e-9 divided by error-simplex count. Same determinant cutoff/threshold in(a,b), no clipping/renormalization. Eight fixed12-parent batches, each≤55CPU,1thread/2GB/primary lock; preserve every negative, exception, unmet bound and partial. Source original total must reproduce final3 recipe .9752542405196474; otherwise investigate provenance before attribution.

Compare positive recipe accuracy against T131 corrected exact-reference .9761608075466027; compare original/doubled masses against this same recipe geometry's positive value. Report signed and per-cone absolute numerical errors, preventing cancellation from masquerading as accuracy. Orthant and coordinate-order Gaussian controls precede replay; all-off first-index controls precede capture. No parameter tuning, new MC, other checkpoints, sharedsource edits or passed-rung claim.

Alternative1: numerical quadrature dominates the old recipe discrepancy (same-ray replacement separates it). Alternative2: freezing gates creates genuine approximation bias (positive recipe versus positive exact). Alternative3: stale source/geometry/tie semantics explain discrepancies (hashes, cell counts, original reproduction and tie audit). Independent reviewers examine source and saved-array attribution; direct true-network membership is not presumed for approximate cells.

## Status
Bounded decomposition complete; independent review d7e0d5e passes. Claim a99881b preceded protocol85de4b3 and implementation.

## Results
Results below. Reproduce capture.py, replay.py BATCH for0..7, summarize.py using project interpreter and one-thread resource locks.

## Compute accounting
Capture, import/rule setup, all original/doubled/positive work, exact radial certificates and checkpoints count. Existing T131 exact-reference and T140 research costs remain dependencies, never free production effort. Inherited named counters are partial proxies; T146 separately owns completecost inventory. No sampling win follows solely from these counts.

## Claims
[inferred] Same-ray independent integration separates numerical mass bias from the captured recipe approximation's discrepancy against exact geometry. [speculative] The corrected recipe may retain useful accuracy at lower cost; this is not established.

## Alternatives not ruled out
Other eps values, gate certificates, coarser families, conditional targets, improved integration and new representations. This tests exactly one prior recipe on one checkpoint, not an exhaustive negative or general success.

## Failure modes
Floating geometry, approximate gate ownership, output ties, endpoint quadrature, cancellation, failed source reproduction, determinant cutoff, unknown cost and conditional—not full-arithmetic—error bounds. Retain failures and incomplete coverage.

## Bounded decomposition result
All96 parents complete, matching final3's6182 recipe leaves and capturing28000 error simplices. No all-off cells or coefficient-identical active readout rows occur. Original accuracy **.9752542405196474** and original full named counter201163614 reproduce the published values exactly.

| Calculation | Accuracy |
|---|---:|
| Original recipe mass | .9752542405196474 |
| Fixed doubled-rule recipe | .9761655721035335 |
| Independent positive mass on recipe geometry | .976160677036335 |
| Independent positive exact-geometry reference | .9761608075466027 |

The captured recipe approximation's accuracy difference is **−1.30510267704e-7**, whereas original numerical mass error contributes **−.000906436516688**. The numerical contribution is about6945 times larger on this fixed checkpoint. The combined conditional integration bound for the recipe-minus-exact comparison is2.13788310235e-10, excluding floating geometry and arithmetic. This is a measured difference of captured floating calculations, not a real-network exactness theorem.

The doubled-rule recipe's total discrepancy is **+4.76455693077e-6**. Its own numerical discrepancy against positive recipe is+4.89506719847e-6; thus the small approximation and numerical terms have opposite signs. Per-cone absolute numerical error sum improves .00185465768823→.0000465761282012; maximum1.74487669674e-7. Original2038 negative masses and candidate503 negative masses are retained, candidate signed negative sum−4.79192081950e-6. No positivity/conditional error certificate is claimed for the doubled rule.

Positive recipe integration uses19951 GL20 and8049 GL32 cones, total conditional bound1.09566649490e-10, no target misses. Capture14.537045709CPU plus all three replay methods146.221977482CPU are charged, in addition to earlier exact reference dependence. Source geometry named count36398526 is reconstructed by subtracting recording/control charges and adding initial gate comparisons; source/doubled mass counts164765088/325162176 yield original/doubled full partial counters201163614/361560702. These are inherited named counts, not portable totals; T146 separately audits their kernel/setup omissions.

[measured] On H8,T4 seed0, the unchanged recipe approximation is already much more accurate than the original apparent discrepancy suggested; finite mass integration dominates that discrepancy. [inferred] The original numerical failure is not evidence that this recipe's approximation necessarily fails in the target regime. No conclusion about all seeds/noise, other rungs, floating topology certification or sampling competitiveness follows yet. T55 independent source/array/decomposition review requested. A fixed held-out checkpoint is the next generalization discriminator; no parameters are changed from this calculation.

Independent T55 audit d7e0d5e verifies all28000 indices, source hashes, exact sums of stored binary64 values, approximation/numerical decomposition and original/doubled named-counter identities. No integration rerun is claimed. T145 handed off; frozen held-out seed1 T148 remains a separate validation, with full-panel T149 blocked on that independent handoff and T146 cost evidence.
