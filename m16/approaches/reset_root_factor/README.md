# T66 exact reset-root factorization

## Question
Do exact zero-hidden-state resets explain the residual H2 gate-piece discrepancy between long-double and mixed native circle walkers? TASKS-only claim76556aa was published and rechecked before implementation. Prior owner confirms this variant was untested. Sources remain unchanged in compiled_geometry; this folder owns the followup.

## Method
On a current affine branch h=a+b*t, if every previous hidden coefficient a,b is exactly zero from gate suppression, each next preactivation is Wi_j*(U_step+V_step*t). Cancel the common Wi factor before root division at every such step, using sign(Wi_j*V_step) for activation direction. Wi_j=0 stays off; V_step=0 uses the constant sign. This is the same algebraic cancellation previously used only at step0. No tolerance merging or reference-dependent arithmetic selection is allowed. Keep both long-double and mixed recurrence variants, and count all zero-prefix checks/reset uses.

## Frozen protocol and alternatives
Copy the prior factored/factored_mixed sources into this owned folder with source hashes. Build two separate reset variants; retain the old binaries/code for matched controls. First repeat the same21 Fraction-oracle rational/tie/scale fixtures and cap0/1 failures. Add a constructed late-reset fixture with zero affine history after a nonzero state, and verify first-index/zero-Wi/V=0 cases. Then repeat exactly32 orthonormal planes from RNG660001 on H2T10 and the original H16T10 plus relative-RMS .0025/.02 RNG1 copies. No new planes, seeds or accuracy truths. Compare outputs, piece counts per plane and reset counts for old/new long-double and mixed versions. Require integral agreement<=1e-12 with prior strict variants and oracle controls; report every count discrepancy rather than waiving the criterion.

Alternative1: all remaining H2 discrepancies come from late exact resets; factoring those roots makes paired counts equal. Alternative2: nonzero hidden affine dependencies or recurrence rounding create other coincident roots, so the repair is partial or null. A negative does not establish that double precision is invalid. Alternative3: apparently zero floating coefficients arose through cancellation, not structural gating; instrument active flags if necessary before claiming structural exactness. Cheapest discriminator is the unchanged per-plane count/output comparison and a concrete residual fixture, not a larger benchmark.

## Status
Protocol frozen before implementation. Bounded <=110CPU cycles, one thread, <=2GB, shared locks; short controls <=20CPU may use review lock.

## Results
Pending. Preserve failures and all cold/warm results. Independent T55 review requested before handoff.

## Compute accounting
Charge compilation, copied-source preparation, wrapper loading, plane/model preparation, finite checks and all kernel runs. Added zero-prefix tests cost at most2H coefficient comparisons per recomputed step and are counted explicitly. Record cold/warm CPU separately; existing arithmetic counters are partial, not a complete portable FLOP ledger. Random planes make this a sampling-based baseline primitive, not a deductive challenge solution.

## Claims
[inferred] A structurally zero previous hidden map makes all current nonzero-input-weight gates share the input root exactly. [speculative] This explains the remaining H2 finite topology discrepancies. No speed or accuracy result yet.

## Alternatives not ruled out
Compensated recurrence, general exact dependency detection and interval predicates remain untested. A finite parity result cannot certify arbitrary floating topology. No shared deployment or full challenge claim.

## Failure modes
Floating coefficients can be zero by cancellation, so a structural gate mask is the safer certificate. Tiny nonzero maps, root rounding, overflow/underflow, target ties on degenerate planes and compiler behavior remain qualified. CPU improvement alone is not equal-compute MSE improvement.

## Pre-run structural refinement
Use a cached Boolean all-gates-inactive certificate, initialized true at h0, rather than testing rounded coefficients for zero. Set the next certificate only when every gate is inactive on the represented branch. This excludes cancellation-only zeros; it does not certify the upstream floating gate topology. Charge one certificate lookup per recomputed step and H active-flag conjunctions; the previous proposed2H coefficient checks are not used. Preserve certificate values alongside cached affine coefficients on prefix reuse.
