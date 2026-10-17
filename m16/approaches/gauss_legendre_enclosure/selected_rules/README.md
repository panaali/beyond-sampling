# T11592 fixed selected-rule constants

## Question
Can the missing universal constants be bounded tightly enough to preserve the already selected ideal-rule certificates?

## Method
Claim647c4682. Freeze new GL orders12/22/28/30 only; existing6 constants reused. Same reviewed exact Legendre/Sturm160-bit root brackets and128-bit weight enclosure. Pre-run34 review caught the original explicit order guard would reject new orders; no calls had run. Local exact_extended.py extends only that list, with full byte parity after reversal enforced. Original exact.py remains immutable. One source leggauss call per new order; no source mass/rank/model calls. After independent saved certificate audit, apply unchanged original4D perturbation bound on only11 already fixed split-parent selections and add pinned ideal-tail upper bounds. Selection is never changed from outcomes.

## Status
All four fixed node chunks complete03b3af71 after narrow source gate. Independently reviewed ed1fbdf5; combined-bound source gate pending. Four independent durable chunks at20CPU each, max80CPU phase.

## Results
All92 nodes/weights and184 moment checks pass, no unresolved brackets. Total recorded CPU6.317148656, four model-free source-generation calls. Independent closed Rodrigues audit ed1fbdf5 validates all enclosures and source distances without bisection or leggauss replay. Fixed moment inclusion is a consistency check, not by itself an integrand certificate. Combined11 selected-rule bounds remain pending.

## Compute accounting
One thread2GB/review lock. Universal exact setup and source-generation entry calls retained; PRECOMPUTE_FLOPS null because native/hardware work is unmeasured. No ranked entry. Named categories/reduced retained widths exclude some comparison/control/GCD/runtime and transient integers.

## Claims
[inferred] Reviewed root/weight theorem applies to these fixed even orders. [speculative] Existing tail margin will exceed the newly bounded constant contribution.

## Alternatives not ruled out
If global bounds remain loose, narrower enclosures or local sensitivities might help; no adaptive change is allowed in this phase. Evaluation rounding and generic geometry/scaling remain outside even a successful tail-plus-constant bound.

## Failure modes
Accidental order reselection, mismatched source rule/model-independent constants, silently altering precision, losing failed chunks, or conflating ideal-plus-constant scope with full source arithmetic.
