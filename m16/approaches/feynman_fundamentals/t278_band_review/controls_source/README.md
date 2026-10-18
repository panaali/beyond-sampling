# T278 fixed primitive source/resource gate

## Question
Are the finite exact interval and Gaussian-displacement controls ready for one bounded run before a separately reviewed numerical model application?

## Method
Written algebra, source inspection and byte/AST metadata checks only. The seven direct files are intervals.py, gaussian.py, controls.py, controls_launch.sh, controls_protocol.json, PLAN.md and CONTROLS.md. No producer imports, controls, CDF evaluations or target payload decodes occur in this review.

The exact partition recursively propagates affine functions of delta, splitting at every strict interior preactivation root. A rational midpoint determines each open subinterval mask; identically-zero gates stay OFF. The frozen path uses masks evaluated at delta=0 and continues the resulting affine masked classifier, which may have negative hidden coordinates away from the face. That is the stated surrogate, not a claim that those gates remain realizable.

For each target label, all pairwise affine margins define its interval of success. A constant zero margin against an earlier label rejects success over positive length; strict versus nonstrict nonconstant-root boundaries differ only on conditional null points. The merge routine can discard isolated endpoint distinctions; therefore the error intervals are an almost-everywhere representation, not a complete first-index label function at every boundary. Direct exact logit probes retain continuity at gate endpoints. The controls exclude ordinary error endpoints from interval-membership assertions and explicitly test positive-volume zero-logit ties.

Signed difference partitions the union of endpoints and subtracts membership indicators, yielding disjoint nonzero-sign bands. The four literal cases test overlap subtraction, both residual signs on a negative halfline, and alternating signs. Future production must split bands at zero before the Gaussian helper; its guard explicitly rejects crossings.

## Gaussian algebra
With z=sqrt(2)*delta, the delta density is psi(c)=exp(-c^2)/sqrt(pi). For nearer-zero endpoint c, outward signed displacement d and orientation q=sign(d), positive band mass is q times the CDF increment. Its first two terms are q*psi(c)*d and q*psi(c)*(d-c*d^2). The implemented bounds follow from sup|phi'|=phi(1) and sup|phi''|=phi(0): phi(1)*d^2 and |d|^3/(3*sqrt(pi)). Negative-halfline orientation is correct. Stable erfc differences on one side and erf across zero implement the delta law directly. These are ideal-real Taylor bounds; the fixed3e-15 slack compares floating diagnostics, not certified outward CDF values. Negative second-order values on wide bands remain unmodified.

## Status
Source/resource review passed for source c2f4534e/shared28cde6b7 under explicit delegation35a5a2e2. A published source_gate.json and separately published immutable producer controls_gate.json are required before the sole run. Target application remains unauthorized by this gate.

## Results
[inferred] The four network fixtures' literal error intervals match their source recurrences: zero output uses first index everywhere; sign and shifted-sign readouts cross at0 and1/2; the recurrent fixture crosses at0. Four fixed-mask setup recurrences and28 separate masked probes check the frozen affine maps, alongside28 direct ReLU probes. This is source reasoning, not a replay of the planned controls.
[inferred] Fixed coverage is18 rows and38 events: four networks, four differences, four Gaussian bands and six guards; eight valid partition calls,28 direct plus28 masked recurrences and four mask setups. Guards make three additional partition entries, one error entry and two correction entries. The counters do not conflate these with eight valid calls.

## Compute accounting
One review-lock process,17 internal CPU seconds,19 soft/20 hard CPU seconds,2GiB address space,25 wall seconds after the lock and one numerical-library thread. The launcher sets soft limits before lowering hard limits. Protocol/gate/reviewer hashes and resolved executable are checked before producer primitive imports; runtime asserts exact CPU/address-space limits. Fresh output directory and noclobber log prevent overwriting prior attempts. Row inputs/call counters/returned values are atomically retained before assertions, and failed attempts remain incomplete. No automatic retry.

Named rational counters and returned numerator/denominator maxima exclude comparison/sort internals, GCD intermediates, encoding, memory and control work. Process CPU includes them; no native/FPE total is established.

## Claims
[inferred] Clearance covers only the specified finite synthetic suite. It does not validate arbitrary malformed inputs, target model weights, face strata/window laws, Gaussian table normalization, outer quadrature or performance. The planned exact partition has a10000-piece cap; exceeding it preserves failure rather than silently truncating. Statistical output-boundary ownership remains a null-set qualification under the continuous fibre law.

## Alternatives not ruled out
Target topology/cost may be much larger than these networks. Finite face maps and outer quadrature may dominate inner Taylor error. Full band discovery may cost more than direct integration. Richer/cheaper residual discovery remains open regardless of these controls.

## Failure modes
Floating CDF/kernel errors are not outward-enclosed. Shared helper logic does not make a synthetic pass a target correctness proof. Separate saved-output auditing and a target source/resource gate remain required before model application. No experimental or numerical assertion was independently replayed here.
