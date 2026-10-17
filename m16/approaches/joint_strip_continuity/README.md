# T147 joint-Gaussian strip continuity bound

## Question
Can joint input geometry replace the unusable final-y-only continuity bound from T132? Claim a368b72 preceded implementation; T55 reviewed the finite affine-strip proof before trained application.

## Method
Chronology h -> x -> fixed target u -> y for an H2 RNN. For each of16 first/middle gate words, derive the final preactivation z=A*x+B*y+C*h+E*u. Include its two final-gate boundary forms, then each nonempty final support and each target-versus-competitor readout margin. First-index ties are unchanged; permanent support-identical margins do not define a moving boundary. A zero joint input normal with nonzero hidden coefficient is unknown, not a continuity result. Hidden-independent forms contribute zero.

Join two hidden states by a straight path. If correctness changes, the path crosses some nontrivial readout-margin or final-support boundary. Earlier ReLU gates are continuous; if one adjoining margin is identically zero, the nonconstant adjoining margin or a final-support boundary records the change. On a fixed word, extend the affine boundary to the entire hidden path. A crossing forces opposite endpoint signs of this extension, except fixed zero-measure input hyperplanes. The corresponding Gaussian strip has width at most ||C||1*||delta h||inf / sqrt(A^2+B^2), so mass is at most that width/sqrt(2pi). Union over all words is safe even when most words/facets are unreachable. No gate-probability differentiation is required.

Use the fully rational conservative K=sum ||C||1/max(abs(A),abs(B)), with Gaussian projection density<1. Freeze both literal word sum and exact proportional-boundary deduplication (include the x,y,h,u coefficients in the key); both are implemented and reported. This only removes duplicate identical strips, not hypothetical geometry work. Divide by the actual joint rank-domain probability: L^2, 2LU, U^2 for need0/1/2, where L=Phi(u)-Phi(-6), U=Phi(6)-Phi(u). Individual side01/10 uses LU. Conditioning on the truncated Gaussian square cancels its normalization. For u=1, also report elementary certified denominators L>1/3, U>1/27 (integrate densities on[-1,1] and[1,2], using sqrt(2pi)<3 and e<3); thus certified constants9K,(81/2)K,729K. Floating CDF displays are separate. Need1 is the physically relevant branch for the saved +.25 prefix/u=1 case; the other rank-domain constants describe conditional kernels, not additional claimed physical cases.

Freeze exact primitive controls before trained code: memoryless Wh=0 and constant-output models give0; a moving one-dimensional Gaussian threshold has the expected strip coefficient; Wi=(1,1), Wh=diag(1,2), target readout(1,-1), other rows0 is a y-degenerate but joint-nondegenerate candidate rescue; replacing Wh by identity must return unknown for a zero input-normal/nonzero-hidden margin. Test proportional deduplication invariance under scaling/sign and unique boundaries. Before trained evaluation, also freeze a matched ablation using the same exact per-word hidden coefficients and proportional deduplication but only abs(B) as denominator. This separates the joint-normal contribution from tighter hidden derivatives and duplicate removal; zero B with nonzero hidden coefficient remains unknown in that ablation. Then load only the existing H2T4 seed0 checkpoint from T132 and compare constants; no new hidden/input probe, interpolation grid, geometry, MC or accuracy run.

Competing explanations: (1) huge bound reflects a truly sharp value function; joint-normal bound may remain huge. (2) conditioning only on y creates spurious nearzero denominators; joint normals may remove them. (3) hidden-dependent constant-input margins permit jumps; unknown controls must detect them. A smaller global bound alone does not validate a practical interpolation grid.

## Status
Protocol frozen; exact primitive implementation next.

## Results
Pending.

## Compute accounting
Only finite exact matrix/boundary algebra, one checkpoint load and scalar CDF displays. Count emitted/unique/unknown forms and CPU. Zero trained-model forward passes/random inputs;880 exact synthetic prescribed-gate map evaluations are counted separately. One thread,2GB, review lock <=20CPU. Proof coefficient is not an operation-count estimate.

## Claims
[inferred, proof reviewed] A finite union of affine Gaussian strips gives the stated sufficient continuity criterion. [speculative] Its exact joint-input coefficient is substantially smaller for the trained pair kernel.

## Alternatives not ruled out
Facet-restricted integration, reachable-word pruning, local state-dependent density bounds, adaptive interpolation, other terminal representations and exact integration. A still-large coefficient would not exclude these alternatives.

## Failure modes
Omitting final-support/positive-volume tie transitions, discarding zero joint normals with hidden dependence, counting only actual endpoint gate words, dividing need1 by LU rather than2LU, or treating an improved continuity coefficient as an accuracy/cost result. Exact coefficient arithmetic does not certify inherited floating kernels.

Primitive phase: all stated controls pass; the joint-rescue toy has literal K=84 and unique K=12, while the earlier final-y criterion is unknown.880 exact noncommuting forced-gate coefficient checks pass. Initial controls .012000041CPU, extended controls .035985731CPU; both retained in accounting, no failed controls. Trained application waits independent source review.
