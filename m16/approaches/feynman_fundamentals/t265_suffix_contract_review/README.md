# T265 independent suffix-success contract review

## Question
Is the proposed weak conditional-success state law mathematically sound, and do the old source references support its limited reuse plan?

## Method
Read THEORY/PLAN/SOURCE_INVENTORY and eight exact text snapshots at4e2f2dbd under assignment0e95e67d. Independently bind all snapshots to their specified037f7857 Git bytes and hashes. No producer imports, archived states, probability routine, model input or CDF evaluations.

## Status
Written/source contract passes. execution_allowed=false. No implementation, trained comparison, numerical probability or resource clearance is given.

## Results
[inferred law] For fixed weights and2<=t<=T, h_t plus the two largest prefix values and their distinct valid prefix indices suffice to continue the recurrence and update the eventual second-largest index. Independent fresh iid suffix inputs therefore define a measurable transition kernel q_t; on actual prefix states it represents conditional correctness. Integrating against the JOINT state/rank pushforward gives accuracy. Separate marginals do not determine this integral. Supplied states outside the reachable support define an extension, not additional observations.

[inferred last-input construction] At most H nonzero-slope gate roots divide the line. Each cell yields affine logits; adding every nonconstant pairwise difference root and the two rank roots makes output/target constant on each open interval. Zero-slope gates and identically equal readout rows must retain permanent first-index ownership. Isolated endpoints have zero mass under the continuous final Gaussian. The target labels are i_b below b, last-index between b and a, and i_a above a. Thus the finite sum of matching Gaussian interval masses is exact in real arithmetic. H+2+(H+1)*T(T-1)/2 bounds distinct finite breakpoints by overcounting; it is not a complete price. Floating roots, CDFs and near ties are not certified by this theorem.

[inferred weak equality] In the supplied Wi=0,Wh=I toy, h=e1 andh=e2 predict constants0 and1. At rank state(A,-A,0,1), correctness sets y>A andy<-A are disjoint yet have equal Gaussian mass Phi(-A). At rank state(2,-1,0,1), crossing a positive h1=h2 tie changes the mass between Phi(-2) andPhi(-1), with label0 owning equality. Both are extension-only examples: zero initialization withWi=0 cannot reach those hidden states. They show neither trained compression nor a discontinuity under an actual prefix distribution.

[inferred approximation] A pointwise error envelope integrates to an accuracy-error bound under the same joint law. Conditional expectation onto a fixed summary minimizes L2 for that law, but learning/integrating it is not free. Equal or close finite q values do not certify between-point or population error. Signed cancellation can make small integrated bias compatible with substantial local errors; it cannot be presumed. Global future-margin nonmatches do not imply distinct q values.

[inferred source scope] The frozen T188 handoff is the small recipe-capture phase, with incoming feasibility/stored-versus-exact limitations; no exact true-prefix partition is obtained merely by reusing its maps. T127 MiddleTarget is specifically H2, chronology randomx/fixedu/randomy, truncated support and penultimate label except diagnostics; its conditioned normalization is not the new unconditioned H4 last-input law. T67's fixed-target-before-random-input chronology and cancellation/refinement failure remain separate. T132 continuity requires its nonzero gate/readout slopes; the Wi=0 extension example lies outside them. Snapshot prose results are inherited citations, not numerically revalidated here.

[measured metadata] Eight source snapshots and five written files match exact Git bytes and SHA/size. The concrete candidate summary retains full top-two rank state plus current target margin; its sufficiency/continuity/utility remains explicitly unproved. Prefix comparisons must be fixed before observing reference q, certified feasible or labeled point/extension diagnostics, and charged in full. Later validators must require valid distinct indices in0..t-1, strict ranks, finite dimensions and explicit tie handling, not merely a>b.

## Compute accounting
Text/Git metadata/hash work only. Zero source imports, trained-array reads, model forwards, RNG, CDF, LP/geometry or probability calculations. Command: `python m16/approaches/feynman_fundamentals/t265_suffix_contract_review/check_metadata.py` (immutable output).

## Claims
[inferred] Conditional-law, partition and extension examples are sound with stated premises. [measured metadata] Text bindings pass. No cheaper representation, actual reachable weak equivalence, sampling comparison or empirical numerical conclusion.

## Alternatives not ruled out
Richer summaries, controlled interpolation, signed corrections and direct integration of the coupled law remain open. Reference construction may cost as much as the original method; finite summary collisions can be hard to find. A point-only failure cannot reject a broad summary family.

## Failure modes
Confusing extended and reachable states, losing hidden/rank coupling, ignoring permanent ties, transferring truncated H2 kernels, uncertified floating roots/CDFs or treating reference-table setup as free would invalidate the intended conclusion. The next phase must separately freeze controls, source/resource scope and provenance before any calls.
