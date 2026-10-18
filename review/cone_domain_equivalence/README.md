# T208: exact equivalence on represented cone domains

## Question
Can exact future-margin equality on a finite positive hull be decided completely, rather than only certified in some cases?

## Method
Restrict the functions to an exact basis of the cone's span, replace fixed-sign hinges by their linear/zero restrictions, and canonicalize the remainder. `PROOF.md` shows that any surviving mixed-sign hyperplane crosses the relative interior and has an isolatable gradient jump. `EDGE_CASES.md` covers zero span, lower dimension, rank one, lineality, touching hyperplanes, vector outputs and nonconvex unions.

## Status
Proof frozen after claim06d554c8; independent theorem review3d46e08f (sharedd4e754b2) passes all statements and nine edge cases. No implementation or numerical execution. Distinct from h4 T207 generic reset theorem,56 T11596 sufficient controls, and paused T190 cache claims.

## Results
[inferred, independently reviewed3d46e08f] For finite exact represented cones and homogeneous vector ReLU-plus-linear functions, zero canonical remainder **after span restriction** is necessary and sufficient for equality on the entire cone. Both-domain tests decide equality on the union without convexifying it. Mathematical full-dimensional specialization also applies to the old sufficient algebra, but code/application claims require their own verification.

Reproduction: read `PROOF.md` and substitute the nine identities in `EDGE_CASES.md`. No executable experiment is supplied or authorized by this phase.

## Compute accounting
Source/proof writing and elementary algebra only. Zero model/estimator/geometry/mass/MC/control runs, zero new reference calls. Exact basis construction and canonicalization are prospective work, not free advice or a measured speedup.

## Claims
- [inferred] The mixed-sign condition ensures an actual relative-interior crossing, not just an ambient hyperplane intersection.
- [inferred] Exact span restriction repairs the lower-dimensional linear and coincident-hyperplane gaps in an ambient-only test.
- [speculative] Such domain equalities may reduce exact enumeration work. No trained frequency or merging-cost evidence exists here.

## Alternatives not ruled out
An algebraic gap could invalidate necessity; independent review must examine rank-one and touching/lineality cases. Exact equality may be rare or costly to discover. Label-only, projective, approximate and feasible-domain variants remain open even if exact functions differ. A wrong incoming history/cone binding can invalidate a neural reuse conclusion despite a sound represented-domain theorem.

## Failure modes
No guarantee that floating ray archives cover true neural regions, no Gaussian-mass inference on lower-dimensional spans, no convex-union replacement, and no claim that different margins imply different labels or accuracy. The theorem is about one finite ReLU layer plus linear terms; no longer-horizon extension is automatic.

## Separately scoped implementation controls

Renewed scope6cba7409, sourceb47fc62f independently38442eff, binding1fc5cceb and raw12dec3ed implement the span procedure only on fixed synthetic controls. All24 pairs plus9 guards pass (19 equal/5 unequal),880 exact synthetic function calls,0.47945922CPU. The former lower-dimensional unknown is now certified. Independent saved review remains pending; no trained application follows from these controls. See implementation/README.md for source/resource/accounting scope. The proof-only phase above had zero executions.
