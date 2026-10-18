# Independent T245 rank-sensitive history bound review

## Question
Does the effective hidden-map rank, rather than the whole past-input dimension,
bound the number of positive-volume next-step gate histories?

## Method
Delegated claim050ebb65. Written review of frozen proof2ec6aa59 and clarification94e2fc83, their analytic
edge cases and inherited T236/T219 text. Metadata verification binds Git blobs;
no author code, matrix rank, cone enumeration or target data is evaluated.

## Status
The main theorem passes in its exact-real full-dimensional-parent scope.
The final artifact records the reviewed edge-case clarification94e2fc83. No execution
or trained rank/cost inference is authorized.

## Results
For a full-dimensional parent, A maps its open interior onto a relatively open
convex subset J of im(A). A nonzero-dimensional J contained in the nonnegative
orthant cannot contain zero, since relative openness would include a nonzero
vector and its negative. Thus sum(h_i)>0 throughout J. Its unit section has
dimension r-1; adjoining the normalized fresh scalar input gives a convex open
r-dimensional domain. Positive normalization preserves all new gate signs.

Nonzero gate zero sets cannot contain an open child; identically-zero gates
remain OFF. Realized strict sign regions are convex, and lifting a nonempty
open section region gives an open set in the original parent cylinder. Thus
positive-volume child words are counted by chambers of at most H affine
hyperplanes on this domain. The deletion/restriction recurrence bounds those
chambers by sum_{j<=min(H,r)} binom(H,j). Restricting to a convex domain cannot
split one sign chamber into disconnected pieces.

For r=0 only the sign of the fresh scalar can vary, giving at most two words
when w is nonzero, and one when w=0. Boundary-only all-OFF words do not alter
this ambient-volume count. Summing parentwise bounds is valid for distinct
complete histories, even if some parents have equal hidden maps. Exact ranks
are invariant under the stated invertible left/right coordinate changes; this
does not preserve Gaussian masses under arbitrary input rescaling.

The written reset, zero-form, rank-one sharpness, rank-two crossing, coincident
plane, restricted-parent and kernel-lineality cases agree with these premises.
For the signed-image counterexample, a width-preserving explicit embedding is
h=(x,0), W rows both (1,0), w=(1,-1), C=R. It has rank one and four plane
quadrants, so dropping nonnegativity really can violate B(2,1)=3. The initial
scalar-hidden shorthand was requested to be clarified to avoid changing width
at the same time. No numeric control is necessary for this algebraic example.

## Compute accounting
Written reasoning, source-text reads and Git/hash metadata only. No matrix,
rank, history, geometry, model, checkpoint, array, mass, RNG or fit calls.

## Claims
[inferred] The upper bound and aggregate consequence hold under the stated
exact-domain assumptions. [measured] Bound predecessor text matches Git bytes.
No actual rank deficiency, chamber reduction, target-growth rate, mass coverage
or algorithmic-cost advantage is observed. A smaller upper bound need not
reduce the realized count, and a full-rank image can still have few chambers.

## Alternatives not ruled out
Domain-specific restrictions, output-label equivalence and aggregation may
reduce work independently of rank. Future rank measurements require complete
exact maps/domains and their own scope, not floating-rank thresholds or witnesses.

## Failure modes
Counting boundary words, using matrix rank for an unexamined lower-dimensional
parent, dropping nonnegative-image premises, treating distinct histories as
minimal affine regions, or interpreting an upper bound as necessary work.
