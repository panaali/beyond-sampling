# T198 source review of pinned T195

Author commit `6a2a54036bb0bbd08ede419025f1e69eb5284b6f`; no execution of
model, geometry, mass, attribution, clock or reference routines in this review.
Saved equality and independent arithmetic checks are a separate phase.

## What the source measures
`attr.py::_record` assigns the exact chamber rank of a heuristic driver input:
argmax of `a_i*(mean_positive_ray_i-mean_negative_ray_i)`. Fractional positive
contrast and gradient magnitude are different attribution rules. The rank of
a specified input is fixed by the chamber, but choosing that input is not a
unique causal decomposition. Contrast changes when the ray representation or
means change. The sampled diagnostic uses minority rays plus 64 Dirichlet
combinations; clock labels are sampled too.

`_margin_split` tags already subdivided pieces by each gate's minority side.
`_finish` flips one tagged gate on each such piece and reruns the downstream
recipe with separate Ops, restoring STATS in a finally block. The numerical
`absdE` is the sum of absolute differences on those pieces, while `dE` sums
signed differences. This is a useful intervention on the fixed refined pieces.
It does not directly replay the adaptive estimator with the original split
removed: downstream approximate decisions can depend on the supplied cell's
rays and geometry. No representation-invariance proof bridges those two
operations here. Overlapping gate interventions cannot be added to obtain
joint pruning error or a probability of correctness change. Numerical cone
masses also preclude an unqualified exact-classification-effect description.

Normal source operations and diagnostic operations are separated explicitly.
Reference-output/counter parity can support this on saved runs; it does not
prove universal numerical equivalence of copied code or original checkpoint
identity. Final NPZ metadata does not store checkpoint bytes/hash or a run-time
source hash. Resumable per-half files are not in the frozen source snapshot.

## Cumulative guard interpretation
The guard log is collected only when the original per-gate trigger list is
empty. It records maxima of individual and summed shift ratios at a finite
union of candidate points. A cumulative ratio above one demonstrates that the
individual tests do not establish the corresponding summed pointwise bound.
It does not itself count how many extra splits a repaired algorithm would
perform, nor prove whole-cell soundness. No repaired traversal is run here.

## Pruning and its bound
In real arithmetic, exact unit rays, a defined normalized ray-sum axis and
positive minimum axis cosine give cap containment of their positive hull.
The cap mass formula is valid. For nonpositive cosine the fallback one is safe.
The implementation does not explicitly guard a zero axis or round the norm,
dot products, subtraction or incomplete beta evaluation outwards. Our four
frozen analytic controls test only that finite primitive behavior.

An ideal sum B can bound classification replacement on a disjoint partition of
correctly covered domains, assuming each term really upper-bounds that domain's
mass. Traversal pruning aims to replace a subtree once, but this source review
does not establish floating H/V geometry coverage or domain disjointness.
Wrong-centroid contributions call numerical `estT.mass` (the mass-mode bound
uses the same numerical routine). Those errors and the retained recipe error
remain separate; ordinary summation of B is not outward rounding. Therefore
`certified error bound`/`exact mass` in the author docstrings require these
additional qualifications for the returned floating estimate.

The mathematical narrow-cone control has positive mass even if floating axis
cosines round to one. A zero result there would disprove universal outward
certification of this implementation, not the geometric theorem, trained
accuracy, practical usefulness of pruning, or every possible conservative
repair. Outward enclosures, positive-mass fallbacks and a differently chosen
safe cap remain untested alternatives.

## Coverage and cost interpretation
Frozen attribution has H8T3/T4 seeds 0 through 4. There is a T5 reference and
clock record but no T5 attribution NPZ. README contains STATUS/RESULTS
placeholders and its all-11-runs parity sentence exceeds this snapshot's raw
coverage. This audit preserves these gaps; no missing result is synthesized.

Pruning comparisons use the author's T196-style convention: twice Ops divided
by `6000*flops(H,T)/flops(16,10)`, a continuous sampling count. This is not our
previous explicit native/sampler count and not a verified full-runtime budget.
The existing exact-labelled reference values are treated as saved numerical
references; no new uncertainty or exact-real truth certificate is supplied.

The later shared T196 board ruling explicitly lists sampler511 atH8T3 and682
atH8T4. The frozen reporting formula instead gives500 and2000/3, respectively.
Thus its `T196` docstring label does not make these ratios a reproduction of
that later full-sampler convention. We audit the actual saved formula, disclose
the distinction, and do not silently recalculate or certify full cost.
