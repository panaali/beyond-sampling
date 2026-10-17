# T177 structural constant-first-label shortcut

## Question
Can a sufficient weight-only test resolve the preserved alloff control before geometry/quadrature, while leaving nonconstant trained calls unchanged?

## Method
Claimbe027783 before code. Require finitebinary64 arrays, zero initialhidden, biasfree ReLU scalar-input recurrence. If Wi==0 then hidden remainszero. Otherwise, if readoutrow0 is coordinatewise>=every rival, nonnegativehidden implies logit0>=every rival; first-index tie ownership givesprediction0 everywhere. Identical rows are covered by that same dominance test rather than redundant work. Under continuous exchangeable inputs, the second-largest position is uniform, so accuracy exactly1/T. Return exactpair(1,T) plus its binary64 division. No sampled inputs, mass/geometry calls or storedweight dependency in the primitive.

## Status
Protocol/primitive prepared; independent theorem gate supplied, source/control gate pending.

## Results
Pending. Prior T171alloff72-cone firstfinitecap failure remains pinned, never overwritten. No estimator target/order changes.

## Compute accounting
Fullfinite scan H+H²+TH, Boolean reductions, Wi equalities and optional(T-1)H dominance comparisons/reduction plusonefinaldivision explicitly charged. Metadata predicates counted as written validationcalls, not expanded interpreter/integer/tuple internals. Inputarray coercion/copy avoided by requiring binary64 NumPy arrays. Boolean masks allocate transient storage; runtime/allocation/indexing/native internals remain explicit exclusions, no fullcostclaim. Trainedfalse-branch validation reads existing two checkpoints only, with hashes and validation CPU separate; no geometry/mass rerun.

## Claims
[inferred, independently reviewed] Sufficient real-weight proof ofconstant label0 and exact1/T. Firstindex is essential. No assertion about other fixedlabels, arbitrary initialstates/biases, finite inputoverflow semantics or general always-return behavior.

## Alternatives not ruled out
Reachable-hidden restrictions could certify more models even when coordinatewise dominance fails; otherconstantlabels require proving positivehidden toexcludealloff ties. Both are outside this fixedphase. Original generic quadraturecap failures remain possible for nonshortcut models.

## Failure modes
Nonfinite/shape/dtypeinvalid inputs reject explicitly. Exact zero/comparisons only; no tolerance. A failed sufficientpredicate delegates unchanged to caller's fallback, rather than asserting nonconstant or inventing accuracy. Final1/T float is rounded; exactrationalpair retained.
