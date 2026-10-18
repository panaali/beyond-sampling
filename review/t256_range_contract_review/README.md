# T256 independent written range-proof review

## Question
Does the finite clipped-section construction give exact affine extrema on one
labeled gap-constrained box, with correct physical-coordinate conversion?

## Method
Delegation `c02f8a1d`. Read frozen `9ebdd477` and manifest `93fc43c8`, the
preserved T240 unit failure and corrected diagnostic handoff, and inherited
enclosure source/theory. Inherited reviewer payloads are hash-bound only;
this is not another saved-array audit. No primitive, LP, vertex enumeration,
CDF, model, checkpoint or archive call is made.

## Status
The written theorem/protocol passes. `theorem_review.json` binds all ten
declared predecessor/source pins plus the manifest. Execution is not cleared;
implementation and finite controls require their own source/resource gates.

## Results
For fixed i,j, t=x_j is feasible exactly on

    [max(l_j,max_{r != i,j}(l_r+g)), min(u_j,u_i-g)].

Necessity follows from each pair of gap/box inequalities. Conversely each t
in that interval makes all independent remaining coordinate intervals nonempty.
The original valid l<=u conditions explain why l_i and u_r add clipping
breakpoints but no new feasibility bounds. Empty and singleton intervals
are correctly distinguished; neither feasibility nor a witness implies
positive Gaussian mass.

For a fixed t, the affine objective attains each conditional extremum by
coefficient-sign endpoint selection. Zero coefficients may always take the
lower endpoint, as the explicit deterministic exception specifies. Both
conditional objective functions are continuous piecewise affine. Their only
possible kinks are l_i-g and u_r+g. Including all such points inside the
closed feasible interval and its endpoints therefore captures both global
extrema; no missing interior smooth extremum is possible for an affine piece.
The completed coordinate choices are attained feasible witnesses, not just
upper/lower estimates. Compactness and finite endpoints exclude unbounded cases.

At most T+1 candidates times O(T) expression work gives O(T^2) per objective
and ordered pair. Input checks, exact comparisons/deduplication, rational
bit complexity and the T(T-1) all-pairs multiplier remain separate. This is
not a native-FLOP or runtime statement.

The two input-unit adapters are correct for v=2x: either divide endpoints
by two and replace a_v by 2*a_v at physical gap g, or retain v and use gap
2g with the original objective. For the inherited integer numerator
(a_v dot v+e)/S, this doubles only the affine coefficient on x; S and the
additive numerator error remain unchanged. A final positive-denominator
division still belongs to that caller's representation. The new core is
physical x; the old unconverted T240 results are not silently repaired.

For positive g the two specified labels are unique top and second labels.
At g=0, the closed polytope includes rank ties, so a pointwise true-label
certificate needs its own tie convention or strict-domain argument. Network
argmax first-index ownership is a different predicate. Allowing lower-dimensional
feasible boxes is consistent with a range theorem but not positive-mass accounting.

An inherited uniform remainder interval remains valid on a subset. Adding it
to the exact restricted affine range is sound; intersection with another valid
range is also sound. Rechoosing later ReLU slopes changes the decomposition,
so the proof correctly avoids claiming monotone improvement of the entire
network certificate. Retaining the original valid certificate as fallback
and charging both paths is appropriate. Event mass remains a separate problem:
full unconditioned box mass cannot be attached to a partial-gap certificate.

The proposed future controls address the important failure classes: interior
kinks, signs and zeros, degeneracy/emptiness, physical-unit adapters with a
nonzero objective slope, and an independent vertex-based reference. This is
adequate as a written plan, not a frozen executable suite or trained gate.
The prior actual-gap diagnostic is not new numerical evidence for T256.

Reproduce metadata only:
`python review/t256_range_contract_review/check_metadata.py`.

## Compute accounting
Written algebra, source reads and hashing only. No scientific numerical calls,
implementation timing, model evaluation or inherited array replay.

## Claims
[inferred] The finite affine-range reduction and unit conversions are valid
under the stated exact finite inputs. [speculative] It may help downstream
certificates; no improvement in mass, cost or target accuracy is shown.

## Alternatives not ruled out
Dependent ReLU remainders may remain limiting; partial-box integration,
other partitions or stronger correlated enclosures may be needed. No method
failure or broad impossibility follows from this prerequisite.

## Failure modes
Unit/denominator mismatch, missing a kink, accidental first-index ownership,
invalid exact input, zero-volume mass assignment, or assuming a new slope's
global certificate dominates the old one would invalidate later applications.
