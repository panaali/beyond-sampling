# T229 exact readout obstruction

Contract: a bias-free ReLU recurrence h'=ReLU(Wh h+Wi x), zero initial state,
and readout Wo h. A global-state weight automorphism is an invertible hidden
coordinate transformation A=DP, D positive diagonal and P a permutation,
which leaves the actual weights unchanged:

    A Wi = Wi,    A Wh A^-1 = Wh,    Wo A^-1 = Wo.

ReLU commutes with A. Every column of A^-1 has precisely one positive entry.
The last equation therefore pairs each column of Wo with a positive multiple
of another column of Wo. If every column is nonzero and no two distinct
columns are positively proportional, each column must pair with itself; each
positive scale is then1, so A=I. This is a sufficient exact obstruction to
nonidentity A, independent of the other two necessary equations.

If a column is zero or two are positively proportional, the readout condition
is merely inconclusive; input and recurrence constraints can still exclude
all nonidentity A. No symmetry is declared on those inputs.

The predicate divides each nonzero column by the absolute value of its first
nonzero entry, retaining its sign. Canonical equality is equivalent to positive
proportionality. Pair certificates additionally supply either a nonzero2x2
minor, an exact negative proportionality ratio, a positive ratio, or an
explicit zero column. All comparisons are exact rationals. Negative multiples
cannot be paired by a positive-monomial transformation. Distinct rays can be
arbitrarily close without changing this exact test.

Analytical controls include a three-cycle with equal readout columns (condition
inconclusive although a full network symmetry can exist), near-parallel dyadic
columns, zero columns, opposite columns, positive column rescaling and an
invertible row-coordinate transformation. No trained array is loaded in the
control phase.

Two alternatives are essential. First, numerical near-collinearity might look
like exact symmetry; exact small-minor controls distinguish them. Second, a
readout ambiguity may be killed by Wi/Wh; we therefore return inconclusive,
never a positive symmetry claim. The later target test, if separately gated,
can establish only this algebraic obstruction for the pinned exact weights.

Excluded: arbitrary reparameterizations which change the weights, common-logit
shifts/argmax-only equivalence, reachable-state/domain restrictions, noninvertible
quotients, general nonlinear/linear transformations not commuting globally with
ReLU, approximate symmetries, pruning and computational impossibility. The
identity-only result would not prove that every function-preserving merge or
useful representation is absent.
