# Explicit algebraic limits of the source negatives

These are analytical examples, not new model experiments. Independent review
is requested; none is evidence that the particular target has a symmetry.

## Pair transpositions do not exhaust permutations

Take three hidden coordinates, input Wi=(1,1,1)^T, recurrent matrix

    P = [[0,1,0], [0,0,1], [1,0,0]],

and readout Wo=(1,1,1). The hidden permutation P fixes Wi and Wo and commutes
with the recurrent matrix P. Hence it is a global weight automorphism. Every
nontrivial transposition Q conjugates this 3-cycle to its inverse, not to P,
so no transposition is a weight automorphism. Enumerating all pair swaps can
therefore reject every tested swap while missing a cyclic symmetry. The target
could still lack every permutation; the source scan alone does not establish it.

## Positive gauges and near-angle scores

For h'=D h with positive diagonal D, ReLU(D z)=D ReLU(z), so the same function
is represented by Wi'=D Wi, Wh'=D Wh D^-1, Wo'=Wo D^-1. Start with Wi=(1,1),
Wh=I. Its augmented rows (1,1,0),(1,0,1) have cosine1/2. D=diag(2,1) keeps
Wh'=I and changes them to (2,1,0),(1,0,1), with cosine2/sqrt(10). Exact positive
proportionality is a different predicate from an approximately high cosine.
This example concerns parameterization, not the numerical value of the target's
cosines in the original parameterization.

A stronger finite target test (registered T229) is possible for the restricted
global-state positive-monomial automorphism class A=DP. Readout invariance requires
Wo A^-1=Wo. Since each column of A^-1 contains exactly one positive entry, each
readout column must equal a positive multiple of another. If the readout columns
are all nonzero and pairwise non-proportional, P must be identity and each
scaling must be one, so A=I. Otherwise this necessary condition is inconclusive
and the input/recurrent constraints remain. This is not a criterion for every
reachable-domain equality or noninvertible quotient.

## Recovering a full history is stronger than computing the output

For any joint distribution of event label E and full gate history G, the best
one-word-per-event exact-history recovery probability is sum_e max_g p(e,g).
This follows by choosing the largest conditional mass independently for each e.
It is the probability that the entire history is right, not a lower bound on
output-prediction error or error-estimator bias.

For a concrete distinction, set Wo=0 in any bias-free hidden recurrence. The
first-index argmax always predicts0, regardless of how many hidden histories
occur or how difficult they are to infer from E. Under continuous exchangeable
inputs the second-largest index is uniform, so exact accuracy is1/T without
recovering any history. This does not rescue the trained target; it proves why
low modal gate-word share alone cannot exclude output-preserving aggregation.

## Full exact gate histories define convex cones

For a fixed complete gate word, replace each ReLU by its prescribed diagonal
0/1 mask. Every successive preactivation is then a linear form in the full
input vector. The inputs realizing that word are precisely the intersection
of its ON strict halfspaces and OFF non-strict halfspaces (with exact zero
handling). This set is convex and homogeneous, possibly empty or lower-dimensional.
An artificial initial fan can subdivide it into several pieces, but cannot
make the exact full-word region itself a nonconvex union. Thus the original
README's generic possibility of a nonconvex full-gate-pattern region is not a
real-arithmetic property of this bias-free ReLU recurrence. Numerically assigned
patterns or incomplete histories are separate cases and remain qualified.
