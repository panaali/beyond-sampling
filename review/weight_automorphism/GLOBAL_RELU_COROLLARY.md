# Restricted global linear-coordinate corollary

Proof-only clarification; no new numerical phase. Assume an invertible real
linear matrix A satisfies A ReLU(z)=ReLU(Az) for EVERY z in R^H.

If x>=0 coordinatewise, A x=A ReLU(x)=ReLU(Ax)>=0. Conversely, if y>=0 and
x=A^-1 y, then A ReLU(x)=ReLU(y)=y=Ax, so invertibility gives ReLU(x)=x and
x>=0. Thus A maps the nonnegative orthant bijectively to itself. An invertible
cone automorphism maps extreme rays bijectively to extreme rays: otherwise a
nontrivial positive decomposition of an image ray pulls back to one of the
original extreme ray. The orthant's extreme rays are exactly its positive
coordinate axes, so each column of A has a single positive entry, each in a
different row. Therefore A=DP for a positive diagonal D and permutation P.
The converse holds because coordinate permutations and positive rescalings
commute with coordinatewise ReLU.

Consequently, the T229 exact readout obstruction also excludes nonidentity
invertible linear coordinate changes that both commute with ReLU on ALL
preactivations and leave the actual input, recurrence and logit weights
unchanged. This is a characterization of the SAME positive-monomial class,
not a target search over new transformations.

The universal preactivation quantifier is essential. Commutation only on a
reachable state/input set, common-logit shifts, argmax-only equivalence,
noninvertible quotients and function-preserving rewrites without global
coordinatewise ReLU commutation are outside this statement. No full algorithmic
or all-representation impossibility follows.
