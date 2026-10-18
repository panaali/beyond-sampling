# Exact one-step margin signatures

Let a represented prefix state be h=Mx. For one remaining scalar input z,
readout margins relative to output0 are
F(x,z)=V ReLU(B[x;z]), B=[Wh M, Wi], V=Wo[1:]-Wo[0].
All quantities below are exact rationals. For captured binary arrays this is the
real-arithmetic function of their stored dyadic coefficients, not a claim about
all-input floating execution or correctness of earlier recipe gate freezing.

For each nonzero row b, let alpha be its first nonzero coefficient and u=b/alpha.
Then u has first nonzero entry1, independent of orientation. The row contributes
|alpha| v ReLU(u.y), and if alpha<0 also contributes alpha v u.y to a linear
matrix L. This uses ReLU(-s)=ReLU(s)-s. Combine equal u, remove identically zero
vector coefficients, and sort. The resulting (L, {(u,c)}) is canonical.

Sufficiency follows directly from the identity. Necessity for equality of these
functions on all R^d follows from gradient jumps: at a generic point on each
one of the finitely many distinct hyperplanes, all other hinge terms are linear,
so a nonzero vector coefficient produces a nonzero jump c u. Once all hinge
coefficients vanish in the difference, its remaining linear matrix must vanish.
For d=1 there is only one possible hyperplane and the two slopes give the same
conclusion. This proves equality of margin functions, not completeness for
classification equality on a restricted prefix cone.

A second sufficient classification signature divides all entries of L and all
hinge coefficients by the absolute value of their first nonzero coefficient,
leaving u unchanged. Equality certifies a common positive scalar multiple of
all margins. Positive scaling preserves argmax and every tie; all-zero margins
choose index0. Negative scaling is not permitted. Predictions are obtained from
scores [0,F1,...], with the first maximum chosen.

These tests are deliberately stronger than future-label equality. For example,
|x|+|y| and |x|+2|y| are both positive away from the origin, giving the same
binary first-index classifier everywhere, but their canonical/projective
margin signatures differ. No nonmatch can rule out this family, prefix-domain
restrictions, earlier-state quotienting or longer-horizon methods.

A census must keep truth/depth/initial-chart provenance. Same represented state
repeated across rank charts is distinct from additional equal functions with
different states. Even certified equal functions on multiple cells do not make
their union convex, eliminate truth constraints, or remove the cost of its
Gaussian mass. Initial synthetic controls make no merge-rate or speed claim.
