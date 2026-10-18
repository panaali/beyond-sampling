# Algebraic edge cases (no program execution)

These identities are checked by elementary substitution and are part of the proof-review request, not numerical experiments.

1. **Zero span.** R={(0,0)} implies C={0}. Every homogeneous linear-plus-ReLU function is zero there, regardless of ambient representation.
2. **Lower-dimensional linear equality missed by ambient test.** C=pos((0,1)) and F(x)=x1. Ambient linear matrix is nonzero; restriction x=(0,z) gives F=0. A negative result from the old ambient procedure must remain unknown.
3. **Supporting hyperplane, no interior crossing.** C=pos((1,0),(0,1)) and F=ReLU(x1)−x1. The x1 hinge is nonnegative on every generator and becomes linear, giving zero. Keeping it as a global hinge would incorrectly reject equality on C.
4. **Rank-one crossing.** C=pos((1,0),(-1,0)). F=ReLU(x1)−ReLU(-x1)−x1=0. In its one-dimensional span the only hyperplane is{0}; negative orientation contributes the missing linear residue. G=ReLU(z) alone remains nonzero.
5. **Lineality.** C=pos((1,0),(-1,0),(0,1)) is the half-plane x2≥0. ReLU(x2)−x2 vanishes after fixed-sign elimination; the mixed hinge ReLU(x1) crosses at every(0,t),t>0, so nonzero jump coefficients remain detectable inside the cone.
6. **Vector cancellation is componentwise.** A retained hinge with c=(0,1) and u=(1,0) has nonzero second-coordinate gradient jump even though its first coordinate vanishes. Checking a scalar aggregate of outputs would be unsound.
7. **Disjoint-domain union is not its conic hull.** F=ReLU(x1)+ReLU(x2)−ReLU(x1+x2). It is zero on rays(2,1) and(-1,-2) individually, hence on their union. At their sum(1,-1), F=1. Equality on both cones does not license replacing their union by one convex cone.
8. **Function inequality is not label inequality.** Relative margins x1 and2*x1 differ on the positive x1 ray but give the same winning output. The decision procedure tests exact margin functions, not all useful label equivalences.
9. **Mixed forms become zero on a smaller span.** Ambient forms x1 and x1+x2 are distinct. On span((1,0)) they coincide and must be grouped after projection. Canonicalizing ambient hyperplanes before restricting can miss this equality.

Unproved extensions: affine biases on nonconic domains, equality only on a nonconvex set without finite cone decomposition, approximate/probabilistic equivalence, trained-domain validity and savings of an actual merged enumerator. None is excluded by this theorem.
