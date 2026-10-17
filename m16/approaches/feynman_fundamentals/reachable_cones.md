# Exact H2 reachable-direction recursion for positive input weights

T55 independent derivation for T77, 2026-10-17. Consider
h'=ReLU(W*h+w*x), h0=0, with w1,w2 strictly positive and x having full support
on the real line. Let D=diag(w), z=D^-1*h, A=D^-1*W*D. Positive scaling commutes
with ReLU, so

    z'=ReLU(A*z+x*(1,1)).

Define d=A_row2-A_row1. The difference between the two preactivations is d*z,
independent of the fresh input. Four closed cones suffice after the first step:

    C0 = {r*(1,1): r>=0},
    C+ = {z2>=z1>=0},
    C- = {z1>=z2>=0},
    CQ = {z1>=0,z2>=0}.

The first-step support is C0. If the current support is one of these cones C,
the next support is determined by the signs of the linear functional d on C:

* identically zero: C0;
* nonnegative, but positive somewhere: C+;
* nonpositive, but negative somewhere: C-;
* both positive and negative somewhere: CQ.

## Proof of exactness, not merely an outer bound

For fixed z let a=A*z. If a2>a1, ReLU(a+x*(1,1)) always has its second
coordinate at least its first. As x varies, it traces the origin, part of the
second axis, then a line with positive fixed coordinate difference a2-a1.
Scaling the source z by any nonnegative number scales this difference by the
same amount. Hence a single source direction with d*z>0, together with scaling
and unrestricted x, reaches every point of C+ with strictly positive coordinate
difference. The diagonal is reached from the source origin. This proves that
the image is the whole closed wedge C+ whenever d is nonnegative and nonzero
on C. Reversing coordinates proves the C- case. If both signs occur, the union
C+ union C- is the full quadrant. If d is identically zero, both preactivations
are identical and the image is exactly the diagonal ray.

All these sets are closed polyhedral cones. Gaussian inputs have positive
density near every finite input sequence, and the recurrence is continuous.
Thus the closed reachable set equals the distribution's topological support.
This statement concerns support, not the proportions of origin, axis, ray and
area probability. An axis belongs to a support closure without necessarily
carrying positive singular mass, so density claims still need their own proof.

## Exact finite tests in original coordinates

Let n=(-w2,w1). The sign of n*W*h equals the sign of d*z, since they differ
by the positive factor w1*w2. Test this functional only on the extreme rays:

| Current cone in h coordinates | Extreme rays |
| --- | --- |
| input ray | w |
| upper wedge h2/w2>=h1/w1 | e2,w |
| lower wedge h1/w1>=h2/w2 | e1,w |
| full quadrant | e1,e2 |

Nonnegative coefficients on both generators imply nonnegativity throughout
the cone. Exact rational comparisons on binary-rational weights avoid inventing
a tolerance for zero signs. In normalized coordinates, C+ is invariant exactly
when d2>=0 and d1+d2>=0. Necessity follows by taking a large positive fresh input
so both gates are active; sufficiency follows from monotonicity of ReLU.

For the toy w=(1,1), W=diag(1,2), d=(-1,2). Starting from the diagonal, the next
cone is C+; d is nonnegative on its two generators, so all later cones remain
C+. Therefore h2>=h1 and positive axis0 mass is impossible at every depth.
An operator that conserves mass but creates positive axis0 mass fails this
support test, exactly as observed in the earlier rectangular projection.

## Input truncation and projection

If all inputs are restricted to [-L,L] with L>0, the actual reachable set is
bounded and is not the entire cone. Its set of directions is nevertheless
unchanged: any finite unbounded-input trajectory can be scaled by a sufficiently
small positive factor to put every input inside [-L,L], while the final hidden
state scales by that same factor. Thus the cones above remain exact directional
supports and valid constraints for the truncated computation. The separate
finite box determines an outer radial extent; cone intersect box need not be
the exact full reachable set.

Projecting a represented area mass into a cell intersected with its proven cone
can preserve these support constraints. Positive quadrature nodes should lie in
that actual clipped polygon. The approximation is still a density projection;
neither clipping nor row conservation proves its accuracy. In particular,
discarding outgoing forbidden mass and renormalizing afterward is not the same
as constructing a source representation whose nodes obey the support theorem.

The same recursion applies at every stage, so an early narrow support can be
useful even if a later stage reaches the full quadrant. Conversely, once the
full quadrant maps to itself, this particular support constraint yields no
additional clipping benefit. Trained-model benefit must be measured separately.

## Scope and falsifiers

The proof assumes strictly positive entries of w and unrestricted fresh scalar
input before the optional symmetric truncation argument. A zero or mixed-sign
input vector requires a different coordinate/support classification. A fixed
target insertion is a deterministic translation and need not preserve these
four conic types; this theorem must not be reused unchanged for target-augmented
transport. Biases likewise break the scaling argument.

Useful exact controls are W=I (diagonal forever), W=diag(1,2) (upper forever),
W=diag(2,1) (lower forever), and a matrix with d=(1,-2): the diagonal first maps
to lower, but the next test on e1 and(1,1) has opposite signs and reaches the
full quadrant. These exercise all branches without model-input sampling.
Numerical implementations still need exact or conservative half-plane signs
and boundary handling; a correct support theorem does not certify float clips.
