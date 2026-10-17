# Which singular supports must H2 transport keep?

Consider h_t=ReLU(W h_(t-1)+w X_t), h_0=0, w in R^2, with jointly
absolutely continuous inputs (independent Gaussians are the intended case).
At every finite t, the hidden-state law is a sum of:

1. a measure absolutely continuous with respect to area in the open quadrant;
2. one measure absolutely continuous with respect to length on each open axis;
3. an atom at the origin;
4. possibly a measure absolutely continuous with respect to length on the
   single interior ray span(w) intersected with the open positive quadrant.

The last support exists only when both coordinates of w are nonzero and have
the same sign. If w=0 the law is simply the origin atom. Components can have
zero mass; this theorem does not claim that each occurs or give their densities.

## Proof from the last gate, not a smoothness assumption

Partition input space into the finitely many ReLU gate cells. On each cell,
h_t=B x is linear and the restricted input law remains absolutely continuous
with respect to t-dimensional Lebesgue measure. Its pushforward is absolutely
continuous with respect to Lebesgue measure on the image of B. This is a
finite-dimensional linear change of variables, or the coarea formula.

If both final gates are active, the last column of B is w. Rank2 therefore
gives area density; rank1, when w is nonzero, has image exactly span(w).
If w=0, all states from h0=0 vanish. If only one final gate is active, the
image lies on that axis. Rank1 gives its length density and rank0 gives zero.
With both final gates inactive the state is zero. Thus no additional interior
line or nonzero atom is possible. A boundary with identically zero preactivation
can be assigned either consistent gate value: the state is unchanged, and the
same rank argument still applies. Nonidentically zero boundaries have input
measure zero.

Rank/order constraints on the inputs only restrict the cell density and do not
invalidate this argument. They may, however, make the component densities
difficult to compute. A finite list of supports does not imply a cheap exact
representation of the densities after many steps.

## After a fixed target insertion: two fresh inputs suffice generically

Condition on target value u. A deterministic target insertion can produce
offset lines or nonzero atoms, so the preceding homogeneous proof cannot be
applied immediately. Assume both coordinates of w are nonzero and

    det(W w, w) != 0.

After two additional independent continuous scalar inputs x,y, the same four
types of support suffice for ANY initial hidden-state distribution independent
of x,y. To see this, condition first on a fixed initial state h. For a gate
matrix D at the first fresh step and both gates active at the second,

    h'' = W D W h + W D w*x + w*y.

If D=I, the columns Ww and w are independent by assumption; hence the
pushforward has a two-dimensional density even though its offset depends on h.
If D selects coordinate i, write alpha=(Wh)_i. The formula becomes

    h'' = W e_i*(alpha+w_i*x) + w*y.

Since w_i is nonzero, either W e_i and w are independent and give area density,
or they are collinear and the offset also lies in span(w), giving the input
ray. If D=0, only the input ray is possible. If the final gate selects an axis
or zero, its nonzero w coordinate gives an axis density or the origin atom.
Finally mix over h: a mixture of densities on these fixed supports is still a
density on those supports. Additional rank-count restrictions merely restrict
the joint input density and preserve the argument.

This is a conditional fixed-u statement. The joint law of (h,u), or normalized
(h/R,u/R), needs its own representation; the result must not be interpreted as
proof that the three-dimensional augmented law has a smooth density everywhere.

## Both nondegeneracy conditions matter

If det(Ww,w)=0, take W=I, w=(1,1), h=(1,2). On the positive-measure region
where both steps keep both coordinates positive,

    h'' = (1+x+y, 2+x+y),

which remains on the offset line h''_2-h''_1=1 after two fresh inputs.

If a coordinate of w vanishes, take w=(1,0),
W=[[1,1],[1,0]], h=(1,1). Here det(Ww,w)=-1 is nonzero. For x<-2 the first
state is (0,1), and for y>-1 the second state is (1+y,0), which is just an
axis; this particular choice does NOT falsify the theorem. A correct example
uses W=[[1,1],[1,1]], h=(1,1): for x<-2 the first state is (0,2), then for
y>-2 the second state is (2+y,2), a positive offset horizontal line.
Its determinant det(Ww,w)=-1 remains nonzero. The missing input into coordinate2
lets the first single-active gate retain a nonzero atom, so two fresh steps do
not smooth the result. The rejected first example is kept to expose why
checking the last coordinates matters.

## What this buys and what it does not

For the ordinary H2 recurrence from zero, a transport solver can explicitly
track at most one interior ray, two axes, and the zero atom alongside its
area density. For opposite-sign input weights no interior ray is needed.
Following a conditioned target insertion, offset strata must be retained until
the two-input smoothing condition applies, or integrated by a terminal kernel.
This narrows the representation question. It does not solve discretization
error, certify rank-count conservation, or establish a cost advantage over
Monte Carlo. The original H16 challenge remains substantially larger.
