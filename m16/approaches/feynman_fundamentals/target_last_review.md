# Target-last terminal cache: completeness and boundary review

Read-only terminal_cache.py review supports the exact-arithmetic construction
for the stated H2 scope. For fixed incoming hidden state, the fresh x step is
piecewise affine in x, and the final target-u step is affine in(x,u) on each
two-step gate pattern. Each retained original inequality a*u+b*x+c>=0 gives
an affine x lower/upper bound if b is nonzero, or a u-only feasibility test
if b=0. Include support bounds, x=u rank split, every line-pair crossing,
and every u-only root. Between these knots, active max/min bounds and
feasibility are fixed, so truncated-normal mass is analytic.

Source preserves original constraints rather than reconstructing them from
rounded polygon edges. Active gate constraints are strict, identically zero
gates have one inactive branch, and every competing readout margin is strict
because the target is the last index. An identically zero margin therefore
rejects the whole cell. Strictness of x-dependent boundaries is immaterial
to conditional continuous-x mass; strict u-only boundaries are retained in
endpoint inclusion flags, which matters for exact fixed-u queries. Internal
slab boundaries use single-sided ownership. First-step input boundaries
have zero conditional-x mass. These decisions agree with first-index ties.

The final scalar CDF uses survival differences in positive tails and divides
by the same truncated-input Z*q or Z*(1-q). Earlier fixed-grid transitions
are linear combinations of analytic functions on the union of their
StagePieces knots and these terminal knots; they introduce no new branch
thresholds. This is an exact-arithmetic completeness argument, not a proof
that floating roots, midpoint classifications, or near-coincident lines are
certified. Tiny omitted panels retain their own explicit mass bound.

The same-panel q-coordinate rescue is mathematically appropriate: mapping
u knots through the monotone truncated CDF preserves topology, while the
Beta(3,2) density12q^2(1-q) is cubic. Two-point Gauss integrates constant
payoffs exactly on each represented q panel in real arithmetic, without
renormalization. Nonconstant inverse-CDF derivatives still need numerical
assessment. Positive u panels that round to zero q width must retain an
omission bound; using the original-u density bound does so.

Reported h5 cache/direct parity is bit-identical. The original u-rule
GL2/4 change3.17e-7 fails its frozen further-grid gate. The distinct q-rule
reduces that change to1.77e-8, with raw mass defect4.35e-14 and qGL4/uGL4
agreement2.31e-12. These support finite terminal-cache and outer-rule
functionality. They do not establish spatial convergence, a full estimator
error certificate, or permission to discard the failed u-rule or expand
the grid without the separate protocol. No extra full engine run was made
in this source review.
