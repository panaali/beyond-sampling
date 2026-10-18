# T23 fixed-dictionary reconstruction review

On nine positions, the eight nonconstant real Fourier functions form a
basis of the zero-mean position functions. Centered Legendre polynomials
of degrees1..8 also form such a basis: the full degree0..8 evaluation
matrix is nonsingular by the distinct-grid Vandermonde argument, and
centering removes only the constant direction. Unit-RMS rescaling is
invertible. Constant gap=m1-m2 and m2 directions already belong to the
first two features. The final position interaction is identical. Thus
both full20 dictionaries span the same functions; smaller prefixes need
not. ReLU after an invertible coefficient change preserves the hypothesis
class, but does not guarantee the same nonconvex optimized fit.

There are exactlyK*H fitted scalar entries, no fitted offsets or hidden
normalizers. This is not a complete bit-description measure: universal
feature algorithms, selected family identifier and real coefficient
precision carry structural information. The rank-only family deliberately
omits position identity, so its poor fit is not evidence against all
compact reconstruction formulas.

The source gradient is the derivative of mean squared rectified hidden
error, normalized by the number of entries. The frozen infinity-gradient
criterion, with derivative0 at a ReLU kink, is a first-order diagnostic,
not global optimality. Capped fits must remain unconverged even if the
optimizer reports success. Accuracy intervals are conditional on fits;
paired adjacent-budget intervals do not cover feature/seed selection.

A future reversible training-only QR transform Phi=QR, gamma=R*beta
preserves the class when R invertible and stores predictions back as
original beta. Its gradient criterion must remain in original-beta
coordinates for comparison; whitened gradient norms are different.
Rank-truncated SVD can change the class and must not be called a merely
reversible preconditioner. No further fit or new task was run in review.

## Frozen QR implementation

The source uses S=sqrt(N)*Q and gamma=R*beta/sqrt(N), then solves
beta=R^-1*(sqrt(N)*gamma). The gradient relation is
original_gradient=R^T*transformed_gradient/sqrt(N). The synthetic finite
-difference control avoids ReLU kinks; transform/objective controls and
original-coordinate stopping are appropriate. Numerical rank deficiency
raises rather than truncating. Checkpoint hash pinning was requested and
added before measurements, independently of immutable old-result hashes.
Rescue runs remain restricted to original nonconverged fits and retain
failures/caps. No QR numerical fit results were independently rerun here.

## Final active-set rescue and saved-run audit

On a fixed neuron and direction, all ReLU breakpoints partition the line
into quadratics. The implemented coefficients and crossing updates agree
with direct expansion; endpoints and clipped stationary points give the
real-arithmetic minimum. Long-double ranking followed by direct original
loss checks on the best eight candidates plus endpoints guarantees only
measured nonincrease, not a certified global line minimum under floating
cancellation. The correction direction A^+(y-A beta) preserves the current
active nullspace component, unlike replacing beta with A^+y. Both optimize
the same fixed active least-squares objective but can follow different
inactive patterns. Neither proves global network reconstruction optimality.

`check_multifamily_final_review.py` independently joins raw saved records,
without executing the author's summary or fitting again. Exactly90,25,14,3
cells enter original, QR, active and correction stages, respectively;
rescue keys equal the preceding stage's failed set. First-passing counts
are65,11,11,3, totaling90. Every original threshold, coefficient shape,
16 source/result hashes,803 nonincreasing saved steps, final stage/accuracy
and gap arithmetic pass. Maximum final gradient/threshold is.9997099279.
Recorded stage CPU sums to599.344223127, including repeated preparation;
review CPU.0207. This audits stored gradient values and source logic, not
independent recomputation of every fitted gradient. Ordinary floating
first-order convergence remains distinct from global optimality.

For iid continuous inputs, a predictor invariant under permutations of
the first nine positions has accuracy at most1/5 on second-largest-index
prediction. Conditional on the unordered first-nine values and the last
value, the last index is identifiable as target exactly when m2<x9<m1;
this event has probability1/10. Otherwise the true index is uniformly
one of the first nine, so conditional success is at most1/9. Therefore
accuracy <=1/10+(9/10)/9=1/5. Choosing index9 on that event and fixed index0
otherwise attains the bound in the unrestricted invariant predictor class.
Rank-only features followed by any fixed readout inherit this obstruction;
attainability need not hold in the particular linear-ReLU parametrization.
This exact scoped symmetry obstruction says nothing about position-aware
compact descriptions or the ARC accuracy-estimation challenge.
