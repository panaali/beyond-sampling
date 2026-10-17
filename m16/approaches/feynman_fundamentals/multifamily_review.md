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
