# Positive history-distribution primitive review

Conditional on the sign s of X0, write X0=s*R*cos(theta), X1=R*sin(theta),
with theta uniform on[-pi/2,pi/2] and independent Rayleigh R. After two steps,
h2=R*ReLU(c*cos(theta)+Wi*sin(theta)), c=Wh*ReLU(s*Wi). Splitting at all
gate roots makes the hidden direction linear in cos/sin on each panel.
Exact panel means therefore preserve the conditional hidden mean when the
positive radial rule integrates R. The stable sinc formulas in distribution.py
and the angular second-moment formulas are correct.

The moment-Hankel construction forms J=L^-1 H1 L^-T where H=L L^T.
Its spectral weights are squares of first eigenvector entries because the
constant orthonormal polynomial is1. A positive n-node rule integrates
moments through2n-1 in real arithmetic; the implementation validates these
moments and positivity numerically. Hankel conditioning remains explicit.
For n>=2, the represented covariance loss is exactly
2*sum_panel p_panel*Cov_theta(direction), a positive-semidefinite matrix.
This is a formula, whereas the floating eigenvalue tolerance is a check.

Independent check_history_moments_review.py uses a fresh asymmetric
mixed-sign H2 model, direct50-digit angular integration, and separately
constructed gate breakpoints. Both conditional signs agree to1.11e-15 in
mean and4.44e-16 in second moment. This took .14896 CPU seconds. The check
does not reuse source panel moments and does not test suffix accuracy.

The represented distribution deliberately discards the dependence between
prefix hidden state and prefix ranks within each sign. Preserving a mean,
positive support, or some variance does not repair that dependence. A matched
coupling discriminator is valid if both compared methods use identical
expanded polar atoms/weights and the representative inputs propagate to
exactly the represented hidden state. Adding the X0=X1 angular boundary
preserves within-prefix rank on each atom. Pointwise target probabilities
must sum to1, but a finite coupled prefix quadrature need not give each
target label exactly1/4; that defect must be reported, not normalized away.
Such a comparison isolates coupling on the finite rule, not Gaussian-law
convergence or the unique cause of the earlier failures.
