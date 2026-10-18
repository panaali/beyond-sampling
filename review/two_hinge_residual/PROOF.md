# T233: amplitude-preserving two-hinge residual

This is a real-arithmetic proof prerequisite, not an estimator, numerical
implementation or assertion about the target's pair structure. Write t+=max(t,0).

## 1. Exact gate freezing with an explicit correction

For any real a,b define s(a,b)=b 1[a<0] and r=b+−s. Then

    b+ = b 1[a<0] + r,
    r = |b| 1[(a<0 and b<0) or (a>=0 and b>0)] >= 0.

The four strict sign combinations prove the identity immediately. At b=0 the
residual is zero. At a=0 it equals b+, so the boundary ownership is explicit.
The pair (a+,s) preserves the actual amplitude of b; it does not replace b
by a fitted multiple of a. However s can be NEGATIVE inside a mismatch
interval. It is a signed intermediate representation, not an independently
valid nonnegative hidden state. Adding r restores the exact hidden state.

On opposite-sign interiors the correction vanishes. This identity is related
to the already reviewed gate-freezing residual in T108; the proposed distinct
prerequisite is its exact conditional law for two opposite fresh-input slopes.
No independent gate-splitting saving follows merely from rewriting a function.

## 2. Conditioning on the entire previous state

Let H be any past-measurable state and X a fresh independent N(0,1) input.
Fix H=h; let

    a=alpha X+c(h), b=beta X+d(h), alpha*beta<0,
    u=−c(h)/alpha, v=−d(h)/beta.

The slopes are finite nonzero constants. Their signs are opposite, so a and b
have the same strict sign exactly between u and v. With L=min(u,v), U=max(u,v),
up to the null endpoints under this continuous conditional law,

    r = |beta| |X−v| 1[L<X<U].

This statement holds whether the interval has both-positive or both-negative
preactivations and in either slope orientation. It does not require H to be
Gaussian, independent across coordinates, or supported in a full-dimensional
set. Independence of the NEW X from H is essential. Conditioning on rank/order
information about X would change its distribution and invalidates the standard
normal formulas below unless replaced by the corresponding conditional law.

If u=v, the two nonzero forms are exactly opposite proportional for this h;
r is identically zero including their common zero. Otherwise

    P(r>0 | H=h) = Phi(U)−Phi(L) = |Phi(u)−Phi(v)|.

This probability is not asserted to be small. Near-antipodal coefficient
cosines alone do not bound threshold separation, particularly with small slopes.

## 3. Exact first moment and positive bounds

Let phi and Phi be the standard-normal density and CDF. The exact integral is

    E[r | h] / |beta|
       = integral_L^U |x−v| phi(x) dx
       = phi(v)−phi(u)−v [Phi(u)−Phi(v)].

One way to see the sign in the last formula is to write the nonnegative
oriented integral integral_v^u (x−v)phi(x)dx. When u<v, both the integration
orientation and x−v change sign. Integration uses phi'(x)=−x phi(x).
The displayed difference is mathematically nonnegative but can catastrophically
cancel in finite arithmetic. It is NOT a stable numerical recipe or certificate.

With delta=|u−v| and p=|Phi(u)−Phi(v)|, direct positivity yields

    0 <= E[r | h] <= |beta| delta p,
    E[r | h] <= |beta| delta^2 / (2 sqrt(2 pi)).

The second bound integrates the triangular function |x−v| over the interval
and uses the uniform density upper bound1/sqrt(2pi). It is quadratic in
threshold separation, even though disagreement probability is generally linear.
For fixed v and delta→0, continuity gives E[r|h]/delta²→|beta|phi(v)/2.
This limit does not assert uniform relative numerical accuracy.

For fixed opposite slopes and c(h)=w_i·h, d(h)=w_j·h, write

    ell = w_j/beta − w_i/alpha, u−v=ell·h.

If H has a finite second moment, then

    E[r] <= |beta|/(2 sqrt(2pi)) ell^T E[HH^T] ell.

This is a conditional-to-unconditional identity/bound by tower expectation;
it does not make that moment matrix cheap or available. It involves the raw
second moment, including the mean outer product, not covariance alone. At h=0
both thresholds are zero and the residual vanishes exactly; later states need
not have this property.

## 4. What this could bound downstream

Replace only hidden coordinate j at one step by its signed surrogate, while
retaining a+ and every other coordinate. Exact and surrogate states differ by
e_j r. With identical future inputs, ReLU's componentwise1-Lipschitz property
gives after s further recurrent steps

    |h_exact−h_surrogate| <= |Wh|^s e_j r.

This holds even for the signed initial surrogate because the next ReLU map is
defined on every real vector. At the final readout, for any pair y,k,

    |margin_exact(y,k)−margin_surrogate(y,k)|
       <= |Wo_y−Wo_k| |Wh|^s e_j r.

All inequalities are componentwise before the scalar margin. The conditional
expected margin difference is bounded by the same deterministic coefficient
multiplying E[r|h]. A small expected margin difference does NOT establish a
small classification-error probability: margins near zero and tie ownership
still matter. This does not remove the anchor a gate, nor price evaluating the
surrogate, its residual, state moments, or future propagation.

## 5. Domains and untried alternatives

The scalar identity in section1 holds for all finite a,b. The interval and
Gaussian formulas assume opposite NONZERO slopes. Zero slopes, identical-zero
anchor, same-sign slopes, non-Gaussian fresh inputs, and input-rank conditioning
need separate formulas; they are not treated as failures or zero residuals.
The exact identity is not affected by recurrent outgoing weights being large;
useful error bounds can be. This contrasts only mathematically with replacing
weights: no source intervention, model alteration or target test is performed.

Possible explanations for a later unsuccessful application include (i) wide
threshold intervals under the actual state law; (ii) small residuals amplified
by downstream margins; (iii) expensive evaluation/partition requirements even
when residual expectations are small; (iv) cancellation in a naive CDF formula.
Cheapest discriminators would respectively be a separately frozen conditional
state witness/control panel, a saved-weight margin-envelope check, a source-cost
inventory, and exact analytic/interval controls. None is executed or authorized
by this proof. Alternative anchor orientation, multi-hinge grouping, exact
residual integration and state-domain bounds remain open; no family is retired.
