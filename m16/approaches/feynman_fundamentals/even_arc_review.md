# Even arc power recurrence and positive beta rescue

T55 mathematical review supporting T102, 2026-10-17. No production primitive was changed.

For f=a cos(s)+b sin(s), g=f', and R^2=a^2+b^2, differentiation of f^(n-1)g gives `(n-1)*R^2*f^(n-2)-n*f^n`. This proves the real arc recurrence. It does not make the recurrence stable: near f=sin(s) on a tiny arc, the two terms cancel to high relative order. Stable half-angle endpoint differences remove one subtraction but not this fundamental recurrence cancellation.

Center the interval at theta/2 and write f=A cos(t)+B sin(t). For even n and 0<=theta<=pi, odd powers of sin(t) integrate to zero. Every remaining term is nonnegative and its integral is

`binom(n,j)*A^(n-j)*B^j*B_z((j+1)/2,(n-j+1)/2)`, j even, z=sin(theta/2)^2.

There is no extra factor two. The symmetric factor two cancels the half from the substitution. For n=0 return theta directly. The beta expression is a real-arithmetic identity, not a free or certified numerical operation. Squaring a tiny sine can underflow prematurely; near pi, z can round to one. The complementary beta expression and the hypergeometric expression `sin(theta/2)^(j+1)/p * 2F1(p,1-q;p+1;z)` repair these coordinate losses in appropriate domains, with floating special-function and residual cancellation qualifications.

## Independent reference failure diagnosis

The original tiny sin^6 reference from unscaled adaptive quadrature was 1.42857141061152056e-85 at stored binary64 theta=1e-12. An independent elementary bound is

`theta^7/7-theta^9/9 <= integral sin(s)^6 ds <= theta^7/7`.

It follows from sin(s)>=s*(1-s^2/6), Bernoulli's inequality, and sin(s)<=s on [0,1]. `check_arc_oracle_review.py` evaluates this at 180 digits. The relative bracket width is 7.78e-25, while the old reference lies far below it. Thus the reported roughly 1.26e-8 discrepancy was a reference-resolution failure, not evidence against the positive beta identity. The raw reference and failed comparisons must remain saved.

For a normalized adaptive reference, integrate over u in [0,1], divide f(theta*u) by `abs(a)+abs(b)*min(theta,1)`, and multiply the result by theta times that amplitude to power n. The scaled integrand is bounded by one; handle zero amplitude separately. This improves numerical resolution but is still not a rigorous quadrature enclosure. The elementary sin^6 bound is independent of that rescue.

## Endpoint source and fresh orientation control

Read-only inspection of `real_arc_tail/recurrence.py` confirms the centered coefficients A and B, even binomial terms, beta parameter order and complementary endpoint branch. The polygon wrapper still uses the directed short-arc tangent from u to v and separately selects the outward conormal; reversing vertex order preserves the geometric arc integral. It replaces only the matching-parity moment lookup in a private module, retaining existing polygon guards and positive fallback. These guards do not convert ordinary special-function arithmetic into a certified enclosure.

`check_positive_arc_review.py` uses fresh a=.37, b=-.81 and theta=.73/2.4, degrees zero through eight, covering the hypergeometric and complementary-beta branches. Independent normalized 80-digit positive integration differs by at most 6.77e-16 relative. Reversing each arc gives identical binary64 values in this fixture. CPU was 0.179 seconds. This is a primitive/source check; the author's eight trained conditional queries and timings were not duplicated. Integer powers, beta/hypergeometric kernels, geometry and fallback work remain charged separately, so the finite speedup is not a portable full-estimator cost certificate.
