# Independent directed Chebyshev review

T55 review of T78, 2026-10-17. Production source was read only. This reviews stored-simplex analytic bound arithmetic, not the original floating geometry or numerical mass integrals.

## Algebra and source

For s=sqrt(ab), z=(a+b)/(2s)-1, r=(b-a)/(a+b+2s), the nonconstant coefficient of (m+delta*x)^(-nu) is (-r)^k P_nu(k,z)/s^nu. Differentiation at fixed delta gives

`P_(nu+1)=((k+nu+nu*z)*P_nu+(z*z+2*z)*d_z P_nu)/nu`.

The implemented positive rational monomial recurrence agrees. The stable expression `z=(b-a)^2/(2*s*(a+b+2*s))` avoids subtracting nearly equal quantities. Tail index N=degree+1 excludes the exceptional constant Chebyshev coefficient. Shifting k=N+j and using positive Eulerian geometric moments is valid. The positive quadrature remainder multiplier for dimension 2nu is `Gamma(nu)*abs(det)/(pi^nu*(2nu-1)!)`; at nu=3 this is det/(60*pi^3). Polynomial exactness remains a separate, rule-specific assumption.

Decimal input conversion, directed arithmetic, exact rational coefficients and Machin pi enclosure are consistent. Square root is deliberately rounded nearest and then bracketed with adjacent Decimal values, rather than assuming directed sqrt. Underflow and nonpositive denominator bounds raise. The float-export upper endpoint could become infinity after its initial finiteness check; this was reported to the owner and fixed separately in eb0ff4a. Infinity may be a valid loose bound, but must match the declared API.

For the stored T6 rays, the exact Gram minimum bounds squared radius below, maximum squared vertex norm bounds it above. The supporting-plane bound uses the minimum nonnegative projection divided by the squared normal norm. Exact Fraction elimination computes the determinant. Expanding these endpoints outward gives a valid larger radial interval; monotonicity of the tail formula itself is not needed.

## Independent falsifier

`check_directed_tail_review.py` uses separate closed expressions for nu=1,2,3 and 200-digit arithmetic, rather than the production coefficient recurrence. Eighteen cases cover equal endpoints, adjacent binary64 endpoints, tiny scales, a wide interval, and a degree that underflows ordinary binary64 powers. Every Decimal interval contains its independent reference. Float exports either enclose it or explicitly reject overflow. Recorded CPU: 0.00598 seconds. Results are in `directed_tail_review_results.json`.

Alternative failure explanations checked were a coefficient/index normalization mistake and loss of directed containment at small or large scale. This finite check does not prove arbitrary precision-context behavior. Setup polynomial/pi kernels and rational integer arithmetic are not a portable FLOP audit. A separate binary64 vectorized implementation remains a cost alternative, requiring its own source review.
