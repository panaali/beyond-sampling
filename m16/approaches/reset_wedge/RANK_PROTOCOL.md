# Frozen next phase: exact last-pair rank factorization

The last certified pair ending at j restricts only inputs j−1,j and future pair avoidance. The reset input Y is negative almost surely, and the first future input must be positive; further future avoidance is internal to the suffix. Consequently old j−2 iid inputs, the wedge-conditioned reset pair, and the surviving suffix factorize. Earlier pair events are unrestricted. T55 independently reached this factorization before receiving our derivation.

For E={Y<=−kappa*max(X,0)}, let G_l(t)=P(E, exactly l of X,Y exceed t). Then the discarded-prefix count PGF is (Phi(t)+z*Phi(−t))^(j−2) * sum G_l(t)z^l / r. Use unnormalized G to carry the reset-pair probability into integration.

For t>=0: G0=.25+integral_0^t phi(x)Phi(−kappa*x)dx; G1=integral_t^infinity phi(x)Phi(−kappa*x)dx; G2=0.

For t<0: u=Phi(t), A=−t/kappa. G0=u²; G2=(.5−u)²+integral_0^A phi(x)[Phi(−kappa*x)−u]dx. Evaluate G1 positively as 2u(.5−u)+u[Phi(A)−.5]+integral_A^infinity phi(x)Phi(−kappa*x)dx. Handle kappa=0 by independence with Y<=0, and infinity by independent negative X,Y.

Two falsifiers frozen before implementation:
1. Threshold splits or normalization may be wrong. Compare every coefficient against a separate direct rectangle integration over Y first, for slopes0,.1,1,3,infinity and thresholds−3,−.2,0,.2,3. All three masses must be nonnegative and sum to r. Compare convolution to direct finite binomial expansion, including old-prefix length0.
2. Conditioning on LAST rather than FIRST may have been misused. Independently integrate a short four-input certificate pattern at fixed first future positive/negative values, checking older-prefix count stays binomial and pair rank kernel is independent of the positive suffix magnitude. This can be shown by direct event logic; no random model inputs required. Record that future-negative implies impossible surviving suffix.

Initial code/control budget<=20CPU under review lock. Adaptive quadrature reports callbacks and estimates; it is not a certified error bound. Clip integration at8 only with an explicit Gaussian tail allowance, never hidden truncation. No trained accuracy sweep, no unsupported independence, no portable-cost claim.
