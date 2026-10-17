# Frozen next phase: exact last-pair rank factorization

The last certified pair ending at j restricts only inputs j−1,j and future pair avoidance. The reset input Y is negative almost surely, and the first future input must be positive; further future avoidance is internal to the suffix. Consequently old j−2 iid inputs, the wedge-conditioned reset pair, and the surviving suffix factorize. Earlier pair events are unrestricted. T55 independently reached this factorization before receiving our derivation.

For E={Y<=−kappa*max(X,0)}, let G_l(t)=P(E, exactly l of X,Y exceed t). Then the discarded-prefix count PGF is (Phi(t)+z*Phi(−t))^(j−2) * sum G_l(t)z^l / r. Use unnormalized G to carry the reset-pair probability into integration.

For t>=0: G0=.25+integral_0^t phi(x)Phi(−kappa*x)dx; G1=integral_t^infinity phi(x)Phi(−kappa*x)dx; G2=0.

For t<0: u=Phi(t), A=−t/kappa. G0=u²; G2=(.5−u)²+integral_0^A phi(x)[Phi(−kappa*x)−u]dx. Evaluate G1 positively as 2u(.5−u)+u[Phi(A)−.5]+integral_A^infinity phi(x)Phi(−kappa*x)dx. Handle kappa=0 by independence with Y<=0, and infinity by independent negative X,Y.

Two falsifiers frozen before implementation:
1. Threshold splits or normalization may be wrong. Compare every coefficient against a separate direct rectangle integration over Y first, for slopes0,.1,1,3,infinity and thresholds−3,−.2,0,.2,3. All three masses must be nonnegative and sum to r. Compare convolution to direct finite binomial expansion, including old-prefix length0.
2. Conditioning on LAST rather than FIRST may have been misused. Independently integrate a short four-input certificate pattern at fixed first future positive/negative values, checking older-prefix count stays binomial and pair rank kernel is independent of the positive suffix magnitude. This can be shown by direct event logic; no random model inputs required. Record that future-negative implies impossible surviving suffix.

Initial code/control budget<=20CPU under review lock. Adaptive quadrature reports callbacks and estimates; it is not a certified error bound. Clip integration at8 only with an explicit Gaussian tail allowance, never hidden truncation. No trained accuracy sweep, no unsupported independence, no portable-cost claim.

## Bounded partial-accuracy phase (frozen after rank primitive controls)
Evaluate only last pair ending at T and T−1 on the three already frozen H2T8 copies. This is a nonnegative contribution to total accuracy; all other events remain unresolved. At T the final hidden state is exactly zero, so first-index argmax predicts label0. There are N=T−2 exchangeable old labels. Compute its correctness by an independent identity: r=N*p_old+p_X+p_Y, integrating the two labeled reset-pair second-maximum probabilities directly. Cross-check p_old with the marker integral phi(t)*[(N−1)sf(t)Phi(t)^(N−2)*G0(t)+Phi(t)^(N−1)*G1(t)].

At T−1 the surviving suffix is one positive input v, so hidden=Wi*v and prediction is the first maximizer of Wo*Wi, independent of magnitude. Freeze before inspecting: if the prediction is final label T−1, use integral_0^infinity phi(v)*[N*sf(v)Phi(v)^(N−1)*G0(v)+Phi(v)^N*G1(v)]dv with N=T−3. If not, record unsupported in this limited primitive, do not silently substitute a label. Independent reference exchanges integration order and integrates reset-pair X, analytically integrating the suffix order-statistic interval with incomplete beta; this avoids the production rank kernel.

Falsifiers: (1) label0 tie or one-step ray argmax misidentified: compute exact-rational dot products/argmax from binary checkpoint weights and direct hidden-zero case; preserve unsupported labels. (2) missed joint rank dependence: compare marker and labeled-pair identities, plus swapped-order beta integral. Positive quadratures report callback/error estimates and explicit Gaussian-tail allowances; ordinary quadrature errors are not rigorous bounds. Initial budget<=20CPU, no MC, no full accuracy or MSE claim. Numerical errors plus omitted event mass prevent treating the partial sum as a formal floating-point enclosure.
