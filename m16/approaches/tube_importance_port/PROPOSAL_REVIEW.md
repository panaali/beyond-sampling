# Proposal objective review

This analysis concerns the existing T20 pilot; it makes no stochastic-unbiasedness claim for the deterministic production nodes.

For K equal-mass strata, let conditional face mass F have mean m_j, variance v_j and second moment s_j. The target is mu=(sum m_j)/K. A random mixture chooses J with probability q_j and returns F/(K q_j). Its second moment is sum s_j/(K² q_j), and variance subtracts mu². Thus, at a fixed number of draws, q_j proportional to sqrt(s_j) minimizes variance. Using sqrt(v_j) can suppress a stratum with constant nonzero contribution; the uniform20% floor prevents complete loss of support, but not inefficient weighting.

The implemented sqrt(v_j/c_j) score is instead the continuous allocation solution for independent, separately averaged strata with predetermined counts n_j and cost sum n_j c_j: the estimator averages the K conditional means, has variance sum v_j/(K² n_j), and the Lagrange equations give n_j proportional to sqrt(v_j/c_j). The production code selects mixture strata through a transformed Sobol coordinate; it does not construct K separately averaged fixed-count estimators. Calling the current score optimal would therefore be wrong.

A cost-adjusted sqrt(s_j/c_j) proposal minimizes the product of expected per-draw cost C=sum q_j c_j and the *uncentered* second moment sum s_j/(K² q_j), by Cauchy–Schwarz. It need not minimize the actual cost-times-variance C[sum s_j/(K² q_j)-mu²], because the subtracted term depends on q through C. Neither argument proves superiority for deterministic quadrature, where errors depend on transformed geometry and node structure. Model-dependent pilot fit costs also enter any total-budget comparison.

A discriminating synthetic control is F=a_j>0 constant in each stratum. The current within-stratum-variance score falls back to uniform even for very unequal a_j; the fixed-count mixture optimum q_j=a_j/(sum a_j) makes every weighted contribution identical. A frozen RMS proposal can test this mathematical distinction, but should not be tuned to the observed neural estimates or be advertised as globally optimal. Keep both objectives and existing outputs visible.

Review status: author derivation, independent review requested. No method has been ruled out and no production rule changed by this note.
