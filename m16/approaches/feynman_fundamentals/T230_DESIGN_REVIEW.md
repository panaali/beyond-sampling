# Independent T230 design and theorem review

This is a written requirements review, not execution clearance. No author module, array, model, fit, random generator or integrator was executed. The attached JSON binds the reviewed document versions and seven inherited source/evidence files. T227's existing numerical and provenance qualifications remain unchanged.

## Population identities

Conditional on a finite measurable label L and finite second moments, write X = mu_L + R, where E[R | L] = 0. Consequently Cov(R, mu_L) = 0 and Var(X) = E[v_L] + Var(mu_L). More generally, for frozen constants a_l, Var(X-a_L) = E[v_L] + Var(mu_L-a_L). Thus conditional means minimize residual variance, but knowing their existence supplies neither their values nor a cheap label algorithm.

With exact p_l and validation observations independent of the fitting pilot, E[sum p_l a_l + mean(X-a_L) | pilot] = E[X]. Independent iid validation then gives variance 4 Var(X-a_L)/N for twice that expression, conditional on the pilot. When the conditional mean remains the same across pilots, unconditional variance is the expectation of this conditional variance. Substituting empirical label proportions from those same validation observations cancels sum p_hat_l a_l against mean(a_L) exactly and returns twice the ordinary sample mean. An independently estimated p_l does not enjoy that cancellation and requires its own error/cost analysis.

For positive-probability strata with independent sample sets across strata, conditional iid means give variance 4 sum p_l^2 v_l/n_l. Cauchy–Schwarz applied to (p_l sqrt(v_l/n_l)) and (sqrt(c_l n_l)) gives the stated lower bound 4(sum p_l sqrt(v_l c_l))^2/B when sum c_l n_l <= B. With all v_l positive and continuous unconstrained allocations, equality holds for n_l proportional to p_l sqrt(v_l/c_l). With zero-variance strata and strictly positive allocations, it is generally an infimum approached as their allocation tends to zero, not an attained positive allocation. If all v_l vanish, any feasible allocation has zero variance. Positive integer minima, setup and sampling feasibility can only make this oracle relaxation less informative; no implementability follows from it.

The co-area normalization e=2 E[X] is conditional on the inherited ideal identity and uniform choice of the two face kinds. This review checks the factor-of-two bookkeeping against the source's sum of two face expectations; it does not reprove the underlying geometric identity or certify the floating CDF/walker implementation. Equal fixed numbers of Sobol points per kind do not make their combined points iid draws from the uniform-kind population.

## Coverage versus regularity

For a fair Bernoulli X and independent label with P(L=0)=.95, E[X^2 1(L=0)]/E[X^2]=.95 while E[v_L]=Var(X). This is a population weighted-coverage identity, not a guaranteed finite-sample fraction. For L=X the oracle within-label variance is zero, but constructing L already observes the outcome. Both examples validly distinguish coverage from an available estimator.

The proposed A-to-B squared-residual comparison is a deterministic prediction diagnostic on fixed arrays. Unseen labels use the frozen A pooled mean, and zero baseline denominators remain undefined. Raw residual second moments and separately centered residuals answer different questions; preserving both prevents selection of a favorable interpretation. Neither is automatically estimator MSE, a sampling variance, an unbiased variance estimate or a confidence interval. Reusing already inspected rules is descriptive.

## Source and protocol checks

The pinned tail driver first computes vector integrands, selects their global X-squared tail, then runs scalar labels only on that selected subset. Its coarse label is exactly (kind, ub, wt, min(nflips,3)); scalar labels require the fibre walk and dominant component. Python max keeps the first component on equal masses. The source uses rounded break locations and tolerances; preserving its label definition does not certify label exactness. The missing full-face labels/values cannot be filled with zero.

The proposed full-array contract retains zero, unknown, failed/capped and nonfinite statuses. Its explicit ZERO extension is restricted to completed source-zero/no-component rows and is distinct from unknown. Actual half-rule testing must select a new tail from all points of that smaller rule; thinning a previously selected tail does not do that. Byte-prefix agreement is necessary for reusing a saved Sobol prefix, and weights must be those of the smaller rule. Otherwise a separately authorized evaluation is required.

Permuting the whole label multiset separately within each rule and kind preserves occupancy, names and zero counts while breaking the observed assignment. Independent A/B permutations are appropriate to the specified null diagnostic, but interpretation is conditional on the explicit exchangeability null and frozen arrays. It is not a Gaussian/quadrature confidence bound. Tail-only permutation is a different, narrower null. Full assignments and the finite shuffle specification must be frozen and retained before any future test.

The requirements correctly charge outcome-derived labels, rejected conditional proposals, unknown probabilities, setup, pilot selection, validation and shuffling. No exact family probabilities, conditional sampler, inexpensive classifier or deterministic family integrator is supplied. A successful residual diagnostic would leave those gaps open.

## Disposition

The design is sound with the final explicit positive-probability/independent-strata and zero-variance-allocation qualifications. The current documents supply no executable target preregistration: target/rule identities, finite repetitions, numerical thresholds, resource envelope, complete arrays and separate source gates are still required. T222/T223 priorities and Claude's scientific adoption authority are unchanged. No numerical approval, broad family rejection or cost advantage is granted.
