# What a family test can and cannot establish

This is a source/theorem design under claim1f396509, not a new experiment.
Let K be a tie-face kind chosen uniformly from {12,23}, let Y be the face drawn
from its normalized law, and X=Z_K F_K(Y), using the source's notation. The
ideal co-area error identity is e=2 E[X]. The inherited floating inverse-CDF,
truncation and walker do not become exact by this notation. Let L be any
measurable finite family label. Assume finite second moments.

Write p_l=P(L=l), mu_l=E[X|L=l], v_l=Var(X|L=l). Total variance gives

    Var(X) = sum_l p_l v_l + Var(mu_L).

The oracle residual X-mu_L has variance sum p_l v_l. This is a representation
fact, not an available estimator. If p_l and frozen fitted constants a_l are
known, 2[sum p_l a_l + average(X-a_L)] is unbiased under independent iid
validation draws; its variance is4 Var(X-a_L)/N. Pilot-fitted a_l may be
conditioned on when pilot and validation are independent and p_l remains exact.
Unknown p_l estimated from the same validation sample cannot be substituted
without an analysis: using its empirical proportions makes this expression
algebraically equal to2 average(X), eliminating the proposed gain. An independent
probability estimate introduces its own uncertainty/cost, not a free constant.

For exact conditional iid sampling within each positive-probability family
(p_l>0), using independent sample sets across strata and n_l>0, estimator
2 sum p_l average(X|L=l) has variance4 sum p_l^2 v_l/n_l; these sums
exclude p_l=0 families, whose conditional moments need not be defined. If each conditional draw
costs c_l>0, an ideal continuous allocation of budget B has oracle variance
lower bound/infimum
4(sum p_l sqrt(v_l c_l))^2/B (Cauchy-Schwarz). It is attained by the
proportional allocation only in the admissible positive-variance interior;
zero-variance families with mandatory positive minimum allocations can make
the bound unattainable. Integer allocations, setup,
probability estimation, rejection, and pilot costs still count. This is only an
oracle benchmark when conditional sampling or p_l is unavailable. Rejection
using an outcome-derived label can require the full network/fibre computation
for every rejected proposal; charge all proposals. It is sampling, not the
required deductive ARC solution. No iid variance or unbiasedness theorem is
transferred to the two existing fixed scrambled Sobol rules.

A high tail-coverage fraction alone implies no variance reduction. For example,
let X be a fair0/1 variable and L independent, P(L=0)=.95. Label0 covers95% of
all positive X^2 weight, but mu_l=.5 for both labels: residual variance is
unchanged. In contrast L=X covers all positive weight with one label and makes
oracle conditional variance zero. These analytic populations are explanations,
not descriptions of the saved target data.

## Fixed-rule diagnostic when full arrays are available

For each rule A/B, use every face, including zero integrands, with its original
kind/index and fixed equal quadrature weights. Fit family means a_l on A only;
unseen B labels receive the A pooled mean. Compare mean_B[(X-a_L)^2] against
mean_B[(X-mean_A X)^2], recording both raw squared residuals and their means.
This is a deterministic out-of-rule prediction diagnostic, not an unbiased
variance estimate, estimator MSE or sampling CI. The same comparison on centered
residuals must be reported separately; do not select whichever is more favorable.
Reverse B-to-A is secondary and never used to choose labels/thresholds.

Source coarse labels require the full scalar fibre walk: dominant error
component, winner relation and gate flips. They therefore are outcome-derived,
not cheap mechanistic features. Even a successful residual diagnostic does not
provide an implementable inexpensive classifier, known family probabilities,
conditional sampler, or deterministic conditional integrator. A practical
follow-up would need a weight-derived rule and a complete cost/accuracy test.
