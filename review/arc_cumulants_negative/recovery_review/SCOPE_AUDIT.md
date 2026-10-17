# E35 negative-review recovery: what the saved evidence does and does not establish

The old failures remain. The stronger explanatory theorem and family-wide exhaustion claim do not follow from them. This is a scope correction, not a claim that the cumulant estimator now works.

## 1. Support dimension is different from covariance rank

For X~N(0,1), h=(max(X,0),max(−X,0)) is a bias-free one-step ReLU network driven by one scalar. Its mean is (1,1)/sqrt(2pi), and its covariance has diagonal1/2−1/(2pi) and off-diagonal−1/(2pi). Its eigenvalues are1/2 and1/2−1/pi>0. Thus covariance rank2 exceeds latent dimension1. This refutes the old `rank(Cov h_t)<=t*d` assertion and its use as an exact mechanism theorem.

What remains true: a finite ReLU RNN maps its td continuous input coordinates through a piecewise-linear Lipschitz function. Its hidden-state support lies in a finite union of sets of dimension at mosttd. When H>td, this can be singular relative to H-dimensional Lebesgue measure, even when its covariance is full rank. Singularity does not itself provide a lower bound on error of any chosen moment or output estimate. The observed scalar-versus-vector width trends remain empirical evidence; they are not a covariance-rank proof.

## 2. A fixed endpoint diagnostic is not a universal accuracy ceiling

If a particular endpoint F takes a specified finite vector of moments, and an upstream estimator converges to the true moments, then a nonzero population discrepancy F(m_true)−p diagnoses asymptotic bias of **that fixed composition**, subject to continuity and numerical accuracy. This is a useful scoped statement.

It does not imply that F(m_true) is the most accurate possible output among approximate moment vectors, alternative parameterizations, conditioning, higher orders, positivity-preserving densities or endpoints. Bias cancellation can improve a particular composed estimate. The saved “exact” cumulants are finite-Monte-Carlo estimates, and some endpoint integrations use additional seeded Gaussian draws; neither is exact population evaluation. The corresponding statistical/numerical uncertainty was not uniformly saved.

The new exact finite control shows a different valid limitation: distributions on i−5/2, i=0..5, with alternating binomial weights have identical moments0..4 but positive probabilities5/16 and11/16. Adding identical independent uniform jitter in[−1/4,1/4] makes both continuous while retaining those moments and sign probabilities. Any estimator using only those four population moments must err by at least3/16 on one of these two laws. This is an information bound over arbitrary laws, not proof that both laws are realizable by the fixed trained RNN, or that all weight-aware/conditional/higher-order methods fail.

## 3. Saved completion and model scope

`inventory.json` pins25 output files and identifies43 printed variant rows,42 unique(T,model,variant) rows. The repeated T10 seed1 joint:arc2 line is a duplicate, not an independent replicate.

| Saved variant | Distinct models | Descriptive RMSE from rounded printed values |
|---|---:|---:|
| T4 cell:arc2 |10|.598126|
| T4 joint:arc2 |10|.686864|
| T4 joint:gauss |10|.689209|
| T10 joint:arc2 |3|.752365|
| T10 cell:arc2 |2|.597342|
| T10 joint:gauss |1|.785410|
| T10 cell:gauss |1|.669330|

These are large observed failures, but are descriptive subsets with approximate costs, not a completed eight-model T10 RMSE panel or fresh common-budget audit. Header-only `out_ceiling_T10.txt` contains no completed endpoint value. The README's CEIL10/ARCCELL/RMSE placeholders must remain unknown rather than filled by inference. `QUEUE_DONE` is a shell completion marker, not proof that every planned numerical result exists.

The saved noisy M16T4 s0.01_0 cell readout has K2−24.97pp, K3+0.98pp and K4+0.20pp. The base seed3 K4 result is−2.66pp. Therefore “every tested endpoint at least2.7pp off” is not true across the saved noisy case; preserve separate models and do not silently treat the near hit as robust or competitive. Costs and reference uncertainty still matter.

## 4. Exact inactivity needs a support proof

For any neuron with nonzero fresh-input coefficient b, its preactivation at every step is v(history)+bX with fresh independent Gaussian X and finite history-dependent v. Conditional positive and negative probabilities are both strictly positive. Consequently E[ReLU(v+bX)] is strictly positive after averaging over histories. A very small finite-MC activation mean does not prove the neuron is never active or has exactly zero mean. Exact inactivity is possible only under additional structure, such as b=0 and a proven nonpositive recurrent contribution.

The frozen second-step discriminator now specializes this theorem to the historical H16T4 seed3 checkpoint: all16 input coefficients are nonzero, all exact second-step means are positive, and independent50-digit angular references agree. The three striking small means are .0001583230354, .008062322542 and .005034181112. Gaussian closure still has relative RMS mean error1.494186402 at that step. See prefix_trained.json and README for counts, controls and floating qualifications. No million-sample reference was repeated.

## 5. Status proposal and preserved alternatives

Recommend **reopen the overbroad theorem/exhaustion wording**, retain the individual estimator failures as measured at their exact saved settings, and leave broad “dead (verified)” unconfirmed until an independent review and a scoped alternatives argument exist. The new controls do not erase the large finite errors, nor demonstrate an ARC solution.

Existing numerical tests cover exact Gaussian pair moments, finite ARC orders, several coarse conditioning choices and recent-input grids. They do not logically exhaust positivity-preserving endpoints, structure-aware exact-prefix moments, arbitrary conditioning or every higher-order scheme. These alternatives need justified bounded tests, not automatic promotion or dismissal. The next proposed discriminator is an analytic exact-prefix moment calculation, followed only if warranted by a separately frozen closure rescue.
