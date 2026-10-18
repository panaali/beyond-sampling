# What the finite training study can establish

This is a written interpretation contract, not a replacement for the owner's
T234 preregistration. No training, checkpoints, numerical arrays or MC were run.

## Achievable performance versus architecture ceiling

For the declared architecture let C=sup_theta P(correct_theta(X)), X~N(0,I).
Any evaluated feasible weights theta give C>=P(correct_theta). Thus a valid lower
confidence bound for one fixed trained model is also a lower bound for C.
An upper confidence bound for that model is NOT an upper bound for C. Failure of
finitely many initializations, optimizers, schedules, data budgets or enriched
objectives cannot establish an architecture ceiling without an additional
representation theorem or certified global bound. This applies equally if five
runs plateau or their seed-average gain is small.

Compare distinct questions explicitly: best feasible checkpoint found; mean
performance under the specified training procedure; paired change for fixed
seeds; and the unconstrained architecture supremum. A wide-network reference is
an empirical different-architecture comparator, not a proof of a narrow-network
ceiling. Weight decay, finite precision and the searched optimizer family may
restrict attainable results independently of representational capacity.

## Fixed-model evaluation uncertainty

For a model frozen before an independent iid evaluation stream, correct-count
K out of N is binomial under the stated target law. Report point estimate and
an explicitly selected interval method with N and confidence level. For the
accuracy p, SE=sqrt(p(1-p)/N)<=1/(2 sqrt(N)); atN=10^8 this upper bound is.005
percentage points. This is MC uncertainty conditional on the checkpoint, not
variation among five training seeds and not a guarantee about future training.
Pseudo-random/floating execution and provenance remain ordinary qualifications.

For independent evaluation streams, paired-seed accuracy differences have MC
variance p_a(1-p_a)/N_a+p_b(1-p_b)/N_b. Training-seed variation is separate and
must be reported across all five paired differences. If the owner instead uses
common test inputs, retain per-input paired correctness to account for covariance;
do not use the independent-stream variance formula silently. Pairing training
seeds and pairing test inputs are different designs. No new evaluation convention
is imposed here, and this non-ARC capacity study is not a rung declaration.

## Stopping and selection

The requested 'gain<.05 percentage points for two doublings' is an operational
stop. Freeze whether gain means seed-average paired gain, best checkpoint gain,
or all-seed gain, and whether the criterion uses point estimates or uncertainty.
A small measured gain is not itself evidence of equivalence or of a permanent
plateau. For a no-material-improvement interpretation, an upper one-sided bound for
the relevant gain must fall below the threshold; an interval containing that
threshold is inconclusive. This does not establish two-sided equivalence: a
large deterioration can also satisfy it. Equivalence requires both confidence
bounds to lie within a separately declared two-sided margin. Preserve every doubling and the actual stopping reason.

Repeated checkpoint inspection, best-size selection, LR/schedule search and
hard-case tuning use data to choose a candidate. Ordinary pointwise CIs do not
become simultaneous or selection-adjusted merely because N is large. Options
include a predeclared finite family with simultaneous intervals, a valid spending
rule over sequential fresh evaluations, or a final fresh evaluation of the
selected checkpoint after all choices are locked. For adaptive choices with
fresh independent test draws, conditional per-look error budgets alpha_i with
sum alpha_i<=alpha support a union bound; historical reused validation cannot be
retroactively made fresh. Keep exploratory and final confirmation records distinct.
These are interpretation choices, not a demand to add training or override the
owner's data/compute budget.

## Hard-case data and error anatomy

Specify the training mixture, near-tie threshold/rank-gap definition, enrichment
probabilities and any loss weights. Fresh N(0,I) evaluation tests the requested
population even when training uses another law. Accuracy on enriched training
examples alone does not estimate Gaussian accuracy. Resampling, importance
weights and mixture schedules have real optimization and compute effects.

For T10 iid centered Gaussian signs, the blind event K_positive<=1 has exact
probability(1+10)/2^10=11/1024. That gives an input-event probability, not its
conditional error rate or the total blind-error contribution. Near-1/2 and
near-2/3 ties can overlap that event and each other. Define disjoint bins with
fixed precedence, or retain the full intersection table; do not sum overlapping
error fractions. For disjoint strata, total errors must equal their sum, and
P(error)=sum_s P(s)P(error|s). Report numerator/denominator per stratum; distinguish
conditional error from joint error and describe rare/zero-count uncertainty.
The rank/first-index tie semantics and exact meaning of 'near' must be frozen.
Co-area face-weight contributions are not interchangeable with actual-input
anatomy bins. Nothing here infers a universal blind-error floor.

## Current evidence status

Only the owner task-row request and local alg-zoo defaults are available at the
frozen author branch. The study's driver, checkpoints, optimizer state, final
MC counts and selection logs were not published there at inspection time.
Their absence limits what this review can verify; it does not assert failure
or stop the owner's authorized work. A later saved-result audit should be
separately claimed and pinned after those artifacts are available, with no
training or MC replay by default.
