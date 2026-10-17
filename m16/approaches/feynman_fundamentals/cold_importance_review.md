# T20 cold proposal and objective review

Read-only source review of `tube_importance_port/estimator.py` and bounded
results throughd5a2a20. The reused interval-native walker was reviewed
previously; this does not rerun those geometric controls.

For K equally weighted target strata and a positive proposal q_j, the
importance multiplier is1/(K*q_j). The 20% uniform floor retains every
stratum. Conditional expectation of each contribution therefore equals
the uniform-stratum target for iid production draws given the proposal.
This algebra does not establish unbiasedness for the implemented fixed
deterministic nodes, or independence of deterministic pilot/production
discrepancy. Pilot values are not reused in production, and model-specific
pilot fitting occurs on every cold call.

The code's standard-deviation/sqrt(cost) score comes from optimal
stratified allocation, not mixture-importance allocation. For iid mixture
sampling with conditional second moments M_j and costs c_j, the variance
times expected cost is

`(sum_j M_j/(K^2*q_j) - mu^2) * sum_j q_j*c_j`.

RMS/sqrt(cost) minimizes the corresponding second-moment-times-cost
upper objective by Cauchy-Schwarz, but need not minimize this exact
variance objective when costs differ and the integral mean is nonzero.
Neither prescription is automatically optimal for deterministic nodes.
An RMS rescue is consequently a reasonable separately frozen hypothesis,
not a theorem of optimality or an excuse to erase the standard-deviation
variant's outputs.

The implementation charges all2304 cold pilot faces, proposal fitting,
table creation, transforms, production weighting and blind-region work
in its explicitly nonportable numerical proxy. Setup can exceed the
advisory budget. Original advice fitting remains unknown and is labeled
incomplete, so its reported partial cost cannot be used as a complete
equal-cost comparator. Failed walks propagate known partial operations;
unresolved opaque failure work remains flagged.

The independent positive scalar face check reproduces zero-readout
aggregates without recurrence or incomplete-beta calls. It localizes
that fixture's remaining discrepancy from exact accuracy.1 to the outer
rule, rather than the tested radial weighting. This is a diagnostic
control, not an independent robust-MSE or complete-cost certification.
No noise sweep, sampling win, or importance-family negative is implied.
