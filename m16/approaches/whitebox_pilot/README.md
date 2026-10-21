# T473: first-layer control variate for white-box activation means

## Question

Can an exact first-layer Gaussian expectation remove enough sampling error to
beat ordinary sampling and the supplied covariance-propagation baseline, after
charging all computation with the challenge meter?

## Method

For Gaussian input, let `h1 = relu(X W1)` and
`m1[j] = ||W1[:,j]|| / sqrt(2 pi)`. This mean is exact. Construct fixed,
weight-dependent matrices implicitly through the recurrence
`delta1 = sample_mean(h1) - m1` and
`delta_l = (delta_(l-1) W_l) * gain_l`. Return
`sample_mean(h_l) - delta_l` at every layer. The gains are Gaussian activation
probabilities from diagonal moment propagation, computed without sample fitting.
This is the shallow surrogate `g_l(X) = (h1(X)-m1) A_l`, whose mean is exactly
zero regardless of the accuracy of those gains. Linearity lets us evaluate its
sample mean after aggregation, avoiding an extra network pass per sample.

Two competing explanations, fixed before experiments:

1. The first layer explains too little of deep output variation. Measure
   variance across repeated estimates and paired residual/sample covariance.
2. Useful variance reduction is outweighed by metered setup/correction cost.
   Measure total FLOPs and the final-layer adjusted score, not variance alone.

`estimator.py` is the candidate; `pilot.py` is development/scoring code and is
not part of a submission. Baselines are loaded unchanged from the official
starterkit. Its random-output example is not Monte Carlo; our sampling control
uses the same forward/sample accumulation as the candidate with correction off.

## Preregistration (2026-10-21, before any model scoring)

- Starterkit: `96e308c518c76a1ae119f1d4c8913bc2e019eee4`.
- Meter: flopscope `0.12.3`; harness: whestbench `0.16.2`.
- One candidate: diagonal-moment soft gains, coefficient one, no fitted knobs.
- Small controls: width/depth `(32,4)` and `(64,16)`; DEV weight seeds 0,1;
  held-out seeds 100,101,102,103. Freeze implementation after DEV, before held-out.
- Four estimator RNG replicates per model; 512 samples each. Budget is ten
  times the nominal forward cost for 512 samples, so sampling is near the 10%
  floor. This is a scaled pilot, not a Phase 2 leaderboard score.
- Reference: 65,536 independent Gaussian inputs, in chunks of at most 2,048;
  report estimated reference MSE, raw scores, and reference-noise-corrected
  diagnostics separately. No precise ranking if reference noise dominates.
- Candidate go to full-size feasibility if DEV aggregate adjusted score is
  below plain MC and the controls pass. Covariance comparison remains decisive
  for competitiveness, but is not required to inspect scaling.
- Full-size gate: width 1,024, depth 16; DEV seed 0 and held-out seed 100;
  6,500 estimator samples, actual budget `2**41`; 65,536 reference samples per
  model, only if projected resource use stays within the cap. One RNG replicate;
  these two networks cannot establish a benchmark-wide win.
- One process, at most two BLAS threads, under 2 GB RSS and 600 CPU seconds for
  the bounded run. Stop between chunks; memory/time checks are runner plumbing.
  macOS does not support the Linux `ulimit -v` recipe reliably, so use RSS and
  CPU watchdogs. No unbounded run, full dataset download or submission.
- Register all tested variants and scores, including failures. No own claim is
  confirmed, and no finite pilot rules out other surrogates.

## Status

Implementation and controls in progress. Challenge remains unsolved.

## Results

Pending. Numerical summaries will be registered once in [INDEX](../../../INDEX.md).
Machine-readable measurements will live under `results/` in this folder.

## Compute accounting

Every estimator numerical operation uses flopscope inside one BudgetContext:
random draws, moment propagation, network evaluations, reductions and corrections.
No fitting or offline precomputation is used. Reference generation and scoring
are outside the estimator budget and recorded separately. Use the official
`MSE * max(0.1, FLOPs / budget)` score. Also report score/MC ratios against both
project bars (strict <=1, lenient <=10), explicitly distinct from AlgZoo F.

## Claims

- [inferred] The correction is unbiased in exact arithmetic: the sample-mean
  first-layer error has mean zero and every subsequent transformation is fixed
  and linear. Approximate Gaussian gains change variance, not this argument.
- [speculative] First-layer fluctuations explain enough deep-layer variation to
  beat covariance propagation at challenge scale.

## Alternatives not ruled out

Full-covariance gains, fitted coefficients with independent fitting data,
multiple exact-mean features, antithetic inputs, radial conditioning, and shallow
ReLU surrogates are not tested. They remain open; there is no novelty claim.

## Failure modes

The surrogate can amplify noise when gains are poor. Floating-point cancellation
and empirical-reference noise can mask the benefit. Small widths need not predict
Phase 2 performance. Local BudgetContext checks do not certify remote submission
eligibility, subprocess behavior, or benchmark-wide performance.
