# T33: DEV-only proxy importance-density diagnostic

## Question

Can a cheap weight-derived proxy identify wide error bands well enough to justify
building a change of measure for deterministic tie-face integration?

## Method

Before numerical runs, freeze: H=16, T=10; exact leaderboard DEV copies
`dev_s0.0025_0`, `dev_s0.02_0`; both R23 and R12 face laws; 16384 IID diagnostic
faces in each of two independent splits (seeds 629031 and 629032). No formula is
selected or tuned between splits. T5 is omitted unless a local checkpoint exists.

Proxies use one exact value+tangent pass at the tie: (1) tied-pair linear wrong
mass on the finite line window, (2) all-logit linear wrong mass on that window,
and (3) the model-free window mass. For both linear proxies also test fixed floors
`max(g, eta*window_mass)`, eta=0.01 and 0.1. All proxies are bounded above by the
window mass. These are frozen formulas, not fitted accuracy or noise estimates.
They are related to E15's linear tube, whose mean bias alone does not determine
its usefulness as an importance proposal.

Alternatives specified before experiments:

1. A biased linear proxy might nevertheless predict the rare large F values.
   Test its correct importance second moment and its correlation within the top
   1% of F, not just mean bias or overall correlation.
2. Proxy zeros or underpredictions could make the importance estimator worse,
   even if most high-F faces are captured. Test fixed positive floors and report
   residual heavy tails under the **changed proposal law**.
3. Apparent gain might vanish after proxy work or be an accidental rare-tail
   fluctuation. Charge implemented proxy work and proposal-weighted exact-walk
   cost; reproduce every setting on the independent fixed held-out split.

For base face density p, proposal q=p*g/Q, Q=E_p[g], and exact mass F:

`Var_q(F*p/q) = E_p[g] E_p[F^2/g] - E_p[F]^2`.

Tail diagnostics sort F/g and use weights g/sum(g) to take the top 1% **under q**;
second-moment contributions are proportional to F^2/g. They never report a
uniform-face tail of F/g as if it were a proposal-law tail. Raw proxies with any
observed g=0 and F>0 fail the support test and have infinite estimated variance.

This is a sampling-based diagnostic of a potential deterministic transform. It
does not construct or sample q, know its exact normalizer, or establish a QMC
convergence rate. Any reported cost-adjusted gain is an optimistic proposal
ceiling that excludes unimplemented density construction and transformation.

## Status

Preparing code. Awaiting reviewed exact-walker fix before numerical conclusions.

## Results

Pending. No greater-than-threefold gain will be claimed without a held-out,
no-selection audit. No leaderboard test model is evaluated.

## Compute accounting

All diagnostic exact walks, proxy forward/tangent operations, envelope operations,
CDFs, and input generation are recorded. The marginal proposal scenario includes
the expected exact-walk cost **under q** plus per-point proxy work. Construction
of a normalized transport is not implemented and is not assigned a fictional zero
cost. Diagnostic setup and evaluation work are reported separately.

## Claims

- [speculative] A linear proxy may be useful for importance weighting even though
  linear extrapolation itself is a poor accuracy estimator.

## Alternatives not ruled out

Conditional cumulant proxies (T35), other adaptive transforms, and more
independently trained DEV models remain untested. Finite reference samples can
underestimate disastrous rare underprediction tails.

## Failure modes

Proxy support gaps, residual F/g tails, proposal construction overhead, uncertain
normalization, and loss of QMC smoothness under the transform.
