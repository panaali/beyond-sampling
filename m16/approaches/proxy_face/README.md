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

Done: bounded DEV diagnostic, no estimator built. Formula/settings were committed
as `614b4ed` before model outcomes. Exact integrals used reviewed corrected walker
`ae05eef` (locally `334b5e7`). T5 was omitted: the cached seed-0 model is not the
leaderboard's seed-2 DEV base for that length.

## Results

Reproduce from repository root after installing the corrected walker:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/proxy_face/check.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/proxy_face/probe.py
```

`results.json` preserves all seven settings, both families, both independent
splits and both DEV models. `run.log` records per-family cost-adjusted gains.
There was no between-split selection, tuning or fitted correction. No leaderboard
test model was evaluated. Diagnostic averages use random faces.

These are **optimistic importance-proposal efficiency gains over uniform
exact-face Monte Carlo**, using equal numbers of R23 and R12 faces. They include
proxy computation and proposal-weighted exact-walk cost, but assume a known
normalizer and a free exact transport that has not been constructed. They are not
measured estimator speedups or QMC convergence gains.

| Frozen proxy | Mild design | Mild held-out | Harsh design | Harsh held-out |
|---|---:|---:|---:|---:|
| Pair linear mass | 0.457 | 0.423 | 0.146 | 0.131 |
| Pair, floor .01 | 1.150 | 1.012 | 0.356 | 0.340 |
| Pair, floor .1 | 1.341 | 1.229 | 1.038 | 1.034 |
| All-logit envelope | 1.145 | 1.084 | 2.473 | 2.403 |
| Envelope, floor .01 | 1.161 | 1.101 | 3.042 | 2.995 |
| Envelope, floor .1 | 1.239 | 1.177 | 3.640 | 3.558 |
| Window mass | 0.724 | 0.691 | 1.971 | 1.971 |

Mild = sigma .0025; harsh = sigma .02. Values below one lose to the existing
uniform-face baseline. Even the harsh-copy envelope improvement does not pay for
exact line integration compared with direct Bernoulli sampling: its held-out
efficiency ratio is only **0.343** at equal face counts, before transport costs.

A **posthoc report-only oracle** also optimizes allocation between families.
For family variance V_k and mean cost c_k the optimal variance-times-cost
coefficient is `(sum_k Z_k sqrt(V_k*c_k))^2`. Both baseline and proposal receive
this oracle; no numerical probe or proxy selection changed. The JSON records
both coefficients independently, not just their ratio.

| Held-out frozen proxy | Mild vs face MC | Mild vs Bernoulli | Harsh vs face MC | Harsh vs Bernoulli |
|---|---:|---:|---:|---:|
| Pair, floor .1 | 1.042 | 0.532 | 1.027 | 0.105 |
| Envelope, floor .1 | 1.016 | 0.519 | 3.392 | 0.347 |
| Window mass | 0.607 | 0.310 | 1.926 | 0.197 |

All seven settings lose to direct Bernoulli sampling even in this optimistic IID
variance/cost comparison. This is not an impossibility result for QMC under a
specially constructed smooth transform; that additional advantage is unmeasured.
These are finite-sample plug-in variance estimates, with no claimed confidence
interval or whole-model accuracy result.

**Why tail reduction is insufficient.** On held-out mild R23, the envelope with
floor .1 reduces the top-1% second-moment share from .529 under the uniform law
to .119 under q. But expected exact-walk-plus-proxy cost grows from 10.69 to
20.75 FPE per face, because q favors expensive lines. Mild R12 retains .284 of
its proposal second moment in the top 1%, versus .507 under the base law. Harsh
R23/R12 proposal shares are .109/.195. Both-family tail control and cost matter.

**Independent audit and correction.** The reviewer checked importance variance,
proposal-weighted costs and fractional proposal-law tail calculations. They found
that the initial support check `F>1e-12` could falsely permit an unsupported tiny
positive mass. It now uses strict `F>0`; `F=[1e-13,1], g=[0,1]` is a regression
check. A complete corrected rerun left production metrics unchanged. The audit
also motivated the separate posthoc oracle-allocation report. Raw proxies had
no observed support failure on these faces; this does not establish global support.

## Compute accounting

All diagnostic exact walks, proxy forward/tangent operations, envelope operations,
CDFs, and input generation are recorded. The marginal proposal scenario includes
the expected exact-walk cost **under q** plus per-point proxy work. Construction
of a normalized transport is not implemented and is not assigned a fictional zero
cost. Diagnostic setup and evaluation work are reported separately.

One full run uses 131,072 face points: 1,354,856.61 FPE of exact walks plus
1,821,507,584 proxy FLOPs (303,584.60 FPE at H16,T10), totaling 1,658,441.21 FPE.
The final run took 4.64 CPU seconds inside `main`, excluding imports. Two full
runs were performed, before and after the support-check correction: approximately
3.317 million FPE total, plus small algebra/proxy checks. No rejection candidates
were generated. The 1,310,720 uniform variates and face transforms are timed;
they construct the single input per queried face and are not extra rejection work.

The proxy tally charges the implemented full value+tangent recurrence, two dense
readouts, finite-window inequalities and Gaussian CDFs. CDF cost uses the walker's
explicit 50-FLOP convention. All diagnostics share one value/tangent pass; each
prospective model-dependent proxy's marginal cost independently includes that
pass. Exact walks use the corrected returned cost, including cached prefixes,
roots, envelopes, finite checks and bookkeeping. Normalizer estimation, density
construction, inverse mapping and their numerical error remain unimplemented and
uncharged, which makes the reported ceiling optimistic.

## Claims

- [measured] The frozen all-logit envelope with .1 window floor improves the
  harsh-DEV importance variance/cost ceiling over uniform-face MC, replicated
  without selection on held-out faces; the mild-DEV benefit is negligible after
  oracle family allocation.
- [measured] Every tested setting remains below ordinary Bernoulli efficiency in
  the optimistic IID variance/cost comparison.
- [measured] Proposal-weighted costs materially reduce apparent gains: high-error
  face proposals preferentially select expensive walks.
- [inferred] These proxies do not support an immediate general estimator claim.
- [speculative] A transform could deliver additional QMC smoothness gains not
  predicted by the IID comparison; no such transform was built here.

## Alternatives not ruled out

Conditional cumulant proxies (T35), other adaptive transforms, and more
independently trained DEV models remain untested. Finite reference samples can
underestimate disastrous rare underprediction tails.

## Failure modes

Proxy support gaps, residual F/g tails, proposal construction overhead, uncertain
normalization, and loss of QMC smoothness under the transform.
