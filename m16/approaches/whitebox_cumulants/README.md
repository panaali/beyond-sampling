# T474: reproduce public cumulant estimators within local memory limits

## Question

Can we run a strong public Phase 2 activation-mean estimator under 2 GB per
process, with the official computation meter and a sufficiently precise answer
key? This is baseline reproduction, not a new solution claim.

## Method

Freeze the public V25/V29, e30_a and logxio submitted files, test their interfaces
on small networks, and profile progressively larger V25 runs in isolated worker
processes. V25 precedes the large persistent Strassen pools and is the first
low-memory candidate. The evaluator/supervisor is not submission code. Submitted
files are unmodified, verified by SHA256, and all numerical prediction work uses
the official flopscope meter. Record setup FLOPs/time separately.

Two alternatives to distinguish before building on the memory hypothesis:

1. V25's persistent arrays already exceed the limit. Small-width RSS plus a
   source-based allocation census is the cheapest test; do not launch full size
   unless a conservative projection fits.
2. V25 fits but ceases to be useful because the full-size branch, old-source
   compression, residual time or updated meter changes its behavior. Width 512
   exercises old-source compression; only 1024×16 exercises calibrated K4 and
   mean corrections. Meter and score the unchanged full-size path if safe.

## Preregistration (2026-10-21, before model runs)

- Sources: 504aldo `a038d17924cd2df82dbf3a0d42da38fee48aa5a5` V25/V29;
  e30_a `3d038e9f48a97671878ee7bbef02e193f128b61c`;
  logxio `90b6d5fb3e49d7b49c7e0b070afcefe45e3785bf`.
- Meter flopscope 0.12.3, whestbench 0.16.2, Python 3.11, NumPy 2.4.6
  (the T473 lock). Original numerical tables retained, despite upstream caveats.
- Interface controls: all four originals at widths/depths 32×4 and 64×16,
  weight seed 2026. V25/V29 output comparison is diagnostic: they are related,
  not an assumed identical pair, and exact costs can differ.
- Memory ladder: V25 at 256×16, then 512×16, seed 2026. Full size only if
  source census plus observed RSS scaling predict <1.7 GB. Otherwise stop and
  specify a separately checked memory-only rewrite. No full-size V29/e30/logxio
  launch based solely on small-width results.
- Runtime caps: one estimator worker at a time, <=2 BLAS threads, <=120 s per
  predict, <=5 s setup, <=0.4 s residual for a valid score, aggregate <=600
  worker CPU seconds this phase. Supervisor samples the worker PID's RSS and
  terminates only that worker above 1.65 GB; this is a guard with headroom,
  not proof of a continuous hard memory ceiling. Limit each run to 150 s wall.
- Full-width DEV: first two rows of Phase 2 public `mini` first shard;
  evaluation: first two rows of public `full` first shard. Pin dataset commit
  `aa99830fdc09fad15407b10e8e3459d3e18bba0a`. These are held out from **our**
  development only; public authors may have used them. Do not call them sealed.
- Obtain only seed/reference columns with HTTP ranges, <=250 MB transfer cap.
  Reconstruct weights via the documented SeedSequence child-0 protocol, verify
  against the official generator and, if feasible, stored weights. No scoring
  until weight provenance is verified. Published references use 1e9 samples;
  retain their finite sampling uncertainty and source exposure.
- Compare original V25 to supplied covariance baseline on precisely the same
  rows. Official score is MSE×max(0.1,FLOPs/2^41). Report strict <=1 and lenient
  <=10 score-ratio bars versus covariance; these are not AlgZoo F. No rank claim.
- Stop on nonfinite output, resource/contract failure, or provenance mismatch;
  document bounded negative/inconclusive outcomes. Do not substitute noisy T473
  truth or tune on reference answers. No organizer contact or submission.

## Status

In progress. Source review and frozen protocol precede execution.

### Memory rewrite extension, frozen before its runs

Original V25's half-width memory measurement does not support a safe full-width
launch (see INDEX/result artifacts). Test a separate, licensed derivative with
128-row blocks in the source-slice and nonlinear-term contractions. Partition
only a free output index; retain all summed indices, coefficients, source ages,
ranks and default flags. V25's old mean-correction flag is off by default and
stays off. The original small deep-network failures remain in the record; they
stop their individual rows, while larger finite cases still support memory and
parity diagnostics because this is not an accuracy ladder claim.

Before scoring, require max absolute prediction difference <=3e-5 and final-layer
mean squared prediction difference <=1e-11 against finite originals at 32×4,
256×16 and 512×16, seed 2026; add independent seed 2027 at 512×16. Charge all new
metered work; changed costs or floating-point regrouping are reported, not assumed
away. Run full width only after the blocked memory projection clears the original
gate. A mismatch halts scaling. This is a memory representation change, not a
new cumulant approximation, and does not establish the strongest public score.

## Results

Pending; numerical results will be registered in [INDEX](../../../INDEX.md).

## Compute accounting

Official BudgetContext encloses each unchanged `predict`. Setup is measured
separately, including any warmup array work. Reference loading, model generation,
supervision and scoring are evaluator overhead, not estimator computation.
Meter version differs from some reported public submissions; reproduce what the
current pinned meter charges, not their historical cost by assertion.

## Claims

- [speculative] V25 can serve as a useful full-size baseline under our memory cap.
- [inferred] Small-width interface success cannot establish the Phase 2 method's
  performance because its source code selects different algorithms at that size.

## Alternatives not ruled out

V29 without recursive pools, exact source chunking, freeing dead temporaries,
and corrected cumulant tables. Accuracy-changing source compression is a new
variant and requires its own protocol; not part of an exact memory rewrite.

## Failure modes

Width-specific branches invalidate naive memory/performance extrapolation.
Static allocations miss temporary contraction buffers. Public references and
public fitted corrections may overlap in training data. Local interface checks
are not certification of remote submission eligibility. More precise evaluation
does not prove an approximation is unbiased or a general solution.
