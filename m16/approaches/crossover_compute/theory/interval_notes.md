# Certified scalar-affine tie-segment integration

## Question
Can one-dimensional affine bounds integrate decision regions without paying for every hidden gate cell?

## Method
Within one target-constant half of a sci_tao tie segment, parameterize x=y+d*(mid+half*s), -1<=s<=1.
Maintain h=a*s+b+epsilon, |epsilon|<=r componentwise. Linear recurrence propagates a,b through W and
r through |W|. With preactivation bounds L=b-|a|-r and U=b+|a|+r, positive/negative stable ReLUs
are exact. For L<0<U let lambda=U/(U-L), mu=-UL/[2(U-L)]. The valid ReLU enclosure is
(a,b,r) -> (lambda*a, lambda*b+mu, lambda*r+mu). This is a scalar affine enclosure with box residuals.

At the output, multiply by EACH LOGIT DIFFERENCE directly. A margin is A*s+B +/- R.
The intersection of inequalities A*s+B-R>=0 is a guaranteed-correct interval C.
The intersection A*s+B+R>=0 is a possible-correct interval P, with C subset P.
Gaussian mass inside C and outside P is settled immediately; only P minus C is unresolved.
That set has at most two intervals. Split those intervals by Gaussian probability midpoint and repeat.
This is stronger than certifying only nodes that are wholly right/wrong. With no ReLU ambiguity,
radius is zero and all decision crossings integrate analytically without further gate walking.

The inequalities are sound in real arithmetic. This prototype uses float64 plus a small conservative
margin inflation, NOT directed interval arithmetic, so numerical certificates are validated diagnostics,
not a machine-checked proof. Only outer test directions are sampled; there is no estimator-class claim.

### Alternatives and cheapest falsification (before code)
1. Absolute-weight propagation amplifies the residual radius until almost every node requires splitting.
Compare settled mass after 8/32/128 nodes with exact gate-walk costs on fixed DEV lines.
2. Tiny error islands require many splits even when most intervals are correct. Record unresolved mass
and counts rather than counting only successful nodes; ensure exact mass lies in every interval.
3. A result might be a floating-point or target-switch mistake. Split tie segments at zero first and
validate every reported error bound against the independent exact sci_tao walker.

## Status
Done. Successful numerical bounds, negative accelerator. All 256 lines converged
within the declared tolerance, but total recurrence work was 3.45–4.10 times the
existing exact full-recomputation gate walk. No challenge submission.

## Results

Reproduce from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python m16/approaches/crossover_compute/theory/interval_probe.py
```

The public checkpoint may require network access. `interval_results.json` retains
every line's exact reference, lower/upper errors, node counts, and snapshots at
8/32/128/256 allowed nodes. Directions are fixed Sobol diagnostic points (seed344),
64 per model/family. Models are the shared DEV sigma.0025 and sigma.02 copy0;
no scored TEST model is used. Reference is unmodified `sci_tao.core.walk_mass`.

| DEV sigma | tie family | mean exact gate cells | mean bound nodes | bound/reference recurrence cost |
|---|---|---:|---:|---:|
| .0025 | 23 | 7.0625 | 19.28125 | 4.0951 |
| .0025 | 12 | 8.609375 | 22.75 | 3.9637 |
| .02 | 23 | 7.359375 | 17.3125 | 3.5287 |
| .02 | 12 | 8.171875 | 18.78125 | 3.4474 |

All lines reach unresolved mass <=1e-6 by 128 node attempts, usually much tighter:
mean final unresolved masses are 1.4e-14–3.4e-14. The maximum discrepancy between
any exact reference and any reported enclosure (all four snapshots) is 1.11e-16,
consistent with floating-point summation. This does not prove sound floating-point
rounding or identify every individual gate event.

At the eight-node cap, 34–53% of lines have resolved, but mean unresolved mass is
still .0317–.0341. Recurrence work already costs 1.22–1.35 times a COMPLETE exact
reference walk. At32 nodes the mean uncertainty is .00101–.00224 and cost2.66–3.20
reference walks. Simply accepting coarse certificates therefore does not rescue
this implementation at comparable accuracy.

The important distinction from global ten-dimensional branch-and-bound is that
these one-dimensional certificates do converge rapidly. The failure is relative
cost: the exact tie segments already have only about7–9 hidden-gate cells, and
safe affine relaxations require roughly17–23 nodes. Sparse lower-dimensional
geometry alone is not a sufficient accelerator.

Total main CPU time1.534 seconds, wall14.379 seconds including public checkpoint
loading. No external source prose or unpublished ARC method was accessed.

## Compute accounting
Each bound propagates a,b,r through all recurrent steps and output differences: approximately THREE
complete forward passes, not two. Interval operations, Gaussian CDF/inverse calls and input generation
are additional and excluded from recurrence ratios, but included in CPU/wall times. Reference sci_tao implementation actually performs two complete
passes/cell, irrespective of its legacy suffix-cost estimate. No fitted advice or test-set tuning.
The run evaluates 5000 bound nodes (15,000 recurrence/readout FPE) plus 3994 exact
reference FPE, for 18,994 total before scalar overhead. The256 outer points and
model-free face-table construction are included in runtime, not the recurrence
ratios. Checkpoints reuse the same computation; their costs are not summed.

## Claims
- [inferred] Scalar-affine envelopes give exact integrable inner/outer decision intervals in real arithmetic.
- [measured] Scalar-affine certificates resolve all256 DEV lines at <=1e-6 mass tolerance,
with no reference-enclosure violations beyond1.11e-16.
- [measured] This implementation is3.45–4.10x MORE expensive in recurrence work than
full-recomputation exact walking. It does not supply the proposed accelerator.
- [inferred] A successful certified skip method would need tighter enclosures,
cheaper bound updates, or a line distribution with substantially more irrelevant gate cells.

## Alternatives not ruled out
Correlation-preserving zonotopes or backward CROWN might give tighter bounds at greater cost.

## Failure modes
Amplification by |Wh|, rare small error intervals, and floating-point certificate slack.

## Follow-up suggestions for parent/DRI

Do not repeat the full sweep or put this on the estimator leaderboard. A different
bound family is not ruled out: full affine-error symbols preserve hidden-state
correlation that this scalar-affine-plus-box method drops, while backward CROWN
can tighten output-specific bounds. Their greater cost must be paid explicitly.
A cheap incremental certificate preserving the prefix before the earliest uncertain
gate could also change the tradeoff; that needs an implemented comparison against
the already-incremental exact walker, not hypothetical prefix discounts. These
are possible new tasks, not completed or claimed work here.

## Reusable bound API for approximation audits

- `certificate(model, y, d, lo, hi, target)` accepts a batch of target-constant
  affine input lines, and returns `(C, P, ambiguous_counts)`. `C` and `P` each have
  shape `(n, 2)` in the ORIGINAL line coordinate. Correct points include C and
  are contained in P; an interval with upper<=lower is empty. Each target must
  stay fixed throughout its supplied interval. The half-tie construction ensures
  this. Arbitrary lines that cross target ranks must first be subdivided.
- `integrate(model, y, a, b, tau, kind, checkpoints, tol)` accepts existing Tao
  face data and returns snapshots of per-face `error_lower`, `error_upper`,
  `unresolved`, and `counts`. Kind is `23` or `12`; the zero split is automatic.
  With a fixed face and any surrogate mass m, its absolute error is at most
  `max(abs(m-error_lower), abs(m-error_upper))`, subject to numerical soundness.
- Nonnegative fixed outer weights preserve bounds by weighted summation. Such a
  sum certifies only that fixed outer rule, not the true outer integral. Auditing
  finitely many sampled faces does not certify an unseen TT/surrogate residual.
- Converting this into formal certificates requires outward-rounded arithmetic
  (including Gaussian CDFs) and documented exact-tie conventions. The present
  1e-12 margin slack and reference checks are diagnostic safeguards only.
