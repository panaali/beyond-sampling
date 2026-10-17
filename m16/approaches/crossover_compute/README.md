# T19: crossover compute per method

## Question

At which measured compute budgets does a method match plain sampling, and does
one reviewed mechanistic method do so in every required model group?

## Method

`report.py` consumes the v3 leaderboard's already-scored JSON table. It performs
no model runs and never changes DRI-owned leaderboard code. It reports the
largest **tested requested budget** with MSE ratio <= 1, together with actual
costs. This is a grid result, not an interpolated or extrapolated crossing.
Point-estimate and upper-95%-CI criteria are reported separately. All-group
results require a common tested budget, every required group, no advice-tainted
row, and finite valid measurements. Only DRI-reviewed `mechanistic` rows count
in the mechanistic headline; other classes remain visible for comparison.

Before analysis, alternatives and cheap distinguishing checks:

1. An apparent crossover is missing model coverage or test-model advice rather
   than general performance. Require complete group coverage at the same budget
   and exclude tuned rows; check deliberately incomplete fixtures.
2. The last passing point is a sampling fluctuation, or performance is
   nonmonotone. Report both point and upper-CI criteria, retain failures below
   passing points, and never assume every lower budget passes.
3. Requested budget misstates work because the estimator overspends or has
   fixed cost. Show actual per-group cost beside the requested budget; make no
   inference about performance at unmeasured actual costs.

The candidate sub-study is documented separately in `candidate_notes.md` when
available. Its classification remains unreviewed until the DRI audits it.

## Status

Done as a bounded reporter/candidate study; the broader T19 task remains open.
The candidate has no observed winning tested point. Integration into the shared v3
report and reviewed classification are DRI-only. Legacy v2 results cannot substitute
for v3 class review, advice declarations, or compute rules. Later upstream snapshots
added v3 sampling raw runs, but no reviewed mechanistic result establishes T19.

## Results

The report passes 11 checks covering incomplete groups, disjoint budget grids,
nonmonotone curves, actual-cost preservation, tuned exclusions, unreviewed classes,
missing uncertainty, malformed data, negative truth-corrected MSE and boundary cases.
An independent agent reviewed the code and independently recomputed every candidate
ratio, bias and bootstrap interval from raw results. This verifies arithmetic, not
DRI classification or scientific confirmation.

The frozen conditional-Gaussian candidate is negative on all 18 tested group/budget
point estimates (30 models, 3 levels). Ratios span 3.120–4706.926; no tested point
matches sampling. One provisional interval crosses 1, so inferiority is not
statistically established in every cell. Actual costs are 587.544 / 1174.944 /
2349.744 FPE at requested budgets 1000 / 2000 / 4000; 100 / 250 / 500 are unsupported.
No continuous crossover, zero-compute threshold, or family-wide impossibility is
inferred. Details and per-group numbers are in `candidate_notes.md`.

Reproduce report checks and the saved candidate summary (no model runs):

```bash
python -m unittest discover -s m16/approaches/crossover_compute -p test_report.py -v
python m16/approaches/crossover_compute/report.py --table m16/approaches/crossover_compute/candidate_table.json --out m16/approaches/crossover_compute/candidate_crossover.json
```

Candidate reproduction, including checkpoint loading, uses one thread:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python m16/approaches/crossover_compute/candidate_experiment.py
```

### DRI integration proposal

After constructing the existing v3 `rows`, the DRI can import `summarize` and
`render` from `m16.approaches.crossover_compute.report`, then print
`render(summarize(rows))`. For a nonstandard (H,T) model set, explicitly pass its
required group names (the standard M16,T10 set includes `orig`). This helper is
pure postprocessing: no source-model evaluations or shared-file mutations. To
use an existing v3 table without changing the harness:

```bash
python m16/approaches/crossover_compute/report.py --table results/leaderboard_v3_H16_T10.json
```

Before interpreting any headline, review candidate class/cost and resolve T20.
Independent review found the shared table baseline is
`mean(p)*(1-mean(p))/mean(cost)`, whereas averaging model-level sampling MSE requires
`mean(p_i*(1-p_i)/cost_i)`. For equal costs the former adds between-model variance
to the denominator. The candidate uses the latter; its table is a provisional
diagnostic, not a DRI v3 report. No shared code was changed.

## Compute accounting

Reporting uses existing group statistics: zero model forward passes, zero
model-dependent FLOPs for estimation, zero LP solves, and no input generation.
Report processing and tests are ordinary host CPU work, recorded separately.
Candidate accounting is separate and includes model-dependent derivatives and
linear algebra, not just forward passes. The frozen candidate evaluation took
1.187 process CPU seconds, excluding interpreter startup; model-free precomputation
is declared as a conservative 10M-FLOP allowance. Equal-cost CPU ratios were
2.90–3.81 times sampling. Reporter tests take under 0.1 seconds; independent
arithmetic checks add small CPU overhead. All runs used one thread within the
light-task limit; no new ground-truth sweep was performed.

## Claims

- [inferred] A finite-grid maximum is useful partial progress, not a proof of a
  continuous crossover or a guarantee for unseen seeds.
- [measured] No v3 table/raw artifact was present at initial checkout a2ce777.
- [measured] This frozen Gaussian/Jacobian candidate has no observed winning point
  on its tested ladder; the broader low-compute question remains open.
- [inferred] The current shared group baseline can flatter heterogeneous groups;
  T20 records the required DRI review.

## Alternatives not ruled out

Bootstrap intervals can be unreliable for few independent models; intervals
provided by the DRI table are used as-is, not strengthened. Per-group confidence
intervals do not supply simultaneous 95% coverage across many budgets/methods.
The group mean does not require every individual model to win. Unmeasured
budgets, models, or noise levels may reverse a conclusion.

## Failure modes

Incomplete groups never establish an all-group result. A nonmonotone curve is
not reduced to a single threshold. An unreviewed class cannot satisfy the
mechanistic headline. Non-finite/malformed data fail validation. Repeated
method/group/budget keys fail validation rather than silently picking a run.
