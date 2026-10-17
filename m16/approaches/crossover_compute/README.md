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

Running. The initial checkout has no v3 report or raw runs; legacy v2 results
cannot substitute for v3 class review, advice declarations, or compute rules.
Integration into `m16/leaderboard.py` is DRI-only and will be proposed here.

## Results

Pending bounded checks and candidate evaluation.

## Compute accounting

Reporting uses existing group statistics: zero model forward passes, zero
model-dependent FLOPs for estimation, zero LP solves, and no input generation.
Report processing and tests are ordinary host CPU work, recorded separately.
Candidate accounting is separate and includes model-dependent derivatives and
linear algebra, not just forward passes. All checks run single-threaded within
the light-task limit; no new ground-truth sweep is requested.

## Claims

- [inferred] A finite-grid maximum is useful partial progress, not a proof of a
  continuous crossover or a guarantee for unseen seeds.
- [measured] No v3 table/raw artifact was present at initial checkout a2ce777.
- [speculative] A cheap mechanistic approximation may win at low budgets.

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
