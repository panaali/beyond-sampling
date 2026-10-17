# T125: whole rank-chart bulk ambiguity

## Question
Can an exact whole-chart uniform correctness certificate cover the H4T4 seed0 distribution before enumerating neural gates, or do fixed ordered-top-two/sign charts already contain both correct and wrong open sets?

## Method
TASKS-only claim33e2148 pushed before implementation. Freeze all24 charts defined by top index a, second index b, a!=b, and sign of x0. The four strict coordinate-difference/sign normals form an invertible integer matrix N. Probe x=N^-1 c for every c in {1,4,16}^4, lexicographic order, 81 per chart,1944 trained direct network evaluations. This is a finite deterministic diagnostic, not a deductive accuracy estimator. No random inputs, truth fitting, gate enumeration or LP solves.

Every accepted witness must have exact chart/gate/readout ownership. Derive gate linear maps using the checkpoint's dyadic integer representation; remove only identically zero OFF forms. Reject nonidentical zero margins as boundary witnesses. Construct an exact positive infinity-norm radius from minimum margin/(2 L1 norm) over the relevant chart, gate and output inequalities. Correct certificates obey first-index ties; wrong certificates use a strictly beating competitor or an identically tied earlier label. Both witness types in one chart prove that no uniform correctness certificate can cover that entire chart. Missing witness types remain unknown, never uniform by a finite search.

Analytic chart masses follow exchangeability and the binomial number of positive iid Gaussian inputs: conditional x0>0 probabilities15/16 if a=0,11/16 if b=0,3/16 otherwise; each ordered pair has mass1/12. Check total mass1. Freeze zero-readout first-index and globally-positive class3 fixtures, one exact rational input per chart each (48 additional direct checks; total1992<=2000). Check fixture predictions without requiring stable gates at their single representative. Cap20CPU under review lock, one thread,2GB. Store all probe outcome counts and both first open witnesses per chart; no early stopping within chart.

Alternative1: observed errors live only on boundaries; strict neighborhood radius discriminates. Alternative2: a finite search misses one outcome; only positive witness pairs establish mixedness, every one-sided chart remains unresolved. Alternative3: map arithmetic or ties cause false witnesses; independently recompute recurrence/rank/box slack from rational weights and analytic fixtures.

## Status
Protocol frozen before implementation.

## Results
Pending.

## Compute accounting
Charge source loading, exact chart inversion, all1992 direct recurrence evaluations, trained gate-map checks and witness radii, output serialization. No sampling comparison or cheap inference cost inferred from wall time. This is distinct from T108 bounds for freezing intermediate gates and T123 mass integration.

## Claims
[inferred] Two exact open witnesses with different correctness prove a whole chart is mixed. [speculative] This may eliminate futile whole-chart uniform certification while preserving adaptive subdivision and integration.

## Alternatives not ruled out
Relaxations on smaller adaptive subregions, nonuniform error approximations, partial correct/error mass, gate-freezing perturbation bounds, other seeds/noise/rungs. A mixed chart can still be approximated well; witness neighborhoods may be tiny.

## Failure modes
Finite probes do not bound missing outcome mass. Small exact radii are formal support witnesses, not practically important error mass. No Gaussian probability lower bound is claimed from a witness unless separately integrated. Exact checkpoint arithmetic differs from finite-precision inference at boundaries.
