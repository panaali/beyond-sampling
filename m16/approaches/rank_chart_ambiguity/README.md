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

## Frozen finite-panel result
`probe.py` completes all1944 trained probes and48 fixture checks in .667074573 CPU seconds;1941 trained inputs have certified open neighborhoods, three are rejected at genuine gate/output boundaries. Both analytic constant-output controls pass all24 charts. Results preserve outcome counts and45 first accepted witnesses.

**21/24 charts are proved mixed**, carrying exact Gaussian input probability173/192≈.9010416667. Therefore a method restricted to whole-chart uniform correctness certificates can cover at most19/192≈.0989583333 of this checkpoint's input distribution. This is a certificate-granularity obstruction, not an accuracy lower bound or failure of approximation/integration on these charts. The positive neighborhoods may carry tiny mass.

Three charts have only wrong witnesses among the81 frozen probes: (top1,second3,x0>0), (top2,second3,x0>0), and(top2,second3,x0<0). Their masses1/64,1/64,13/192 sum19/192. All three remain **unknown**, not uniformly wrong. No threshold or probe was adapted to the result.

Reproduce with:

```sh
ulimit -v 2500000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/rank_chart_ambiguity/probe.py
```

[measured] The finite panel finds exact open outcome ambiguity on21 charts. [inferred] No sound relaxation can prove whole-chart uniform correctness on any of these21 charts, because the exact network itself has both outcomes. This does not preclude cheap margin bounds on smaller regions, T108 gate-freezing approximation, or output-integrated estimators. Independent review pending.

## Frozen reference-artifact discriminator for the three one-sided charts
Before any additional probe or learned rule, scan the already existing exact T68 correct-event simplices (5058) to identify chart ownership at positive ray sums; verify every simplex ray satisfies that chart's rank/sign inequalities. For each of the three unknown charts, try at most the first three corresponding simplex interiors as new exact direct network/neighborhood witnesses, using unchanged witness logic. This costs at most nine extra deterministic checks, reported separately from the completed2000-check-capped phase. Cap5CPU and no new neural geometry, mass integration, LP solve or adaptive search. Existing artifact construction10.009618928CPU is a dependency, not a free production oracle.

A new correct open witness resolves a finite-search miss. Absence of a corresponding full-dimensional correct simplex is only an artifact-conditional observation; retain unknown unless reference completeness and tie semantics are independently accepted. This phase cannot create a weight-only whole-chart certificate. Preserve phase1 results unchanged.
