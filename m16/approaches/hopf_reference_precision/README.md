# Fixed H4T4 reference precision

## Question
Does the unchanged original Hopf h64 estimator beat sampling on the fixed25
H4T4 clean/noisy models after direct source-cost accounting and more precise
independent reference evaluation?

## Method
Claim c78c0999 precedes implementation. Pin T183's25 model identities, original
Hopf estimates and per-row source budgets. Fresh independent named CRC32/PCG64
streams supply exactly100M float64 Gaussian draws per model in100 fixed1M
blocks. The original zoo44 mc_acc body changes only its return from mean to
integer hits. First-index predicted argmax and original argsort rank labels
are unchanged. All2500 before/after states and integer hits are preserved.
The initial pilot is included exactly once; no accuracy-dependent stopping.

Fresh95% simultaneous Clopper–Pearson intervals use one25-truth Bonferroni
family. Existing15 groups (five seeds times clean/noise/pooled), two sampler
costs and two allocations give60 comparisons. Follow-up426df9ce reports ALL25
individual copies under both own-cost baselines (50 comparisons); the same
simultaneous truth family covers these derived intervals without extra draws.
No noise subtraction or h128-as-truth is used.

## Status
Bounded phase complete and independently reviewed. Pilot/source91cbea3c;
all2500 records, independent binomial-tail inversion and60 exact comparisons
8644295e; all50 exact individual comparisons c39c57fa. DRI classification is
requested; this folder does not mark a general challenge claim confirmed.

## Results
All60 group upper ratios and all50 individual upper ratios are below1 under
the declared source budgets and nominal simultaneous truth intervals.

| Seed | Optimized matched pooled point ratio | Simultaneous upper |
|---|---:|---:|
| 0 | .0490593 | .3076300 |
| 1 | .0271648 | .2745163 |
| 2 | .0527538 | .3409843 |
| 3 | .1161159 | .4753996 |
| 4 | .0085623 | .2128810 |

Worst group point/upper: .14283777/.53684092. Worst individual is seed3/copy3
under the stricter zero-state-optimized comparator: point .29471285, upper
.85195809, with1,513,001 matched draws at candidate source allowance243,593,320.
The individual estimate .4572441167332029 versus fresh reference .45702426 is
retained; pooled statistics do not conceal that case.

The original10M/75-family evidence remains immutable in T183: its upper
intervals were inconclusive. The fresh evaluation improves reference precision,
not estimator parameters, model/noise selection or candidate cost. Clean0's
fresh reference is .47703504; its prior independent positive reference remains
.4770131259941194 with a separate integration-only bound.

## Compute accounting
Exactly2,500,000,000 reference forward evaluations plus318 separately counted
deterministic control evaluations. Recorded reference CPU485.652734666;
saved contraction .07999426CPU. The fixed first1M pilot projected539.894275CPU,
below the predeclared600CPU expansion gate; the projection was not a runtime
guarantee. All processes one thread,2GB,<=110CPU, serialized primary lock.
Source/statistical review processes <=20CPU. No failed or discarded reference
blocks. Startup, file I/O and review CPU are outside recorded forward timers.

References are evaluation work and are never supplied to the deductive
estimator. T183's original per-row costs stay frozen: explicit numerical
expressions plus separately qualified sort/search source allowances. Sampler
costs include hit aggregation and batch/index terms; labels/RNG are free under
the stated comparison. This is not a complete native/binary/runtime budget.

## Claims
[measured] All110 fixed group/individual derived comparisons have nominal
simultaneous upper ratio below1 under the stated source-budget convention.
[inferred] The earlier reference-precision obstacle is resolved for this fixed
panel under those assumptions. No M16T10 or wider-model conclusion follows.

## Alternatives not ruled out
Native binary/control/memory work remains incompletely priced. The original
Hopf source retains the independently demonstrated generic root-mask defect;
this is performance evidence on these25 models, not universal correctness.
The same existing weight/noise panel was re-evaluated, not a new weight-noise
holdout. Numerical confidence inversion is not outward rounded, and floating
forward truth is not an exact-real network certificate. Other H/T regimes and
scaled algorithms remain open. Shared mask adoption stays DRI-owned T182.

## Failure modes
RNG state discontinuity, duplicate blocks, shared model seeds, changed dtype
or forward semantics, adaptive sample counts, hidden individual outliers and
confusing source allowances with full runtime. Saved audits explicitly checked
coverage, state chains, hashes, integer counts and all individual comparisons.

## Reproduction
Use /workspace/clean-neurons/.venv/bin/python with thread counts1, external
2GB/110CPU cap and shared compute lock. Saved outputs refuse overwrite; use
an isolated owned reproduction checkout and preserve previous records.

```
python protocol.py
python reference.py --controls
python reference.py
python continue_reference.py --seed S --copy C
python summarize.py
python compare.py
python individual.py
```

The continuation skips only the already saved clean0 block0. The last three
commands read saved data only and need no estimator/reference replay.
