# T122: mass of exact rank-conditioned bulk certificates

## Question
How much probability mass do T120's 28 uniformly classified nonempty rank-conditioned regions cover on H4T4 seed0? Counts alone do not establish coverage.

## Method
Frozen before implementation, 2026-10-17; TASKS-only claim 7b89bb9. Reuse the same ten gate histories selected by gate mass in T120 (total mass .4482031037438533), their final exact feasibility classifications, and only their 130 existing exact T68 gate simplices. Initialize each simplex from exact cofactor facets and intersect it with all 12 ordered top/second rank cones (three coordinate-difference planes). Reuse the read-only exact integer clipper and positive GL20 mass primitive. No new neural gate enumeration or output splitting.

Controls: analytic signed orthants; exact parent-normalized-determinant conservation over all rank children; all children retain the parent orthant and satisfy rank planes; zero full-dimensional children for dual-certified empty pairs; independent existing correct-event simplex attribution excludes uniformly wrong cells and matches uniformly correct mass. Compare GL16 with unchanged GL20. Freeze source/artifact hashes. Cap 30 CPU seconds and 5000 clipping calls, with partial checkpoints after complete parents. Historical geometry construction (10.009618928 CPU seconds) is separately charged. A cap or implementation failure leaves unprocessed mass unresolved.

Two alternatives to useful coverage: uniformly classified cells may carry negligible mass (test by integration); geometry/mass errors may explain apparent coverage (test exact volume conservation, independent correct artifact and GL refinement). Even high coverage may be too expensive for an estimator (report complete incremental cost plus historical dependencies; no matched-cost claim).

## Status
Bounded selected-panel phase complete; independent review pending.

## Results
Run from the research worktree with:

```sh
ulimit -v 2500000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-compute.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/bulk_rank_mass/measure.py
```

`results.json` preserves all 120 pair masses and 130 exact parent partition checks. The selected 130 parents yield 792 trained child simplices. The analytic signed-orthant controls also pass. All 66 dual-certified empty rank cells contain no full-dimensional children.

| T120 classification | Input mass GL20 | Conditional quadrature bound | Child simplices |
|---|---:|---:|---:|
| Uniformly wrong | .12333757610644563 | 6.6655e-9 | 285 |
| Uniformly correct | .04925087793568042 | 2.4637e-9 | 135 |
| Mixed | .2756146497017266 | 1.4481e-8 | 372 |
| Empty | 0 | 0 | 0 |

The selected sum .44820310374385275 agrees with the prior gate mass to 5.6e-16. Uniform certificates cover .17258845404212608; unresolved input mass is .8274115459578739 (mixed selected cells plus all unselected histories). Uniform wrong cells explain about 23.58% of the known checkpoint error. This is useful partial attribution, not a full estimate: its unresolved-accuracy width remains .8274.

All 242 independent existing correct-event simplices were checked to have fixed top/second ownership on every ray. None falls in a wrong/empty cell. Their uniformly correct mass .04925087793568045 agrees with the rank-partition value to 2.8e-17. Total GL16-to20 change is 4.40e-14. These checks share the reference artifact and floating mass primitive; independent source review remains necessary.

## Compute accounting
Charge source/data setup, exact clipping/triangulation and validation, quadrature calls at both orders, and correct-artifact controls. No random input samples. Exact witness computations and all setup count; source geometry cost remains a separate dependency. One thread, 2 GB cap, primary shared compute lock.

## Claims
[inferred] Exact feasibility certifies correctness or error on open rank-conditioned regions, but their mass must be separately computed.

## Alternatives not ruled out
Unselected histories, mixed rank cells, other models/seeds/noise, more efficient rank representations, and cheaper partial-mass estimators remain open. This is not a full-rung or challenge solution and does not assume errors concentrate near boundaries.

## Failure modes
Floating quadrature and input conversion retain their existing limitations; conditional bounds are not an unconditional rounding certificate. Shared source provenance reduces implementation independence. Strict rank/gate boundaries have Gaussian measure zero; identically zero gates were handled explicitly in T120. Exact topology does not make numerical mass exact.


Completed incremental run: **1.060827033 CPU seconds**, including imports, all data loading/hashes, controls, exact clipping/triangulation, GL16/20, correct-artifact checks and checkpoints. Recorded 3273 clips, 12546 integer determinants, 485 certified cones and 830 total simplices including controls; peak RSS and integer bit maxima are in results.json. No random model inputs and no new gate/output enumeration. Historical exact geometry cost is 10.009618928 CPU seconds, with earlier T116/T120 setup and feasibility costs additionally required; this run alone is not total estimator cost.

[measured] Rank conditioning substantially improves the coverage of this selected set over whole-gate certification, but **82.74% of all input mass remains unresolved**. [inferred] Counts of feasible cells were insufficient; numerical mass was essential. [speculative] Selective mixed-cell output clipping or broader cheap feasibility screens might improve coverage per cost; neither is evaluated here. No broad method is retired by this bounded result.

Independent T55 review868c69b passes source/raw parent and Decimal arithmetic checks. All130 parent cofactor facets, normalized determinants and orthants were independently checked; child reclipping and mass were not independently rerun. See HANDOFF.md for next-step cost/uncertainty limits.
