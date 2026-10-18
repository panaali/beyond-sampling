# width16_probe — T194: frozen recipe108 at the target width H = 16 ((16,3) and (16,4))

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · source of the recipe: [estT_fix](../estT_fix/README.md)

**All claims below are unverified** (one agent, one code base; nobody has re-checked them).

## Question
Every scaling number for recipe108 was at H <= 8; at T = 3 its cells/regions fell 4.8 -> 2.0 from H = 4 to 8 (H-U4,
RECIPE112 cells). Does that hold at the target width H = 16, and what does recipe108 cost there?

## Method
Deductive (no random model inputs in the estimator). `estT.py` is an unchanged copy of `estT_fix/estT.py` (the
T131-fixed masses, T185 5-D code unused); `zoo44.py` copied from `understand_m43`. Recipe settings frozen as
`estT_fix/rescore.py` (mode `recipe108`, eps 0.1). Models: `alg_zoo.zoo_2nd_argmax(H, T, seed)` seeds 0-4 and 4 noisy
copies of seed 0 with the estT_fix convention `leaderboard._noisy(zoo(H,T,0), 4, 0.005, 7)`.
- `run.py H T name mode` -> `res/H{H}_T{T}_<name>_<mode>.json`: recipe108 or exact enumeration (the reference), with the
  sum of leaf-cell masses and, in exact mode, the **exact region count** = number of distinct all-step gate patterns
  over the exact leaf cells (pattern taken at the mean of each leaf's rays; diagnostic, no ops). Guards: stop at 1e11
  ops or 60 CPU-min (never triggered). `drive.sh`, `drive2.sh`, `drive3.sh`: sequential, resumable drivers.
- `mc.py H T names` -> `mc.json`: 1e7 MC (rng 2026, as estT_fix) and sampled region lower bounds (2e5 and 2e6 inputs,
  rng 5, as final3/run_regions). `mc2.py` -> `mc2.json`: a second 2e7 MC (rng 7) for 4 models.
- `summary.py` -> `summary.json`: per-model table. Ratio = err^2 / (p(1-p)/N), N = recipe ops / fpe_flops(H,T)
  (estT_fix convention, kept for comparability); `ratio_T196` restates it under the T196 ruling
  (estimator 2 x ops FLOPs vs c_s(H,T) sampling FLOPs per input): x1.96 at H = 8, x2.02 at H = 16. (8,4) recipe108 numbers are read from `estT_fix/rescore/` (same code); its exact runs (for the
  region counts) were redone here and reproduce estT_fix's exact accuracies and op counts digit for digit.
- (8,3) was run here too so that the H comparison uses recipe108 cells (H-U4's 2.0 used RECIPE112/est75 cells).

## Status
done (2026-10-18). Beyond the brief: (16,4) was run on all 5 seeds + 4 noisy copies (not only seed 0), since one model
cost ~7 CPU-min.

## Results
Means over the 5 clean seeds; "all 9" = 5 seeds + 4 noisy copies. Regions = exact count (above); lb2e5 = the sampled
lower bound used in earlier tables. Reproduce: `./drive.sh H T names...`, `python mc.py H T names...`, `python summary.py`.

| (H,T) | recipe ops | exact ops | recipe cells | exact cells | regions (exact) | lb2e5 | **cells/regions** | recipe/exact cells | max \|err\| pp (all 9) | max MSE ratio (all 9) | same, T196 units |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (8,3) | 3.9e5 | 3.5e5 | 233 | 452 | 236 | 165 | **0.99** | 0.52 | 0.011 | 6e-4 | 0.0012 |
| (16,3) | 1.7e6 | 1.3e6 | 389 | 1244 | 838 | 421 | **0.46** | 0.31 | 0.0033 | 4e-5 | 9e-5 |
| (8,4) | 8.7e8 | 1.14e9 | 6276 | 9411 | 3160 | 1171 | **1.99** | 0.67 | 0.0073 | 0.27 | 0.52 |
| (16,4) | 2.9e9 | 4.5e9 | 24543 | 53912 | 28212 | 4090 | **0.87** | 0.46 | 0.0020 | 0.017 | 0.035 |

Per model (16,·) (exact acc; z vs 1e7 MC; recipe err pp; MSE ratio; recipe ops; recipe cells; exact regions):

| model | (16,3) acc | z | err pp | ratio | ops | cells | regions | (16,4) acc | z | err pp | ratio | ops | cells | regions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zoo0 | .99938996 | +1.33 (+0.16†) | −0.00003 | 9e-8 | 1.55e6 | 326 | 776 | .99787228 | −0.44 (−0.47†) | −0.00015 | 0.0013 | 2.97e9 | 25378 | 26742 |
| zoo1 | .99950815 | +1.32 | +0.00011 | 2e-6 | 1.89e6 | 480 | 882 | .99772712 | +0.06 | −0.00006 | 2e-4 | 2.68e9 | 21673 | 27084 |
| zoo2 | .99942387 | +1.77 (+0.21†) | −0.00007 | 8e-7 | 1.72e6 | 386 | 898 | .99780316 | +0.15 | −0.00027 | 0.0042 | 2.91e9 | 25225 | 31537 |
| zoo3 | .99949474 | −0.22 | −0.00049 | 4e-5 | 1.67e6 | 385 | 850 | .99830684 | +0.43 | −0.00048 | 0.016 | 2.77e9 | 23973 | 28082 |
| zoo4 | .99944820 | +1.03 | +0.00001 | 3e-8 | 1.60e6 | 369 | 786 | .99727737 | +0.17 | −0.00004 | 8e-5 | 2.97e9 | 26467 | 27616 |
| noisy0 | .97465151 | +1.27 | +0.00162 | 1e-5 | 1.66e6 | 371 | 758 | .97364031 | +0.29 | −0.00200 | 0.017 | 2.63e9 | 21174 | 27112 |
| noisy1 | .97567662 | +0.55 | −0.00062 | 2e-6 | 1.68e6 | 339 | 776 | .97646667 | −0.80 | −0.00068 | 0.0023 | 2.75e9 | 21741 | 26840 |
| noisy2 | .96934420 | +0.86 | +0.00008 | 2e-8 | 1.74e6 | 368 | 782 | .97493052 | −0.57 | +0.00033 | 5e-4 | 2.56e9 | 20314 | 25700 |
| noisy3 | .97311615 | +2.98 (−1.99†) | +0.00334 | 4e-5 | 1.65e6 | 344 | 766 | .96227484 | −0.57 | +0.00014 | 6e-5 | 2.54e9 | 21311 | 25438 |

† second, independent 2e7 MC (rng 7, `mc2.py`). The 1e7 MC uses the same inputs (rng 2026) for every model, so its z
values are correlated across models (most (8,3) and (16,3) z are positive together); the fresh MC removes the skew.
Leaf masses sum to 1 within 5.2e-10 on all 36 exact runs (`leafsum_minus_1` in `res/*exact.json`; 1e-12..5e-10 at
(16,·)). Zoo (16,3) MC accuracy ~0.9994, (16,4) ~0.998; noise sigma 0.005 drops them to 0.96-0.98 (more than at H = 8).

**Answers.**
1. **The T = 3 cells/regions fall continues to H = 16** (recipe108 cells, exact regions): 0.99 at (8,3) -> 0.46 at
   (16,3); at T = 4: 1.99 -> 0.87. From H = 8 to 16, regions grow x3.5 (T=3) and x8.9 (T=4), recipe cells x1.7 and x3.9.
2. **The growth along T is not fixed by width.** At H = 16, T = 3 -> 4: regions x34, exact cells x43, recipe cells
   x63, ops x1700 (H = 8: regions x13, cells x27, ops x2200). Cells/regions roughly doubles per step at both widths
   (0.99 -> 1.99; 0.46 -> 0.87).
3. **Cost at H = 16:** (16,3) 1.6-1.9e6 ops (< 1 CPU-s); (16,4) 2.5-3.0e9 ops (2.3-2.7 CPU-min). recipe108 beats
   sampling on all 18 (16,·) models (max ratio 0.017) and saves only 1.6x (T=3: costs 1.3x **more**) ops vs exact
   enumeration: ops are dominated by per-cell mass/certificate work (~1.2e5 ops per cell at (16,4) vs 1.4e5 at (8,4)).
4. **Earlier cells/regions tables used a sampled lower bound that undercounts badly**: lb2e5/exact = 0.70 (8,3),
   0.50 (16,3), 0.37 (8,4), 0.15 (16,4). So E45/E46's (8,4) "5.4"/"5.5" is 2.0 against exact regions, and the (8,5)
   "33" (lb 3433 from 2e5 samples) is very likely a large overstatement [inferred; (8,5) regions not counted here].
5. Projection [speculative]: if ops grow ~x1700 per step from (16,4), M_16,10 needs ~3e9 x 1700^6 ~ 7e28 ops; even at the
   region growth (x34/step) and 1.2e5 ops/cell, ~4e18. Out of reach either way.

**Side by side with other reports (DRI note, 2026-10-17; not re-checked).** (a) A Codex T113 review reportedly ran
RECIPE112 at (16,3) and got cells/regions ~0.47 and recipe/exact ops ~2.06; its artifacts are on no git ref. Here
(recipe108, not RECIPE112): cells/regions 0.46 (clean mean, 0.42-0.54), recipe/exact ops 1.27 (1.24-1.34). The
cells/regions agree; the ops ratios differ, perhaps because the recipe variant or the op counting differs (not
separated). (b) The DRI reports that `understand_m43/evalH.py` records STATS after the rescale re-run, which doubles E43's
cell counts. That fits H-U4's (8,3) "2.0" (496/238): half of it, 248/238 = 1.04, matches the 0.99 recipe108 value here.
All H <= 8 baselines above were counted with this folder's code.

## T205: cells/regions restated against exact region counts (2026-10-18; unverified)
**Method.** `regions_exact.py H T name`: estT exact mode with `mass()` stubbed to 0 (exact splitting never uses masses),
so only geometry is paid. Regions = distinct all-step gate patterns over the exact leaves, counted two ways: *certified*
(forward every ray of the leaf; per step and neuron the gate is on iff the max pre-activation over the rays > 1e-9 x
scale; a leaf with rays strictly on both sides of a gate is a violation) and *center* (the T194 mean-ray hook). Both
agree on all 54 models counted ((4,5,6,8,16)x3, (8,4) and (16,4) x 9 models, (8,5) zoo0), with 0 violations.
`recount112.py`: RECIPE112 (`understand_m43/est75.py`, imported unchanged) clean cells counted once per run.
`restate.py` -> `restate.json` (table below). `mc.py` gives the sampled lower bounds.

| source | (H,T) | published cells / regions = ratio | cells (corrected) | exact regions | lb2e5 (undercount) | **restated** |
|---|---|---|---|---|---|---|
| E43 H-U4 RECIPE112 | (4,3) | 310 / 66 = 4.72 | 155 | 63 | 58 (x1.09) | **2.44** |
| E43 H-U4 RECIPE112 | (5,3) | 372 / 97 = 3.83 | 186 | 97 | 78 (x1.24) | **1.91** |
| E43 H-U4 RECIPE112 | (6,3) | 370 / 134 = 2.76 | 185 | 134 | 106 (x1.27) | **1.38** |
| E43 H-U4 RECIPE112 | (8,3) | 466 / 236 = 1.98 | 233 | 236 | 165 (x1.42) | **0.99** |
| RECIPE112 = recipe108 cells | (16,3) | (T113 report: ~0.47) | 389 | 838 | 421 (x1.99) | **0.46** |
| E45 T-trend | (8,3) | 466 / 165 = 2.82 | 233 | 236 | 165 (x1.42) | **0.99** |
| E45/E46 recipe108 | (8,4) | 6276 / 1171 = 5.36 | 6276 | 3160 | 1171 (x2.70) | **1.99** |
| E48 recipe108 | (8,5) zoo0 | 114155 / 3433 = 33.25 | 114155 | 42322 | 3433 (x12.3) | **2.70** |
| T194 recipe108 | (16,4) | — | 24543 | 28212 | 4090 (x6.90) | **0.87** |

Means over the 5 clean seeds ((8,5): zoo0 only). Two separate corrections:
1. **Cells:** E43's clean_cells are doubled (evalH.py adds the rescale re-run of model 0 to STATS). The recount is exactly
   half on 24/25 models; (4,3) zoo2 is 148 vs 297/2. RECIPE112 and recipe108 give **identical** cell counts at (8,3) and
   (16,3) on all 10 models (recipe108 generalises RECIPE112 at T = 3). Their op counts differ: RECIPE112 4.7-5.9e5 at
   (16,3), recipe108 1.6-1.9e6.
2. **Regions:** E43's regions (`understand_ladder/exact3.count_regions`) are exact except at (4,3) zoo2, where 64 includes
   11 zero-mass degenerate pieces (masses 4e-38..1e-32). There are 53 full-dimensional regions, as counted here. E45/E46/E48
   used the 2e5-sample lower bound, which misses 8% ((4,3)) to 92% ((8,5)) of the regions.

**Restated H-U4 / E45-E48.** Along H at T = 3, cells/regions is 2.44 -> 1.91 -> 1.38 -> 0.99 -> 0.46 for H = 4, 5, 6, 8, 16
(published 4.8 -> 2.0): the fall along H holds, at about half the level. Along T at H = 8 it is 0.99 -> 1.99 -> 2.70
(published 2.8 -> 5.4 -> 33). Regions grow x13.4 and then x13.4 per step, recipe cells x27 and then x18. So cells still outgrow regions,
but by x2.0 and then x1.36 per step, not the published x1.9 and x6.2. "Cost grows FASTER than regions" (E45/E48) holds for
**ops** (x250 from T = 4 to 5 at H = 8, because 5-D masses cost more per cell). It holds only weakly for cells.
**Reconciling T113.** T113's (16,3) RECIPE112 recipe/exact ops ~2.06 is against `arc_m43_repro/exact.py`: mean 2.01
here (`exactpy_T3.json`; 1.92-2.20 per seed). The T194 ratio of 1.27 is recipe108 against estT exact mode, which costs ~5x
more than exact.py at T = 3.

Claims (T205, unverified): [measured] the restated table above; [measured] certified = center pattern counts on all
54 models; [measured] RECIPE112 and recipe108 cells are identical at T = 3; [inferred] the published per-step excess of cells over
regions along T (x6.2 at T = 4->5) is mostly the lower bound's undercount.
Alternatives: "region" = gate pattern (a pattern could be a non-convex union of convex pieces; true convex counts >=
these). Regions of zero Gaussian mass are excluded by construction (exact leaves are full-dimensional), which is why
(4,3) zoo2 differs from exact3. (8,5) is one seed. Compute: geometry-only exact runs, (8,5) zoo0 1.7e9 ops / 10.4
CPU-min, all others < 80 CPU-s each; RECIPE112 recount and exact.py runs < 1 CPU-min.

## T206: exact-mode cost anatomy at (8,4) and (16,4) zoo0 (2026-10-18; unverified)
**Method.** `anatomy.py H T name [chamber|truth]` runs estT exact mode unchanged. Instrumentation only wraps `mass()`
(its ops go to a separate counter and still count in the total) and `_cell_error` (per leaf: exact error mass, mass ops,
simplices, kind, and the leaf's own mass with a throwaway counter, so the leaf sum can be checked). Leaf kinds: *correct* = no
clip leaves a wrong piece, so no mass is computed; *wrong* = the whole leaf is wrong; *mixed* = the rest. The `truth` start
replaces the 4·T! half-chambers by the 2T(T-1) truth cones {x_i > x_j > rest} x sign(x0) (review/chamber_split, T192).
`orthbench.py`: orthant ops per simplex by d. `anat_summary.py` -> `anatomy_summary.json`. (8,3) and (16,3) are included
for the d = 3 contrast.

| zoo0 | start | ops | mass share | leaves: correct / mixed / wrong | error in mixed / wrong | simplices per mass-leaf | mass ops / simplex |
|---|---|---|---|---|---|---|---|
| (8,3) | chamber | 3.5e5 | 9% | 289 / 137 / 26 | 6.5e-4 / 5.7e-7 | 1.7 | 116 |
| (16,3) | chamber | 1.24e6 | 7% | 795 / 224 / 143 | 6.1e-4 / 3.7e-6 | 1.8 | 125 |
| (8,4) | chamber | 1.10e9 | **97%** | 2865 / 3802 / 1788 | 2.3e-2 / 6.4e-4 | 6.3 | 30536 |
| (8,4) | truth | 7.35e8 (0.67x) | 98% | 1300 / 2622 / 953 (0.58x leaves) | 2.3e-2 / 5.4e-4 | 6.6 | 30562 |
| (16,4) | chamber | 4.19e9 | **95%** | 29031 / 14524 / 7997 | 2.1e-3 / 2.5e-5 | 5.8 | 30389 |
| (16,4) | truth | 3.17e9 (0.76x) | 95% | 19511 / 12102 / 4924 (0.71x leaves) | 2.1e-3 / 2.3e-5 | 5.8 | 30401 |

Orthant ops per simplex (`orthbench.py`, 120 random cones per d): d = 3 69; d = 4 30144-30536; d = 5 51965 (173060 on
thin cones, below the det threshold). In runs a d = 3 "simplex" costs ~120 including the pulling triangulation.

**Error-mass distribution (cells with nonzero error).** (8,4): 5590 cells, median 4.6e-8, 90% quantile 5.6e-6. (16,4):
22521 cells, median 2.2e-10, 90% quantile 5.1e-8. Share of total error (and of mass ops) in cells with error below:

| zoo0, chamber | < 1e-8 | < 1e-7 | < 1e-6 | oracle skip at 1 SE |
|---|---|---|---|---|
| (8,4) | 0.017% of error (20% of mass ops) | 0.24% (44%) | 2.1% (69%) | SE 8.5e-5: skip 75% of leaves = 49% of mass ops -> 5.8e8 ops (1.9x) |
| (16,4) | 0.80% (74%) | 5.4% (90%) | 23% (98%) | SE 2.5e-5: skip 93% of leaves = 78% of mass ops -> 1.1e9 ops (3.8x) |
| (16,4) truth start | 0.65% (69%) | 4.8% (87%) | 23% (97%) | SE 2.8e-5: 76% of mass ops -> 8.8e8 ops (4.8x vs chamber exact) |

"Oracle skip": sort leaves by their exact error and drop the smallest while the dropped error sums to <= the sampling
SE at the same cost (T196 units: 2·ops FLOPs vs c_s(H,T) per input; the bias is then <= 1 SE). It is an upper bound on what
a *certified* per-cell bound could skip, since a real method must bound each cell's error without computing it. It
is also conservative in one way: after skipping, the cost and hence the allowed SE change. At T = 3 the whole error
(6.5e-4) is below 1 SE at that tiny cost, so the T = 3 skip figures are meaningless and not shown.
Checks: leaf masses sum to 1 within 1.1e-10 on all 8 runs. The truth start gives the same accuracy (differences
<= 1.5e-12) and reproduces T192 S3's (8,4) 0.58x leaves / 0.67x ops.

**Reading.**
1. From T = 4 exact cost is almost all cone masses (95-97%): ~6 simplices per leaf that needs a mass, at ~3e4 ops
   each (d = 4 Plackett quadrature, 440x the d = 3 closed form). Geometry is 2-5%.
2. 56% (16,4) / 34% (8,4) of leaves are fully correct and cost no mass. Almost all error sits in mixed leaves, which also
   take most mass ops. Fully wrong leaves carry ~1-3% of the error.
3. Mass work concentrates in negligible cells, more so at H = 16: cells with error < 1e-8 take 74% of the mass ops but
   0.8% of the error. With a certified per-cell error bound, at most 49% (8,4) to 78% (16,4) of the mass ops could be skipped
   at 1 SE of bias (x1.9 / x3.8; x4.8 combined with the truth start at (16,4)). This is a constant factor, not a change in
   the growth law: cells still grow ~x13-34 per step.
4. Cheaper per-simplex masses are the other lever: a d = 4 mass costs 440x a d = 3 one. Quadrature order could
   follow the precision needed (1e-8 absolute is enough for most cells) instead of being fixed at full accuracy.

Claims (T206, unverified): [measured] the two tables above; [measured] truth-cone start is lossless on these 4 models;
[inferred] a certified per-cell error bound could save at most ~2-4x at (8,4)/(16,4); [speculative] precision-adaptive
quadrature could cut the per-simplex cost by a larger factor (not tested).
Alternatives: one seed per size. The oracle uses exact per-leaf error; a real bound (e.g. leaf mass <= cap, or a
solid-angle bound) will skip less. Leaf kinds come from mass-op counts (a wrong piece of zero mass is still computed).
Compute: anatomy runs (8,4) 61/38 CPU-s, (16,4) 241/180 CPU-s (including the diagnostic leaf masses, not in ops).

## T216: weight-symmetry quotient test on the target M_16,10 (2026-10-18; unverified) — negative (kill)
**Method.** `symm.py`: for all 120 neuron pairs of `alg_zoo.example_2nd_argmax()`, cosines of incoming vectors
u = [W_ih, W_hh row] and outgoing vectors v = [W_hh column, W_out column], with and without swapping the pair's own
reads/targets (a permutation symmetry must permute both). Cosines are invariant to the per-neuron positive rescaling the
bias-free model allows. `prefix.py`: T192's prefix region counter ported to single-process numpy on this folder's estT copy.
It reproduces T192 exactly (target regions 34 / 792 / 21032 for k = 2 / 3 / 4; 894 cells at k = 3). `symm_model` projects the
chosen neurons' incoming vectors onto their common (sign-aligned) principal direction, so their gates coincide or are
exactly complementary; outgoing weights are kept. `symm_acc.py` and the inline variant run (`symm_variants.json`) give the
accuracy change at T = 10 (paired MC, same inputs, 1e7 / 2e6 samples).

**Results.** No permutation symmetry: max |cos| of outgoing columns is 0.84 (0.84 with the swap). One near-collinear
incoming group: neurons {1, 6, 7, 8} have pairwise cos 0.985-0.991, and neuron 4 is antiparallel to them (cos -0.983 to
-0.99991; pair (4,7) -0.99991). Their outgoing columns differ (|cos| <= 0.71). No dead neurons.

| symmetrised set | acc change at T = 10 (pp) | prediction changed | regions k = 2 / 3 / 4 | growth per step |
|---|---|---|---|---|
| none (target) | 0 | 0 | 34 / 792 / 21032 | x23.3, x26.6 |
| {1,6,7,8} + anti {4} | **-79.43 ± 0.01** (acc 0.953 -> 0.159) | 84% | 26 / 430 / 7880 | x16.5, x18.3 |
| {7} + anti {4} (cos -0.99991) | **-8.87 ± 0.02** | 12% | 32 / 694 / 17100 | x21.7, x24.6 |
| {6, 8} | -69.5 | 74% | 32 / 708 / 17734 | x22.1, x25.0 |
| {1, 7} | -71.6 | 76% | 32 / 680 / 16054 | x21.3, x23.6 |

**Verdict: kill.** The go criterion was growth < 8x/step at <= 0.05 pp. The best case keeps growth >= x16 per step and
costs 79 pp; the mildest case costs 8.9 pp for a 19% region reduction. The near-collinear incoming rows are not a
symmetry: the small orthogonal parts (|sin| ~ 0.01-0.17) carry function. [measured] all numbers above. [inferred]
"brute force after exploiting symmetries" (ARC) does not refer to a neuron-level weight symmetry of M_16,10 visible in
these cosines.
Not tried: exact function-preserving merges (none exist: no exactly parallel rows); symmetries that mix more than 2
neurons by a general linear map (not a permutation/sign); input-space symmetries (time reversal, x -> -x), which belong to
the task, not the weights. Compute: < 1 CPU-min (prefix k = 4 counts 10-27 CPU-s each; MC 1e7 + 3 x 2e6 paired).

## T217: modal event-word branching on exact prefix leaves (2026-10-18; unverified) — modal test fails
**Method.** `eventword.py model k`: exact prefix leaves (`prefix.run`, the T192 start, estT exact geometry) of the
target M_16,10 and of zoo_2nd_argmax(16,4,0) and (16,5,0) (first k steps), each split by every order plane x_i = x_j.
Each piece then has one ordering and so one event word: for t >= 1, x_t is a new record R, a new 2nd S or low L.
Exact piece masses come from estT.mass (d = k), giving the joint law p(event word, gate word) of the first k steps. Reported:
B|E = exp H(G | E) (mass-weighted branching given the event word); B = exp H(G) unconditional; the modal share
= sum over e of max over g of p(e, g); n99 = mass-weighted number of gate words needed to cover 99% of each event word's mass.
Variant "|E,s0": the event word plus sign(x_0). Step factors (in brackets) use B = 2 at k = 1 (gate word = sign(x_0)).

| model | k | regions | B\|E (step) | B\|E,s0 | B (step) | modal share | modal, +s0 | n99 | mass sum - 1 |
|---|---|---|---|---|---|---|---|---|---|
| target | 2 | 34 | 6.23 (x3.11) | 3.55 | 12.3 (x6.13) | 33.9% | 49.5% | 11.5 | -4e-14 |
| target | 3 | 792 | 13.90 (x2.23) | 8.58 | 65.9 (x5.38) | 23.7% | 31.4% | 44.8 | -2e-13 |
| target | 4 | 21032 | 26.03 (x1.87) | 16.82 | 301.7 (x4.58) | 20.6% | 25.3% | 138.6 | 3e-11 |
| zoo16_4 | 2 | 34 | 5.39 (x2.70) | 3.07 | 10.8 (x5.38) | 36.9% | 58.8% | 12.0 | -5e-14 |
| zoo16_4 | 3 | 870 | 13.02 (x2.41) | 8.04 | 76.0 (x7.06) | 26.7% | 35.9% | 46.3 | -2e-13 |
| zoo16_4 | 4 | 26742 | 45.07 (x3.46) | 29.12 | 655.2 (x8.62) | 13.5% | 17.8% | 241.3 | 4e-11 |
| zoo16_5 | 2 | 34 | 5.51 (x2.76) | 3.14 | 10.4 (x5.22) | 37.7% | 56.5% | 10.0 | -5e-13 |
| zoo16_5 | 3 | 824 | 10.85 (x1.97) | 6.70 | 55.4 (x5.31) | 28.3% | 40.0% | 35.8 | -3e-12 |
| zoo16_5 | 4 | 23956 | 29.77 (x2.74) | 19.24 | 422.3 (x7.62) | 17.5% | 23.2% | 144.8 | 3e-11 |

**Reading.**
1. **Not mass-irrelevant (modal test fails).** At k = 4 the most likely gate word holds only 21% (target), 13% and 18%
   (zoo) of each event word's mass (go: >= 99%). Covering 99% needs ~140-240 gate words per event word, and this grows
   x3-4 per step. Adding sign(x_0) changes little (25%).
2. **The mass-weighted growth is far slower than the region growth.** exp H(G) grows x5.4 then x4.6 per step on the target,
   while regions grow x23 then x27. Given the event word, branching grows x3.1, x2.2, x1.9 per step (<= 3 from k = 3; zoo
   (16,4) x3.5 at k = 4). Most regions carry negligible mass, as T206 found per cell.
3. **Against the go/kill lines.** Go needed branching <= 3/step AND modal >= 99%. The per-step factor passes from k = 3
   on, but the modal share fails by a wide margin, so this is no go. Kill was "branching >= 8": the per-step factor is < 8,
   but the cumulative B|E at k = 4 is 26-45. Which reading counts is for the DRI to decide.
4. [speculative] If the step factor of B|E keeps falling (3.1, 2.2, 1.9), exp H(G | E) at k = 10 is ~1e3-1e4 per event
   word. That is an entropy (typical-set) count, not a certified cover: the error mass sits in the low-mass tail
   (T206: cells with error < 1e-8 hold 0.8% of the error), and the cover size grows like n99 (x3/step), not like B|E.

Claims: [measured] the table (mass sums within 4e-11 of 1; target regions reproduce T192 34/792/21032; zoo16_4 k = 4 = the 26742 exact regions of T194); [inferred]
event words explain the mass but not the count of gate words, since the modal share keeps falling (x0.7 then x0.87 per step on the target).
Alternatives: richer event words (adding values of the top-2 gap, age of the max, or the sign of every input) may raise the
modal share; only sign(x_0) was tried. k = 5 was not run (target 632k regions: too slow in pure numpy here; T192's
numba code could). Compute: masses k = 4 127-173 CPU-s per model (1.37 pieces per leaf at k = 4); all runs < 8 CPU-min.

## T221: mass coverage curve of exact prefix regions (pre-registered 2026-10-18, before computing; unverified)
**Rule (written before any number was computed).** For the target M_16,10 prefixes k = 2, 3, 4 and zoo (16,4) and (16,5)
seed 0 at the same k, sort the exact prefix regions (distinct gate words) by exact Gaussian mass, and let N_delta(k) be the
smallest number of regions covering 1 - delta of the mass, for delta = 1e-2 ... 1e-6. Also report the per-step growth of
N_delta and the mass held by regions below 1e-12.
- **Reopen the exact route** if N_1e-5 grows <= ~6x/step (against ~25x for the count): a best-first, mass-ordered
  enumeration with an exactly known uncovered mass could then reach M_16,10 at ~1e7-1e9 regions.
- **Dead for good** if N_delta grows like the count (~25x/step).
- In between: inconclusive, reported as such.
Data: the T217 pieces (`eventword/*.npz`: per piece, gate-word id and exact mass), summed per gate word. No new enumeration.

**Results** (`coverage.py` -> `coverage.json`; mass sums within 4.9e-11 of 1). N_delta (growth per step):

| model | k | regions | N 1e-2 | N 1e-3 | N 1e-4 | N 1e-5 | N 1e-6 | mass (count) of regions < 1e-12 |
|---|---|---|---|---|---|---|---|---|
| target | 2 | 34 | 22 | 31 | 33 | 34 | 34 | 0.0e+00 (0) |
| target | 3 | 792 (x23.3) | 201 (x9.1) | 388 (x12.5) | 527 (x16.0) | 634 (x18.6) | 708 (x20.8) | 0.0e+00 (0) |
| target | 4 | 21032 (x26.6) | 1737 (x8.6) | 4018 (x10.4) | 7021 (x13.3) | 10233 (x16.1) | 13164 (x18.6) | 3.6e-10 (1783) |
| zoo16_4 | 2 | 34 | 23 | 30 | 34 | 34 | 34 | 0.0e+00 (0) |
| zoo16_4 | 3 | 870 (x25.6) | 244 (x10.6) | 430 (x14.3) | 604 (x17.8) | 720 (x21.2) | 796 (x23.4) | 0.0e+00 (0) |
| zoo16_4 | 4 | 26742 (x30.7) | 3272 (x13.4) | 7007 (x16.3) | 11323 (x18.7) | 15556 (x21.6) | 19190 (x24.1) | 3.4e-10 (1577) |
| zoo16_5 | 2 | 34 | 17 | 27 | 31 | 33 | 34 | 0.0e+00 (0) |
| zoo16_5 | 3 | 824 (x24.2) | 168 (x9.9) | 325 (x12.0) | 490 (x15.8) | 628 (x19.0) | 709 (x20.9) | 1.7e-12 (2) |
| zoo16_5 | 4 | 23956 (x29.1) | 1836 (x10.9) | 4273 (x13.1) | 7467 (x15.2) | 10982 (x17.5) | 14417 (x20.3) | 4.9e-10 (3041) |

**Verdict.** N_1e-5 grows x18.6 then x16.1 per step on the target and x17.5-21.6 on the zoo models. Even N_1e-2 grows
x8.6-13.4. By the letter of the rule this is *in between* (above 6x, below the count's x23-31), but it is far closer
to "dead". Every N_delta grows at 50-80% of the count's rate in log terms, and the rate rises as delta shrinks.
[speculative] Projecting the target from k = 4 at x16/step gives N_1e-5(10) ~ 1e4 x 16^6 ~ 2e11 regions, far above the
1e7-1e9 a mass-ordered exact route needed. Even x10/step (N_1e-3) gives ~4e9. **Mass-ordered exact enumeration does not
reopen the exact route for M_16,10** [inferred, from k <= 4]. This reconciles with T217's slow exp H(G) growth (x5):
the entropy is set by the heavy bulk, while coverage to 1e-5 is set by a long tail of mid-mass regions that grows almost
like the count. Regions below 1e-12 hold <= 4.9e-10 of the mass (1577-3041 regions at k = 4).
Claims: [measured] the table; [inferred] the verdict. Alternatives: k = 5 (T192, numba) could bend the curve. k <= 4 is
short, and the growth of N_1e-5 fell from x18.6 to x16.1 on the target. Coverage per *cell* rather than per region,
or after merging regions with equal futures (T188), would differ. Compute: < 1 CPU-s (saved masses).

## Compute accounting
Counted ops = `m43.Ops` inside the estimator (recipe or exact), as estT_fix; leaf-sum masses, region patterns and MC are
diagnostic and excluded. Totals (CPU, single thread): (8,3)+(16,3) all runs < 1 min; (16,4) recipe 9 x 2.3-2.7 min
= 23 min, exact 9 x 4-5 min = 40 min; (8,4) exact re-runs 9 x ~1 min; MC: 1e7 forward passes x 36 models + 2e7 x 4,
region sampling 2.2e6 x 36 (all minutes). Max estimator ops per model 4.8e9 (exact (16,4) zoo2); no stop guard fired.

## Claims (all unverified)
- [measured] recipe108 (frozen) beats sampling at equal ops on all 5 seeds + 4 noisy copies at (16,3) (ratio <= 4e-5)
  and (16,4) (ratio <= 0.017; <= 0.035 in T196 units), against exact enumeration that matches 1e7 MC (|z| <= 2.98, fresh 2e7 MC |z| <= 1.99).
- [measured] cells/regions (recipe108 cells / exact gate-pattern count) falls with H at fixed T: 0.99 -> 0.46 (T=3),
  1.99 -> 0.87 (T=4), from H = 8 to 16.
- [measured] Along T at H = 16 recipe cells grow faster than regions (x63 vs x34) and ops x1700 per step, as at H = 8:
  width lowers the level, not the per-step growth (2 widths x 2 lengths only).
- [measured] The 2e5-sample region lower bound undercounts exact regions by 1.4x-6.9x, worse at larger H and T; earlier
  cells/regions ratios based on it are overstated.
- [measured] At (8,3) this exact region count equals the independent count in `understand_m43/evalH_8.json`
  (238/232/232/226/250 for seeds 0-4).
- [inferred] recipe108's savings over exact enumeration are small (<= 1.6x ops at T = 4, negative at T = 3); its
  per-cell work, not the cell count, sets the cost.
- [speculative] M_16,10 by recipe108 needs >= 1e18 ops.

## Alternatives not ruled out
- "Region" here = distinct gate pattern; a pattern may be a non-convex union of several linear pieces, so the true
  number of convex linear regions can be higher (ratios would fall further). Thin regions missed by exact enumeration
  would lower the count; leaf masses summing to 1 rule out missing mass, not missing patterns of tiny mass.
- The pattern is evaluated at one interior point (mean of rays) per leaf; a degenerate leaf could give a boundary
  pattern. Not checked beyond the (8,3) agreement.
- The fall of cells/regions with H could reflect the zoo models being better trained at H = 16 (fewer active gate
  switches near tie boundaries) rather than width itself; not separated (one training family).
- Only T = 3, 4 at H = 16: a two-point growth estimate per width; T = 5 at H = 16 not run (estimated ~5e12 ops).
- Noisy copies share one noise draw family (seed 7); other sigmas not tried.

## Failure modes
- Cost: per-step growth x1700 in ops makes (16,5) ~5e12 ops (beyond the 1e11 brief limit; not attempted).
- Noise: at H = 16, sigma = 0.005 relative noise drops accuracy to 0.96-0.98 (bulk errors); the recipe stayed accurate
  here, but heavier noise was not tested.

## Proposals (for the DRI)
- Re-state H-U4 and the E45/E46/E48 cells/regions numbers against exact region counts (the lb2e5 undercount).
- Count exact regions at (8,5) zoo0 (needs ~3e11-op exact run, already done in T185; add the pattern hook) to see
  whether "33" survives.
