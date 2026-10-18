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
