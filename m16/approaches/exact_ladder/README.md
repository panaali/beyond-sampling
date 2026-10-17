# exact_ladder (T46): exact cone enumeration + exact cone measure, H=16, T=2..6

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [arc_m43_repro](../arc_m43_repro/README.md) (E32) · [cone_enum](../cone_enum/README.md) (E14)

## Question

The 2nd-argmax RNNs have no biases, so correctness depends only on x/|x|. Accuracy is the normalised solid angle of a
union of polyhedral cones in R^T. Each cone is (full gate pattern over T steps) x (argmax-output region) x (order
region fixing which index is 2nd). For the zoo models with H=16 and T=2..6 (seeds 0-4 plus the leaderboard's 24 noisy
copies), how many cones are there? What does exact enumeration plus exact cone measure cost, with FLOPs counted as
implemented? Does this purely mechanistic method beat sampling at equal compute, and at which T does it stop? This is
the mechanistic counterpart of H-E11, which is a quadrature result.

## Method

The method is deductive and mechanistic: no random or pseudorandom numbers anywhere, and the leaderboard's determinism
check passes. The pipeline has three stages.

1. **Gate-cell enumeration, exact and incremental over time** (`cones.py: iter_cells, _children, lift, clip`).
   - The step-t gates depend only on (x_1..x_t), so a step-t cell is a pointed polyhedral cone in R^t.
   - Going to step t+1 lifts the cell to C x R and cuts it by the 16 pre-activation hyperplanes z_{t+1} = B x' + Wi y.
     The first cut uses the unit with the largest |Wi| and is applied analytically (`lift`), so the cone stays
     pointed. The other 15 cuts are double-description clips.
   - A cone is represented by its extreme rays plus, for each ray, the bitset of constraints tight at that ray. New
     rays come only from adjacent (+,-) pairs, using the combinatorial adjacency test (Fukuda-Prodon). Tight sets are
     inherited exactly, so the only tolerance is the sign test |a.r| < 1e-11 on unit vectors. Empty cells are never
     created.
   - The traversal is depth-first and streaming, so memory is bounded: under 300 MB at T=5.
2. **Outcome split** (`outcome_pieces`). On a final cell the logits are linear, L = Wo A_T x.
   - For each j, clip by "argmax = j" (T-1 halfspaces), then split on x_j - x_k. A branch is closed as **wrong** as
     soon as two coordinates exceed x_j. Exactly one coordinate above x_j is correct; none above is wrong.
   - The all-dead cell (h_T = 0) and any identically tied logit pair follow np.argmax's first-index rule (T60, below).
   - Correct and wrong pieces partition R^T, so **accuracy = 1 - sum(wrong-piece measures)**. Wrong pieces are
     1.0-1.7x fewer than correct ones, and their total mass is small, which helps the error budget.
3. **Exact cone measure.**
   - Each wrong piece gets a pulling triangulation into simplicial cones (`triangulate`). It is built from the DD face
     lattice, with numeric rank checks on candidate facets.
   - Each simplicial cone with unit rays V has the measure
     P = |det V| Gamma(T/2)/(2 pi^{T/2}) ∫_Δ |V^T μ|^{-T} dμ (`cone_measure`).
   - T=2 uses the angle and T=3 uses the Van Oosterom-Strackee formula (both closed form).
   - For T>=4 a Grundmann-Möller degree-7 rule is used. Its a-posteriori error is |Q_7 - Q_5|, and longest-edge
     bisection continues until err <= 1e-6 x value.
   - The integrand is smooth on narrow cones, and the smallness of thin slivers is carried by |det V|. So this method
     has relative accuracy. A first attempt, Plackett's identity integrated along R(t) = I + t(R - I) with
     tanh-sinh, failed on these cones; see Results 4.

Files:
- `cones.py`: the geometry.
- `ladder.py`: `exact_acc(model)`.
- `estimator.py`: the leaderboard interface. It sets `SUPPORTS` for (16,2..5) and `FIXED_COST=True`. It keeps a
  per-weights cache so the determinism-check re-calls are not recomputed; the charged cost is unchanged.
- `run_one.py`: one model, written to `runs.jsonl`.
- `summarize.py`: produces the table below from `est_info.jsonl`, which is the estimator's per-model log.
- `extrap6.py`: T=6 cost extrapolation from a systematic 1/K subsample of the last step.
- `diag_outcome.py`: counts outcome components.
- `queue1.sh`, `queue2.sh`: the run queues that were used.

## Status

Done (2026-10-17).
- T=2, 3, 4: exact on all 29 leaderboard models.
- T=5: exact on seeds 0-4. The noisy copies were not run, because each takes about 1 CPU-h in Python.
- T=6: cost and count extrapolated from an exact prefix plus a 1/200 subsample of the last step.

## Results

### Main table

Costs are in FPE (forward-pass equivalents, leaderboard rule 1: 1 FPE = 6e3 x flops(16,T)/flops(16,10) FLOPs, i.e.
1200 / 1800 / 2400 / 3000 / 3600 FLOPs for T = 2..6). The table gives group means.

- "err bound" is the largest summed a-posteriori quadrature error estimate of any model.
- "ratio <=" is the MSE ratio to sampling at equal cost if the whole error bound were realised: err² / (p(1-p)/cost).
- The leaderboard's own MSE ratio cannot resolve values below its truths' Monte Carlo noise (truth se 1-9e-5),
  so it prints about 0.00 ± 0.1.
- "z" is (exact - MC truth)/truth_se against the leaderboard truths (4M samples each, independent per model). If the
  method is exact, z ~ N(0,1). That is a test the method could have failed.

Commands: `python m16/leaderboard.py run exact_ladder --H 16 --T <T>` (T=2,3,4), the same with `--groups seed` for
T=5, then `cd m16/approaches/exact_ladder && python summarize.py`.

SUMMARY_TABLE

| T | gate cells (seeds) | wrong pieces | simplices | cost FPE (seeds; noisy) | quadrature err bound | MSE ratio vs sampling at that cost | vs ARC bar 2.4-10 (E32) | fits the 64k / 1M FPE budget? |
|---|---|---|---|---|---|---|---|---|
| 2 | 34 | 2 | 2 | 17 | 0 (closed form) | 0 (z rms 0.95) | far below | yes / yes |
| 3 | 838 | 305 | 511 | 666 | 0 (closed form) | 0 (z rms 0.93) | far below | yes / yes |
| 4 | 28,212 | 18,524 | 96,845 | 3.9e5; 5.6-6.6e5 | <= 7e-8 | <= 2e-8 (z rms 1.10) | far below | **no** / yes |
| 5 | T5_CELLS | T5_WRONG | T5_SIMP | T5_FPE | T5_ERR | T5_RATIO | far below | **no / no** |
| 6 (extrap.) | T6_CELLS | T6_WRONG | T6_SIMP | T6_FPE | - | - | - | no / no |

### 1. Exactness checks (measured)

- **Partition test.** With `check=True`, the correct and wrong pieces are measured separately, and their sum must
  be 1. T=2: 1 - 1.1e-16. T=3: 1 - 1.3e-14.
- **Tolerance changes at T=4, seed 0.** Changing (s, rtol) from (3, 1e-6) to (3, 1e-12) or (5, 1e-9) moves the
  wrong mass by at most 1.3e-11. The reported bound is 7.8e-10.
- **Against Monte Carlo, T=2-4, 29 models.** z rms is 0.95, 0.93 and 1.10, with mean z between -0.35 and +0.15.
  There is no sign of a bias at the 1e-5 level.
- **T=5, seed 0.** The exact value is 0.9901084. The leaderboard truth is 0.9899595 ± 5.0e-5 (z = +3.0). Two
  independent 16M-sample runs of our own gave 0.9901324 and 0.9900694, pooled 0.9901009 ± 1.75e-5, so exact is at
  z = +0.43 against them. The leaderboard's H16_T5 seed-0 truth is therefore about 2.7 sigma low. That is plausible
  as a fluctuation but worth a recheck (see the T5 seeds rows).
- **Synthetic tie cases (T60).** We tested duplicate readout rows (Wo[1] = Wo[0], and Wo[2] = Wo[1]) and a zero
  readout row on the T=3 seed-0 model. The exact result matches 4M-sample MC: z = -0.31, +0.09 and -0.38. The
  partition sums to 1 - 1.3e-14.

### 2. Where the cost goes (seed 0, FLOPs)

| T | enumeration (lift + clip + adjacency + outcome split) | triangulation | measure | total FPE |
|---|---|---|---|---|
| 3 | 1.06e6 | 3.4e4 | 2.9e4 (closed form) | 624 |
| 4 | 9.3e7 | 2.6e7 | 7.0e8 | 3.4e5 |
| 5 | 7.8e9 | 1.09e10 | 3.53e11 | 1.24e8 |

From T=4 on, the measure step dominates, at 86% (T=4) and 95% (T=5). There are two reasons. The number of
simplices per wrong piece grows: 1.7, 5.1 and 22 at T=3, 4, 5. And each degree-7 rule costs about 3-20k FLOPs per
simplex. **Enumeration alone, with the measure free, already costs 7.2e6 FPE at T=5**, which is 7x the leaderboard's
largest budget.

### 3. Growth law (gate cells, all exact except T=6)

Final gate cells, seed-mean: 34, 838, 28.2k, 786k (seed 0) at T = 2, 3, 4, 5. Within one model, the per-step counts
of the T=5 seed-0 model are 2, 34, 824, 23,956, 786,114. Each extra step multiplies the count by 25-33x, with no
sign of saturation.

GROWTH

### 4. Negative results along the way (kept)

- **Plackett / tanh-sinh orthant integration (`cones.orthant`)** is correct on random simplices: it matches Genz to
  1e-9. On the wrong pieces it fails. Their simplicial cones are slivers (median smallest eigenvalue of R 4e-8), and
  the path integral then returns 1/16 + (large cancelling terms). Even at level 6 the sum over 90k simplices moved by
  1e-5, and the T=4 total mass came out as 1 - 1.4e-5. The solid-angle integral with Grundmann-Möller fixes this,
  because the smallness sits in |det V|. The Plackett code stays in `cones.py` for reference; it is not used.
- **Measuring correct pieces instead of wrong ones** costs 1.7x more simplices at T=4. At T=5 the two counts are
  equal (776k vs 794k pieces), so the complement trick stops helping.
- **Outcome merging.** See "outcome cells" below.

OUTCOME

## Compute accounting

- Counted, as implemented (`cones.Ops`):
  - Every model-dependent multiply-add (2 FLOPs).
  - Every transcendental (sqrt, arctan2, pow), charged 20 FLOPs each.
  - Every 64-bit bitset AND/compare/popcount in the adjacency tests (1 FLOP per word).
  - Ray normalisations, lifting matrices B = Wh A_t, logits Wo A_T.
  - Rank checks in the triangulation (SVD charged 2mn² + 2n³).
  - Determinants, quadrature nodes and bisections in the measure.
- Not counted:
  - Python control flow and list handling.
  - Copying arrays.
  - The model-free Grundmann-Möller node tables (`PRECOMPUTE_FLOPS = 0`; tiny).
- Totals per model are in `est_info.jsonl` (field `flops`, plus the `parts` breakdown) and in the table above.
- Wall clock: T=4 takes about 31 CPU-s per model, T=5 about 3,400-3,600 CPU-s, on 1 core.
- The leaderboard's time column is 2-57x sampling's time at equal FPE. Python overhead per clip (about 30 µs) is far
  above its FLOP count. This is under-counting of *time*, not of FLOPs: the charged FLOPs are a count of the
  arithmetic the code performs.
- **Compiled code (DRI suggestion) was not built.** The charged cost is FLOPs, and compiled code changes only wall
  clock. At T=5 the method is already 1.2e8 FPE, and even with a free measure step it is 7e6 FPE, so a faster
  implementation cannot change the verdict. Compiled code would only let T=5 run on all 24 noisy copies and T=6 run
  in full instead of extrapolated (backlog row, see below).

## Claims

- [measured] For H=16, exact enumeration plus exact measure gives the accuracy to within <= 7e-8 (T<=4) and <= T5ERR
  (T=5), with no random inputs. On 29 models per T (T=2-4) it agrees with independent 4M-sample truths at z rms
  0.93-1.10.
- [measured] Cost, seed means: 17 FPE (T=2), 666 FPE (T=3), 3.9e5 FPE (T=4; noisy copies 5.6-6.6e5), T5_FPE (T=5).
  The step-to-step growth is about 40x, about 600x and about 300x.
- [measured] At its own cost the method beats sampling by many orders of magnitude at every T where it finishes. The
  MSE-ratio bound is <= 2e-8 at T=4, far below ARC's demonstrated bar of 2.4-10 (E32). The leaderboard prints 0.00,
  limited by truth noise.
- [measured] Against fixed budgets the method stops early. It fits 64k FPE only for T<=3, and 1M FPE only for T<=4
  (it fits at T=4, where the noisy copies cost up to 7.6e5). At T=5 it is 124x over 1M; enumeration + outcome split without
  any triangulation or measure is already 2.6x over.
- [measured] The gate-cell count grows 25-33x per step through T=5: 34, 838, 28k, 786k.
- [inferred] Extrapolating the per-step factor (about 30) to T=10 gives about 2e13 cells, and the cost factor (about
  300 per step) gives about 1e20 FPE. Exact enumeration at T=10 is infeasible by 14 or more orders of magnitude over a
  1M budget. This is consistent with H-M7, whose sampled lower bound is 1e7-1e8 patterns.
- [measured] Merging cones with the same outcome does not rescue the method. OUTCOME_CLAIM
- [measured] T60 fix: identically tied logits now follow np.argmax's first-index rule. On the leaderboard models
  this case never occurred (0 zero margins at T=2,3). For generic float weights it needs exact cancellation, so it is
  measure-zero [inferred]. So the earlier T=2-5 numbers are unaffected.

## Alternatives not ruled out

- **Is "beats sampling at equal compute" the right comparison for a fixed-cost exact method?** Its error is about 0
  at any cost it reaches, so the MSE ratio is meaningless above its cost and undefined below it. We therefore also
  report the cost against the leaderboard budgets. A budget-capped, anytime variant was **not built**: exact
  prefixes plus a deterministic rule for the unexpanded mass. That variant is the fair test at T=5 and is a backlog
  item. Its ceiling is set by the enumeration: even a perfect remainder rule must stay below 1M FPE, while the full
  T=5 enumeration costs 2.6e6.
- **A better triangulation or measure could cut the T=5 cost by up to about 47x.** That is the factor at which
  triangulation and measure would be free and only the enumeration remained (2.6e6 FPE). It would not reach 1M FPE. We did not try:
  - a Delaunay or placing triangulation with fewer simplices;
  - lower-degree rules for slivers;
  - measuring whole cells with the Gram/Brianchon-Gram relation in even dimension.
- **Different tie or tolerance conventions** could move individual pieces. The partition-sum checks and the z-tests
  bound the effect at 1e-5 (T=4) and 2e-5 (T=5) relative to MC. A deeper exactness test (rational-arithmetic DD) was
  not run.
- **T=6 numbers are an extrapolation** from a systematic 1/200 subsample of step-6 expansions. Validated on T=5 seed
  0 at K=20, it was within 1-2% for cells, pieces and simplices, -17% for measure FLOPs, and -3% for wrong mass.

## Failure modes

- Cost grows about 30x per step in cells and about 300x in FLOPs, and the method has no graceful degradation: it gives
  the exact answer or nothing.
- The method assumes no biases (cones through the origin) and np.argmax's first-index tie rule.
- Wide simplicial cones (rays more than 90° apart) need many bisections in the measure. This was rare here (1% of
  simplices at T=4) but they dominate the measure cost.
- With the Python implementation, T=5 needs about 1 CPU-h per model, and T=6 would need about 30-40 CPU-h.

## Exhaustion checklist (rule 5) for the negative conclusion "exact deduction stops at T=4-5"

Tried:
- the complement (wrong-piece) measure;
- two exact measure methods (Plackett path integral, which failed; solid-angle Grundmann-Möller, which works);
- three rule/tolerance settings;
- streaming DFS enumeration;
- outcome-component merging diagnostic;
- T=6 subsample extrapolation.

Not tried:
- a compiled implementation (wall clock only);
- an anytime / budget-capped hybrid;
- fewer-simplex triangulations;
- symmetry reductions (sign symmetry x -> -x does not hold for ReLU RNNs; we found no other exact symmetry);
- coarser exact classes (e.g. merging cells by the last-k-step gates before the outcome split).

None of the untried items reduces the enumeration count, which alone exceeds 1M FPE at T=5.
