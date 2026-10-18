# r1_cont (T317): R1W V1 with continuity restarts + executed B2 prefix tree (cost only; estimates = V1)

↑ [r1_walk (V1)](../r1_walk/README.md) · [r1_v1b2 (T313)](../r1_v1b2/README.md) · Fable round 4 D1 (branch
claude/project-handoff-priorities-zpzfm8, `review/fable_review_r4/README.md`) · [TASKS](../../../TASKS.md)

## Pre-registered steps (DRI, 2026-10-18; written here before the 80-model runs)
1. Implement continuity restarts and the B2 prefix tree, both executed (`cont.py`).
2. Assert max |est - V1| <= 1e-9 on all 80 DEV2 at G = 48. Stop if this fails.
3. G ruling: G = 16 is allowed only if max |est(G=16) - est(G=48)| <= 1e-4 pp (1e-6 absolute) on DEV2, checked
   before any HP2 run. Otherwise G = 48.
4. Instrument FLOPs (Directive 8). Stop if C > 8.1k FPE (at 5,948 FLOPs/FPE).
5. Freeze, and send to the T281 scorer (HP2).

## Method (`cont.py`)
- **Continuity restart.** At a gate flip t* = (s*, j*), the values z_t(t*) at all steps follow from the current
  cell's affine form. Exactly one gate toggles, and only the tangents of steps > s* are recomputed (560 FLOPs per
  step). Cached per-step root minima of the steps < s* stay valid (their affine pieces are unchanged), so only the
  roots of steps >= s* are recomputed.
- **B2.** The first cell is executed as a prefix tree over (input, tangent) prefixes across the 720 triples.
- The FLOPs are counted as executed (formulas in the docstring).

## Results (`python run80.py`, `results.json`; all 80 DEV2)
- **Step 2, passed:** max |est - V1| = **6.7e-16** (<= 1e-9).
- **Step 3: G = 16 not allowed.** Max |est(G=16) - est(G=48)| = 2.5e-6 (2.5e-4 pp), above the 1e-6 (1e-4 pp)
  ruling, so **G = 48**. For information only: C(G=16) = 6,082 FPE.
- **Step 4, passed:** **C = 7,775 FPE** (mean, 5,948 FLOPs/FPE) <= 8.1k. Frozen V1 was charged 13.3k; T313's
  charged-only B2 count gave 10.96k.
- DEV2 (development): RMS .795 pp, F = 6.78.
- [inferred projection] HP2 at V1's HP2 RMS (.99 pp) and C = 7.78k gives F ≈ 16.35 x 7.78 / 13.3 ≈ 9.6.
- **Step 5:** `estimator.py` frozen and sent to the T281 scorer. **The claim is HP2's score alone.**

Forking paths: the estimates are identical to the already-scored V1, so this adds no new estimate variant. It is a
cost re-implementation, and an HP2 re-run under the executed cost.

## T317b pre-registration (DRI, 2026-10-18, before running)
Choose the smallest G in {20, 24, 32} with max |est(G) - est(48)| <= 1e-4 pp (1e-6 absolute) on all 80 DEV2, and report
C at each G. If one qualifies, freeze it as T317b and send it to the T281 scorer.

## T317b results (`python gscan.py`, `results_g.json`)

| G | max |est(G) - est(48)| | C (FPE) | qualifies (<= 1e-4 pp) |
|---|---|---|---|
| 20 | 1.4e-6 pp | **6,293** | yes |
| 24 | 2.0e-9 pp | 6,505 | yes |
| 32 | 1.7e-13 pp | 6,928 | yes |

**Chosen: G = 20** (the smallest qualifying), frozen as `estimator_b.py` (T317b) and sent to the T281 scorer.
- [inferred projection] HP2 F ≈ 16.35 x 6.29 / 13.3 ≈ 7.7, at unchanged HP2 errors (they move by <= 1.4e-6 pp).
- The claim is HP2's score alone.

## T317 claim: sealed HP2 (T281 scorer, claude/hp1-scorer bbf321c8; hashes match)
- Estimates asserted identical to V1 on HP2 (max |diff| 6.7e-16).
- **HP2: C = 7,765 FPE** (range 5.7k-10.8k), **F = 9.57 [6.89, 12.43]**, bias +0.53 pp, RMS 0.99 pp.
  **Point lenient pass. The upper CI is > 10**, so it is not a candidate under the CI condition.
- HP1: clean 5.3; noisy 52-228 (outside scope).
- Scorer's cost audit: the counting matches the executed loops. Counting every comparison and boolean op puts
  correct_interval at ~216 FLOPs, not 146: +1.2% C, F ≈ 9.69. np.unique keying is treated as uncharged bookkeeping.
- **T317b (G = 20), HP2 (claude/hp1-scorer d956eee4): C = 6,290 FPE, F = 7.75 [5.58, 10.07].**
  - Estimates are within 1.4e-6 pp of V1.
  - **Point lenient pass. The upper CI is 10.07, just above 10**, robustly (10.05-10.16 across 20 bootstrap
    seeds).
  - HP1 clean 4.3; noisy copies fail.
  - This is the best deductive (deductive + declared approximations) HP2 score in the project so far.

## T317c pre-registration (DRI, 2026-10-18; before any T317c run)
Exact, cost-only savings that leave T317b's estimates unchanged to <= 1e-12 on all 80 DEV2. No change to G, the closure or
any approximation. correct_interval is counted at the scorer's ~216 FLOPs, and every saving is actually executed.
If the honest C falls by >= 3%: freeze and send to the scorer. Otherwise report the saving and stop.

**Candidate (from a cost census before any T317c run, `fused_s8`):** walks 81.4%, Phi work 18.3%, combination 0.3% of
C. Of 16,264 charged Phi endpoint evaluations, **13,116 are inner endpoints shared by consecutive cells' correct
pieces** (cell k's piece ends at t*, exactly where cell k+1's piece starts, in the same float). Merging touching
pieces telescopes the Phi (and Phi^8) differences exactly, up to rounding (~1e-16). A further 823 endpoints are at
exactly 0 (Phi(v2) is a model-free table), and the pieces ending at tmax merge with the model-free tail.

## T317c results (`python run_c.py`, `results_c.json`; all 80 DEV2)
- **Estimates equal T317b: max |diff| = 0.0** (<= 1e-12).
  - Merging is real: on fused_s8's nonblind g23 line, 4,769 pieces become 724 intervals.
  - Per-node profile differences are 5.6e-17.
- **Honest C:** T317b recounted (correct_interval 216) = 6,391 FPE; **T317c = 5,435 FPE (-15.0%)**. That clears
  the >= 3% bar, so `estimator_c.py` is frozen and sent to the T281 scorer.
- [inferred projection] HP2 F ≈ 7.75 x (5,435 / 6,290) ≈ 6.7, upper CI ≈ 10.07 x 0.86 ≈ 8.7, at unchanged HP2 errors.
- The claim is HP2 alone.
- Forking paths: the estimates are identical to the already-scored T317b, so this is a cost-only re-implementation
  with no new estimate variant.

**Official count update (T281 scorer 379498b6, DRI ruling):**
- correct_interval is **229 FLOPs** per row-call (itemised in hp1_scorer/cost_t317.py), not the 216 used in
  cont_c.py. T317c's charged C is therefore low by 13 FLOPs per cell call, a few tenths of a percent. The 15% saving
  and the projection are unaffected.
- Official HP2 under 229: T317 C 7,880 FPE, F 9.71 [6.99, 12.61]; T317b C 6,405, F 7.89 [5.68, 10.25].
- T317c's estimates reproduce T317b's (diff 0.0 on DEV2), satisfying the scorer's <= 1e-12 condition for r1_cont
  submissions.

## T317c claim: sealed HP2 (T281 scorer, claude/hp1-scorer 9b4e41df; hashes match)
- Estimates equal T317b to <= 2.2e-16 (asserted).
- Official cost (correct_interval 229): **C = 5,455 FPE**.
- **HP2: F = 6.72 [4.84, 8.73]. Point lenient pass with the upper CI <= 10: the first candidate under the fixed rule.**
- HP1: clean 3.75; noisy copies 39-206 (outside scope).
- **Scorer's literal-execution note:** the frozen profile() computed ndtr everywhere and then masked it with
  np.where, so the model-free-table saving was accounted, not fully executed. Charging those calls gives 5,713 FPE
  and **F 7.04 [5.07, 9.14]**, the same verdict.
- **Fix (after scoring; estimates unchanged):** profile() now evaluates Phi only at weight-dependent arguments,
  masking before the ndtr call, and correct_interval is set to 229.
  - Max |est - T317b| = 0.0 on all 80 DEV2.
  - Cost as executed: **5,488 FPE** on DEV2 (it also charges the a = tmax tail-start Phi calls that are actually
    made).
  - The fixed file goes to the scorer to confirm. The HP2 verdict holds under either count.
  - **Scorer recount of c70e22cd on HP2 (claude/hp1-scorer 7dd19ff5):** estimates equal T317b to <= 2.2e-16;
    C 5,489 FPE; **F 6.76 [4.87, 8.79]**; HP1 clean 3.78. [measured by T281; unverified]
  - The family is frozen at the DRI's request (no further code changes). HP3 (T321) scores @1c742430 as frozen.

## T322 / T317d pre-registration (DRI, 2026-10-18; before any T317d run)
Close T320's three convention gaps, with no other change:
1. Precompute every model-free table once, outside est.
2. List the tables with their sizes here.
3. Replace the seeded model-free MC transverse points with their exact-law values.

**Gate:** estimates within **.01 pp** of T317c on all 80 DEV2, cost as executed. No HP3 submission until the DRI says
so, after T317c's HP3 score.

**T317c on HP3 (T281 scorer, claude/hp1-scorer 67e57f00; fresh panel, seeds 200-299):**
- c70e22cd as executed, 5,427 FPE: **F 5.08 [3.42, 7.04]**, RMS 0.81 pp, bias +0.42 pp;
- 1c742430 at the official count: 5.05;
- every-ndtr sensitivity: 5.29 [3.56, 7.33].

Lenient pass with the upper CI <= 10 under every count. [measured by T281; unverified]

## T317d method (`cont_d.py`, `estimator_d.py`)
- **(1) Tables.** `cont_d.TABLES` is built once at import. `est` only reads it: the walk, the merge, the Phi profile
  and all FLOP charges are cont_c's functions, called unchanged.
- **(3) Exact-law transverse points.** For the top 3 of 10 iid N(0,1), given v2 in the region (lo, hi]:
  - v2 has density 90 φ(v2) (1 − Φ(v2)) Φ(v2)^8;
  - E[v1 − v2 | v2] = φ/(1 − Φ) − v2, in closed form;
  - E[v2 − v3 | v2] = ∫_{−∞}^{v2} (Φ(v)/Φ(v2))^8 dv.

  These are integrated on a deterministic trapezoid grid (`exact_points`, 2 × 400,001 nodes, lo a node). The points
  agree with nested scipy `quad` to <= 2.5e-10 and with a 4x finer grid to <= 2.4e-10.
- **Points:** nb (v2*, g12*, g23*) = (1.013713, .533986, .347640); b = (−.136559, .851440, .129593). The old MC
  points were off by up to 2.2e-3 (b region).
- **v4 node:** still `r1w.gauss_v4(v3*, 1)`, a deterministic model-free 40,000-point rule, now evaluated once.
- `r1w2.REG` (the seeded MC) is still computed at import because cont.py imports r1w2. T317d never reads it except
  in the `points="mc"` equality check.

**Model-free tables (per region, G = 20; all float64, identical for every model, uncharged):**

| table | size per region | regions | role |
|---|---|---|---|
| GL nodes, weights | 20 + 20 | nb, b | v2 quadrature |
| Φ(nodes), Φ(nodes)^8 | 20 + 20 | nb, b | profile endpoints at exactly 0 |
| 1 − Φ(nodes) | 20 | nb, b | blind I12 (g12 assumed correct) |
| 90 φ(nodes) | 20 | nb, b | v2 density |
| transverse point (v2*, g12*, g23*) | 3 | nb, b | exact law (above) |
| v4 node, μ(v4), μ'(v4) | 3 | nb, b | closure c |
| line-23 inputs x, dx | 720 × 10 + 720 × 10 | nb, b | 720 triple lines along g23 |
| line-12 inputs x, dx | 720 × 10 + 720 × 10 | nb only | 720 triple lines along g12 |

Total: 266 scalars plus 43,200 input/direction entries. `cont_d.table_sizes()` prints this list.

## T317d results (`python run_d.py`, `results_d.json`, `run_d.log`; all 80 DEV2, ~1 CPU-min)
- **Refactor check:** cont_d with T317c's MC points equals cont_c exactly, in estimates and FLOPs (max diff 0.0 / 0.0).
- **Exact points vs T317c:** max |diff| **.0115 pp** (torch_s19); the next largest is .0076. The mean is −.0009 pp and
  the RMS .0035 pp. **One model of 80 exceeds .01 pp, so the pre-registered gate FAILS, narrowly.** T320's
  "<= .008 pp" was measured on 10 models.
- Cost as executed: C 5,490 FPE (T317c 5,488). The points move a few walk cells.
- DEV2 (development only): RMS .7944 pp vs .7950, bias +.444 pp both, F 4.78 both.
- Not frozen and not submitted. No variant tuning (forking paths); the gate result goes to the DRI for a ruling.
- **DRI ruling (2026-10-18): option (a), accepted with the overshoot noted.** The .0115 pp overshoot (vs .01 pp, one model)
  comes from a model-free change fixed in advance, which cannot be tuned to the data. **T317c's HP3 score remains the
  confirmed result (H-L1).** T317d is frozen at @5b4d4806 (`estimator_d.py` -> `cont_d.est`) and goes to the T281 scorer
  as HP3's second submission, labelled the "ARC-ready version". Its score is reported alongside T317c's.
- **T317d scored (T281 scorer, claude/hp1-scorer 29bef882; hashes match):**
  - HP3: C 5,431 FPE, **F 5.07 [3.41, 7.03]** (T317c 5.08 [3.42, 7.04]);
  - HP2: F 6.75 [4.86, 8.78];
  - max |est − T317c| is .0076 pp on HP3 and .0068 pp on HP2.

  [measured by T281; unverified]

## T317e: final cleanup, no Monte Carlo anywhere (DRI-authorised after the red-team review; `cont_e.py`, `estimator_e.py`)
- **What changed:** T317d's code path is now one self-contained file. Functions are copied verbatim from r0, r1w,
  walk, cont, cont_c and cont_d, so the arithmetic is unchanged. Two imports are removed:
  - the dead imports of `walk.py` and `r1w2.py`, which each built a seeded MC table at import (walk.TP, r1w2.REG)
    that was never read;
  - `scipy.stats`, which itself draws random numbers at import to build docstring examples. `norm.pdf` is replaced
    by scipy's own formula exp(-x²/2)/√(2π), bitwise equal on 2e6 points.
- **Imports:** only numpy, scipy.special.ndtr and itertools.
- **Gate (`python run_e.py`, `results_e.json`):**
  - A fresh subprocess patches numpy's random generators to raise, then imports estimator_e and calls est. No error
    is raised, and none of walk / r1w2 / r1w / cont / cont_c / cont_d / r0 / v1b2 / scipy.stats is loaded.
  - On all 80 DEV2, estimates **and** FLOPs are **bitwise identical** to T317d (max diff 0.0 / 0.0). C = 5,490 FPE as
    executed, unchanged.
- **Scorer identity check (T281, claude/hp1-scorer e8102588):** hashes match. Estimates and charged FLOPs are bitwise identical to T317d on all 100 HP3 and 100 HP2 models, so the scores are T317d's: HP3 F 5.07 [3.41, 7.03] at 5,431 FPE; HP2 6.75 [4.86, 8.78]. [measured by T281]
- **Frozen** (hashes in the commit message). Its HP2/HP3 scores are T317d's, since the estimates and costs are
  identical.

## Status
- T317, T317b, T317c: done and scored. T317c is the HP2 and HP3 candidate (unverified; T320 independent recount done).
- T317d: frozen @5b4d4806 after the DRI accepted the narrow gate overshoot; scored HP3 F 5.07, HP2 6.75.
- T317e: T317d with no MC anywhere, bitwise identical; frozen. Done.

## Compute accounting
- Charged per model, as executed, under Directive 8:
  - walks: tangent recomputes, roots, B2 prefix cells, correct_interval 229;
  - merge comparisons;
  - Φ at weight-dependent arguments, 20 each (+3 for the ^8);
  - the combination.
- Uncharged: model-free tables (listed above), indexing and np.unique keys.
- Development compute: each 80-model DEV2 run takes ~1-2 CPU-min. The family total is well under 1 CPU-hour.

## Claims
- [measured] T317c/T317d estimates are deterministic, with no random model inputs (deductive + declared
  approximations of R1W V1).
- [measured, T281 scorer; unverified] T317c: HP2 F 6.72 [4.84, 8.73] (as executed 6.76); HP3 F 5.08 [3.42, 7.04].
- [measured] T317d moves estimates by <= .0115 pp, at unchanged cost and DEV2 F.

## Alternatives not ruled out
- The candidate's margin rests on the R1W V1 approximations: closure c, a single v4 node, the blind g12 assumed
  correct, and the gap beyond tmax assumed correct. A panel with different retrain statistics could break them.
- The cost depends on the implementation: T320's independent count was +8.9% with a direction-testing root rule.

## Failure modes
- The noisy HP1 copies fail (F 39-228): the approximations are tuned to clean retrains.
- Models whose errors sit beyond tmax = 0.5, or in the blind region, are mis-estimated by construction.

## T326 (strict-bar lane A, cost): step 1, T317e's executed cost by component (`python cost_table.py`, `cost_table.log`, `results_cost_table.json`)
The tally is an instrumented copy of cont_e's walk. It reproduces cont_e.est's charged FLOPs exactly (max diff 0.0)
on all 80 DEV2 models. Mean C = 5,490 FPE.

| component (summed over the 3 lines) | FPE | share |
|---|---|---|
| later cells: tangent recompute (steps > s*) | 2,004 | 36.5% |
| B2 prefix tree (first-cell value + tangent + roots) | 1,729 | 31.5% |
| later cells: readout + correct set (229) | 588 | 10.7% |
| later cells: roots (steps >= s*) | 368 | 6.7% |
| later cells: values at t* + gate toggle | 346 | 6.3% |
| first-cell readout + correct set | 317 | 5.8% |
| Φ profiles | 115 | 2.1% |
| combination | 20 | 0.4% |
| merges | 3 | 0.1% |
| anchors, transverse points, closure (μ, v4 node), GL tables | 0 | model-free, precomputed |

| line | FPE | share | prefix nodes / 7,200 row-steps | later cells (720 rows) |
|---|---|---|---|---|
| nb g23 | 2,203 | 40.1% | 2,815 | 2,584 |
| nb g12 | 1,774 | 32.3% | 2,815 | 2,401 |
| b g23 | 1,495 | 27.2% | 2,815 | 1,117 |

Reading:
- Each line costs ~680 FPE before any gate flip: a 576-FPE prefix tree plus a 106-FPE first-cell readout.
- Later cells cost ~0.54 FPE each.
- The strict go line (F <= .8 at C <= 2k) needs RMS <= .54 pp at 2k, or <= .77 pp at 1k. So cost cuts alone cannot
  pass at T317e's RMS (.80 pp): even 1k FPE gives F .87. The cost lever has to be combined with an error cut, or
  reach C ≈ 1k while holding RMS <= .77.

## T326 step 2 pre-registration (2026-10-18; written before any frontier run)
**Knobs.** All are deductive, or declared approximations of the same kind as V1's "gap beyond tmax assumed correct".
They are shared across lines, except for the line subset.
- **tmax** ∈ {.5, .3, .2, .1}: the walk window. The gap beyond tmax is assumed correct, as in V1.
- **cap** ∈ {∞, 4, 2, 1, 0}: the number of later cells followed per row. When a row hits the cap, the rest of its
  window [t, tmax] uses the frozen last cell's affine logits (one correct_interval, 229 FLOPs, charged). cap = 0 means
  the first cell, frozen to tmax.
- **line subset** ∈ {all 3; drop nb-g12 (g12 assumed correct, as in the blind region); drop b-g23 (b g23 assumed
  correct); nb-g23 only}.
- Fixed: G = 20, closure c, exact-law transverse points (T317e).
- The grid has 4 × 5 × 4 = **80 configurations**. Each is evaluated on all 80 DEV2 with charged FLOPs as executed.
- Check: (tmax .5, cap ∞, all lines) must reproduce T317e bitwise.

**Selection rule.**
- Frontier: the Pareto set of (C, F) over the 80 configurations. Also reported: RMS, bias and the strict F, defined
  as RMS² · max(C, 1k) / .074.
- **Selected:** the configuration with minimum strict F among those with C <= 2,000 FPE.
- **Go:** strict F <= 0.8 on DEV2. If met: freeze, then ask the DRI for HP4.
- If not met: report the frontier and the shortfall. Margin or mass screens per triple, per-line knobs and closure
  order are then the untried variants (a follow-up only if the frontier comes within ~25% of the go line).

**Forking paths.**
- 80 configurations are selected on DEV2, so the selected DEV2 F is optimistic. A sealed HP4 is the only claim.
- T317e's DEV2 numbers (RMS .79, bias +.44 pp) were known before this grid was chosen.

## T326 step 2 results: DEV2 frontier (`python knobs.py`, `knobs.log`, `results_knobs.json`; ~3.5 CPU-min)
- Check: (tmax .5, cap ∞, all lines) reproduces T317e bitwise on all 80 DEV2 (estimates and FLOPs).
- Strict F = RMS² · max(C, 1k) / .074 (RMS as a fraction). T317e itself has strict F **4.68** at 5,490 FPE.

**Pareto frontier (C, strict F) over the 80 configurations:**

| C (FPE) | RMS pp | bias pp | strict F | lenient F | tmax | cap | lines |
|---|---|---|---|---|---|---|---|
| 740 | 3.92 | +3.10 | 20.8 | 15.7 | .1 | 0 | nb-g23 only |
| 742 | 3.13 | +1.38 | 13.2 | 10.0 | .2 | 0 | nb-g23 only |
| 2,868 | 1.72 | +1.24 | 11.4 | 11.6 | .2 | 4 | drop b-g23 |
| 3,115 | 1.54 | +.65 | 9.9 | 10.1 | .3 | 4 | drop b-g23 |
| 3,985 | 1.32 | +.66 | 9.4 | 9.6 | .2 | 4 | all |
| 4,667 | 1.03 | +.71 | 6.7 | 6.8 | .3 | ∞ | all |
| **5,490** | **.79** | **+.44** | **4.68** | **4.78** | .5 | ∞ | all (T317e) |

- **Selected (min strict F at C <= 2k):** tmax .2, cap 0, nb-g23 only: C 742, RMS 3.13 pp, **strict F 13.2**.
- **GO: NO.** The best point is 16x above the .8 go line, far outside the 25% band that would trigger the follow-up.
- Every cut costs far more error than it saves in cost: **no configuration beats T317e on either F.**
  - **Cap with a frozen extension:** at tmax .5 with all lines, RMS goes .79 → 2.31 (cap 4), 8.1 (cap 2), 14.5
    (cap 0) pp, with bias turning strongly negative. Gates flip often inside the window, so a frozen cell
    misclassifies.
  - **Dropping lines:** drop b-g23 → 1.35 pp, drop nb-g12 → 2.42, nb-g23 only → 3.07 pp. Each line carries real error
    mass.
  - **tmax:** .3 → 1.03 pp, .2 → 1.42 pp. The gap beyond tmax is not negligible below .5.
- **Structural floor:** with all 3 lines, even zero later cells costs ~2.2k FPE (3 × ~680 prefix tree + readout,
  plus Φ). C <= 2k therefore forces dropping a line, and the cheapest drop already costs RMS >= 1.35 pp. Strict at 2k
  needs <= .54 pp.

**Claim [measured, DEV2; negative (unverified)]:** within the pre-registered knob space (tmax, cap with frozen extension,
line subsets), the T317e family has no DEV2 point with strict F <= .8 at C <= 2k. The best is 13.2. Its cost cannot
be traded for error: T317e's own point is the frontier's best.

**Exhaustion checklist, variants NOT tried, and why:**
1. **Per-triple margin or mass screens** (ANAT-80's ~44 triples). They act only on later cells (~3.3k FPE). The
   prefix-tree floor (~2.2k with 3 lines) remains, so even a perfect screen leaves C ≈ 2.2-2.5k at RMS >= .79, i.e.
   strict F >= ~1.86 at best (corrected by the DRI; I first wrote 4.8). F is linear in C at fixed RMS:
   .0079² × 2,200 / .074 = 1.86. It cannot reach the .8 go line alone, but it is worth ~2.5x and is half of the strict
   route (see T330).
2. **Cap with "rest of the window assumed correct"** instead of a frozen extension. Its bias would go positive rather
   than negative, so it might be milder than the frozen cap. It is bounded below by the same ~2.2k floor, so the
   same argument applies.
3. **Per-line knobs.** The same floor argument applies: the prefix trees dominate below ~2.2k.
4. **Cheaper prefix tree (sharing across lines).** The keys differ (different transverse values and directions), so
   there is no exact sharing.
5. **Closure order:** that is lane B's (T327) error route.
6. **A bias correction** (+.44 pp of the .79 RMS): a DEV2-fitted offset is not deductive. Not done.

Lane A's conclusion: at T317e's error the strict bar needs C ≈ 1k, but this family's deductive cost floor is ~2.2k
with all lines. The strict route requires a structurally different, cheaper geometry, or the closure fix (T327) to
cut RMS to ~.5 pp at the current cost.

Forking paths: 80 configurations were selected on DEV2. The negative is robust to selection optimism, since that
optimism only makes the best point look better.

## T326, DRI follow-up: ORACLE floors of the two big items (`python floors.py`, `floors.log`, `results_floors.json`)
These are oracle lower bounds, not estimators: rows and cells are selected using the walk's own outcome. A row's
"deficit" on a line is the estimate with that line assumed correct, minus the actual estimate. A row "carries mass"
if its deficit > 1e-13. "Ideal cap" follows a row only up to its last cell that holds a wrong piece. Means over the
80 DEV2 models, in FPE:

| line | rows carrying error | prefix tree: all 720 → those rows (q = 100%) → 90% → 50% of deficit | later cells: full → ideal cap | of which tangents |
|---|---|---|---|---|
| nb g23 | 373 | 576 → 335 → 212 → 81 | 1,485 → 127 | 923 → 75 |
| nb g12 | 373 | 576 → 331 → 191 → 54 | 1,060 → 55 | 578 → 25 |
| b g23 | 565 | 576 → 465 → 381 → 211 | 762 → 519 | 504 → 344 |

- **At q = 100% with the ideal cap, the estimate is unchanged (exact).** Its cost is about 1,130 (prefix) + 700
  (later cells) + ~190 (first-cell readout of the kept rows) + ~115 (Φ) + 20, i.e. **≈ 2.2k FPE**. That is the floor
  for *any* screen that preserves T317e's estimates. At RMS .79 it gives strict F ≈ **1.86** (corrected by the DRI; I
  first wrote 6.3).
- **The two big items behave differently:**
  - **The tangent recompute is mostly unnecessary:** on the non-blind lines, 92-95% of later cells come *after* the
    row's last wrong piece. A deductive "row is now safely correct to tmax" screen could in principle recover most of
    the 2.0k FPE of tangents.
  - **The prefix tree is not:** restricting to the mass-carrying rows saves only ~40% on the non-blind lines and ~20%
    on b-g23. ~1.1k FPE of prefix tree remains even for an oracle.
- [inferred] Even a perfect screen leaves T317e's estimates at ~2.2k FPE, so strict F ≈ 1.86 at the current error
  (corrected; I first wrote 6.3). The cost lever is worth ~2.5x (5.5k → 2.2k). Combined with a bias fix to RMS ≈ .58 pp
  (T324), it would give F ≈ 1.0. That is the strict route's other half: T330.

## T326 note for T329: where the ~2.2k floor comes from (`python tree_note.py`, `results_tree_note.json`)
(DRI arithmetic correction: at 5.5k FPE, strict needs RMS <= .37 pp, not ~.5; .5 gives F ≈ 1.9. The T326 text above
that says "~.5 pp at the current cost" should be read with this correction.)

**The tree is pure combinatorics.**
- Per line, the B2 tree has **4, 13, 34, 73, 136, 229, 358, 528, 720, 720 distinct prefix states at steps 1-10 (2,815
  in total)**. The count is identical for all three lines and all models.
- It equals the number of distinct placements of the three labelled top values (1, 2, 3) in a length-s prefix. v4
  adds no split, because its position (the latest free one) is fixed by the triple. Lows add none either: they share
  one value.
- The input values never add splits.
- Steps 9-10 alone hold **1,440 nodes (51%)**, since every triple's prefix is distinct once 9 positions are fixed.
- Floor accounting:
  - prefix trees: 3 lines × 2,815 nodes × 1,218 FLOPs = **1,729 FPE**;
  - per-row first-cell readout + correct set: 3 × 720 × 873 FLOPs = **317 FPE**;
  - kept later cells (oracle ideal cap, ~700) + Φ (~115) bring the exact-estimate floor to ~2.2k (floors.py).

**Error mass is diffuse (oracle; the deficit is attributed per row as in floors.py, the node mass sums the rows
through it).**

| line | line deficit | nodes with mass > 1e-4 / 1e-5 / 1e-6 (abs. accuracy) | prune rows with deficit < 1e-4: nodes kept, deficit dropped | < 1e-5 | < 1e-6 |
|---|---|---|---|---|---|
| nb g23 | 4.66 pp | 843 / 1,514 / 1,622 | 802, **.86 pp** | 1,512, .016 pp | 1,622, .0002 pp |
| nb g12 | 1.65 pp | 300 / 1,127 / 1,557 | 233, **.78 pp** | 1,098, .059 pp | 1,556, .0008 pp |
| b g23 | .72 pp | 66 / 1,948 / 2,210 | 0, **.72 pp** (no row reaches 1e-4) | 1,928, .038 pp | 2,208, .0008 pp |

- "> 1e-4 of the error mass", read as absolute accuracy 1e-4 (.01 pp per row): 843 + 300 + 66 = 1,209 nodes carry it
  (≈ 250 FPE of tree). Pruning the rest, however, drops .72-.86 pp per line, more than the whole RMS budget.
- Read as relative (1e-4 of a ~1-5 pp line deficit ≈ 1e-6 absolute), ~1,550-2,200 nodes per line are needed. So the
  mass is spread over most of the tree: no small subtree carries it. b-g23 is the extreme case, with its .72 pp spread
  thinly over 565 rows.

**Homogeneity and linearity:**
- **Positive homogeneity** (no biases: h(c·x) = c·h(x), and likewise the tangent). This could share only the all-low
  prefixes across lines, since their values differ non-proportionally elsewhere. There are only **6 all-low nodes per
  line** (18 of 8,445), so it is negligible.
- **Linearity at fixed gates.** Within the first cell, a node's state is linear in (v1, v2, v3, v4, low), i.e. 5 basis
  vectors per gate history. Sharing across the 3 lines would need 5 basis matvecs to replace 3 line matvecs, which is
  worse. It pays only with >= 5 value-variants per role string: more lines, or the K > 1 v4 nodes or GL-in-v2 lines of
  a richer closure.
- **For T329:** the cost is set by the 720 × 3 distinct leaves × ~10 steps. A cheaper geometry has to reduce the
  number of *distinct inputs*, not share their computation. For example: fewer triples per line (merging positions
  with a deductive symmetry bound), fewer lines, or a backward / readout-side decomposition that does not enumerate
  full prefixes. Mass pruning cannot do it: the mass is diffuse.

## T330 pre-registration (DRI, 2026-10-18; written before any T330 code is run)
**Goal.** A deductive "safely correct to tmax" screen on T317e's walks. At the start of each cell (the first cell
included), a row may stop walking only if a rigorous bound proves that every point of its remaining segment
[t_c, tmax] is classified correctly, whatever gate flips happen there. The row then gets one correct piece
[t_c, tmax]. The screen must never cut a cell that holds a wrong piece.

**Bound family (sound by construction):**
- **Tier A (per model and line, row-independent; computed once):**
  - Box on the tangent dh_s over all gate patterns: b_0 = 0, b_s = |Wh| b_{s-1} + δ_s |Wi|, where δ_s = max over the
    line's rows of |d_s|.
  - Margin slopes: for target k and competitor j, |dm_kj/dt| <= β_kj = |Wo_k − Wo_j| · b_T at every t. This is valid
    across flips, because the logits are continuous and piecewise linear.
  - Safe iff min over j≠k of [m_kj(t_c) − (tmax − t_c) β_kj] > 0, with m_kj(t_c) = L_k(t_c) − L_j(t_c). A strict
    inequality makes numpy's first-index tie rule irrelevant.
- **Tier B (row-specific; used only if tier A fires too rarely to meet the go line):** the same interval propagation
  per row, from the row's own z(t_c), dz(t_c). A gate is fixed when |z_s,i(t_c)| > Δ · sup|dz_s,i| over the box, so
  that it cannot cross zero in the window, and free otherwise. This is sound by induction over the steps.
- Charges as executed:
  - tier A: ~12k FLOPs per line;
  - each tier-A check: 4C + 2 = 42 FLOPs;
  - a tier-B check: its interval matvecs (≈ 2 × 2H² per step), charged in full.
- Lines: the non-blind ones first (nb-g23, nb-g12). b-g23 is screened too only if it pays. Everything else is
  T317e's code unchanged.

**Go line (DRI):** max |est − T317e| <= **1e-4 pp** on all 80 DEV2, at **C <= 3.0k FPE** as executed.

**Checks:**
1. For every screened row, the per-row line profile equals T317e's to <= 1e-12.
2. An independent soundness audit runs T317e's full walk and asserts that no screened segment contains a wrong piece.

**Forking paths.** The screen is exact by construction, so the estimate cannot be tuned. Any variant iterated on DEV2
changes only the cost, which is deterministic. The cost is panel-dependent, though, so a sealed panel would see some
cost shrinkage. No sealed scoring here: HP4 is reserved for the combined attempt.

## T330 results: GO = NO (`screen.py`, `screen_b.py`, `screen_c.py`, `run_t330.py`, `results_t330.json`, `run_t330.log`)
- **Tier A** (row-independent slope box over all gate patterns) **fires on 0 rows.** Its bound β_kj ≈ 4e7-2e8: |Wh|
  compounds over 10 steps, so the box is useless.
- **Tier B** (row-specific interval propagation; a gate is fixed when it provably cannot cross 0 in the window; v2
  `screen_c` reuses the restart's exact tangents as the centre and propagates only a radius):
  - **sound and exact**: max |est − T317e| = **0.0 pp**, and per-row line profiles are identical (max diff 0.0) on
    all 80 DEV2;
  - it stops 85-95% of rows on the non-blind lines eventually.
  - **But it never pays.** Each try costs ~3.4k FLOPs, about a whole later cell, and succeeds ~25% of the time, mostly
    when Δ = tmax − t is small and few cells remain. Rows needed ~4 tries per stop, and the stops land late: on
    fused_s0's nb-g23, 2,025 cells were still walked, against the oracle's 297.

| schedule (80 DEV2) | C (FPE) | max \|d\| | go (<= 1e-4 pp at <= 3.0k) |
|---|---|---|---|
| T317e (no screen) | 5,490 | — | — |
| tier B, try at every flip | 7,787 | 0.0 | no |
| tier B, only when Δ <= .05, at most once per row (cheapest of 10 schedules tried on 4 models) | 5,525 | 0.0 | no |

The other schedules (on 4 models: Δ <= .5/.3/.2/.1/.05 × every flip/once) all cost more than the baseline.

**Claim [measured; negative (unverified)]:** a box/interval "safe to tmax" bound, row-independent or row-specific, is
exact but costs more than the cells it saves on T317e's lines. The deductive certificate is too loose at the window
sizes where saving would matter.

**Exhaustion checklist, variants NOT tried:**
1. **Zonotope / affine arithmetic** (track correlations between neurons, not a box). It is tighter but costs
   ~2H² × (number of noise symbols) per step, so it is unlikely to beat ~3.2k FLOPs per avoided cell.
2. **Per-gate-pattern operator norms** (‖D Wh‖ over reachable patterns) in place of the |Wh| box. This could be much
   tighter if few gates are free. Untested.
3. **An amortised start-of-walk check over the B2 prefix tree** (one check per shared prefix). It costs ~300 FPE per
   line, but only ~10% of rows pass at Δ = .5, so it would not pay.
4. **Sub-window certification** (prove [t, t + h] and hop). The state at t + h is unknown without walking, so it
   reduces to the walk itself.

The oracle slack (92-95% of non-blind later cells after the last wrong piece) is real, but a cheap deductive
certificate of it was not found.

## T330 tier C + T331 pre-registration (DRI, after Fable r5 D1/D2; 2026-10-18, before running either)
Source: review/fable_review_r5/README.md (internal review, DRI branch claude/project-handoff-priorities-zpzfm8).

**T330 tier C: a declared heuristic screen (its error is measured, not bounded).** At the walk start or a gate flip
(after the restart's exact tangent and dL), with Δ = tmax − t: stop the row if the current cell's affine margin is
positive over the whole remaining window, i.e. min over j≠k of [m_kj(t) + Δ · min(0, dm_kj)] > 0 (frozen gates). It
then gets a correct piece [t, tmax].
- **C1 (DRI wording):** the first such cell, the walk start included.
- **C2 (Fable's wording):** the same, but only after the row's first wrong piece has ended. Rows that have not yet
  had a wrong piece are not screened.
- Cost: 2C + 4C + 2 per try, charged. Tier A's firing rate is 0 rows (measured above).
- **Oracle pre-check (DRI):** the error mass the screen ignores is est_C − est_T317e per model.
  - Go if its RMS over the 80 DEV2 is <= .10 pp **and** the screen skips >= 70% of the post-last-wrong-piece later
    cells. That fraction is (later cells, T317e − later cells, screen) / (later cells, T317e − oracle ideal-cap
    cells), on the non-blind lines.
  - Stop if the RMS is >= .25 pp.
- **Tier C go line (DRI):** declared error <= .1 pp at C <= 2.5k FPE.
  - Note: even the oracle ideal cap on all three lines gives ≈ 5,490 − (1,358 + 1,005 + 243) ≈ 2.9k FPE, so C <= 2.5k
    is probably out of reach. Reported regardless.
- Lines: the non-blind ones, plus b-g23 as a second configuration.

**T331: tmax upward (Fable D1).** T317e (exact points, G = 20, closure c) at:
- (i) tmax 1;
- (ii) tmax 2;
- (iii) tmax .5 with a **zero-cost affine extension**: the last walked cell's correct set is solved on [t, ∞) instead of
  [t, tmax], in the same single correct_interval call. The beyond-tmax gap is then wrong past the frozen cell's root,
  instead of assumed correct.
- For information: (iv) tmax 1 + extension, (v) tmax 2 + extension.

All on 80 DEV2, charged as executed, with RMS and bias reported separately.
- **Go:** DEV2 MSE × C <= .80 × T317e's.
- **Stop:** C grows > 15%, or the bias moves by < .08 pp.
- Selection among (i)-(v): minimum MSE × C. These are 5 variants on DEV2, so the selected value is optimistic
  (forking paths). Any HP4 claim belongs to the combined attempt.

## T330 tier C results (`python run_tierc.py`, `results_tierc.json`, `run_tierc.log`; all 80 DEV2)
Declared error d = est − est_T317e, per model.

| config | C (FPE) | declared d: RMS / mean / max (pp) | fires on post-last-wrong cells (nb-g23 / nb-g12) | RMS vs truth | bias | lenient F | strict F |
|---|---|---|---|---|---|---|---|
| T317e | 5,490 | — | — | .794 | +.444 | 4.78 | 4.68 |
| **C1, non-blind lines** | **3,522** | **.076** / +.045 / .336 | **96.7% / 95.5%** | .827 | +.489 | **3.32** | 3.25 |
| C1, all 3 lines | 3,328 | .077 / +.045 / .336 | 96.7% / 95.5% | .827 | +.489 | **3.14** | 3.08 |
| C2, non-blind | 4,676 | .018 / +.007 / .088 | 47.1% / 47.0% | .800 | +.451 | 4.13 | 4.05 |
| C2, all 3 | 4,643 | .018 / +.007 / .088 | 47.1% / 47.0% | .800 | +.451 | 4.10 | 4.02 |

- **Oracle pre-check (DRI go: declared RMS <= .10 pp and fires on >= 70%): C1 = GO** (.076 pp, 96%). **C2 = no**: it is
  accurate (.018 pp) but fires on only 47%.
- **Tier C go line (declared <= .1 pp at C <= 2.5k): NO.** C1 reaches 3.3-3.5k, as the oracle floor predicted (~2.9k
  even for an ideal cap).
- **The declared error is one-signed (+.045 pp mean):** the screen ignores later islands of wrong mass, which adds
  to T317e's positive bias. MSE vs truth rises 8%, but C falls 36-39%, so **lenient F falls 4.78 → 3.14 (C1, all
  lines).**
- C1 is a declared, measured heuristic, not a deductive certificate. Its error on a new panel is not bounded: it is
  measured on DEV2 here, with max .34 pp on one model.
- [measured, DEV2, unverified] C1 is the best cost point of the T317 family so far: lenient F 3.14 on DEV2 (T317e
  4.78).

## T331 results: GO = NO (`python t331.py`, `results_t331.json`, `t331.log`; all 80 DEV2; T317e reproduced bitwise)

| variant | C (FPE) | C growth | RMS pp | bias pp | MSE × C / T317e's | stop rule |
|---|---|---|---|---|---|---|
| T317e (tmax .5) | 5,490 | — | .794 | +.444 | 1.00 | — |
| (i) tmax 1 | 8,329 | **+52%** | .731 | +.304 | 1.28 | C > +15% |
| (ii) tmax 2 | 12,416 | +126% | .696 | +.219 | 1.74 | C > +15% |
| (iii) tmax .5 + affine extension | 5,553 | +1.1% | .834 | **−.396** | 1.11 | — (bias moves −.84) |
| (iv) tmax 1 + extension | 8,388 | +53% | **.669** | **+.104** | 1.08 | C > +15% |
| (v) tmax 2 + extension | 12,463 | +127% | .693 | +.211 | 1.73 | C > +15% |

- **No variant meets the go line** (MSE × C <= .80 × T317e's).
- **The error moves as T302 and T324 said.**
  - tmax 1 cuts the bias by .14 pp and the MSE by 15%. With the extension it reaches the best RMS so far, .669 pp,
    with bias +.10.
  - **But the cells beyond .5 are not cheap in this implementation:** tmax 1 costs +52% (Fable projected +5-12%).
    Each row averages ~2.9 more later cells between .5 and 1.
- **The zero-cost extension overshoots.** At tmax .5 it flips the bias from +.44 to −.40 pp. The frozen last cell
  marks too much of the far gap wrong, because gates flip back beyond .5 (H-E15's coarse cells still flip). So it is
  not a free bias fix on its own.
- [inferred, not pre-registered; a follow-up candidate] **(iv) combined with tier C1** may be worth testing:
  - (iv) has the right error (RMS .669, bias +.10);
  - C1 removes ~96% of the post-last-wrong-piece cells, and the extra .5-1 cells are mostly such cells;
  - if C1's saving carries over, C ≈ 3.5-4.5k at RMS ≈ .67-.70 would give lenient F ≈ 2.4-3.

Forking paths: 5 variants on DEV2. The negative needs no selection correction.

## T330b pre-registration (DRI; 2026-10-18, before any T330b run): rigorous screen with per-gate-pattern norms
Tier B v2 (`screen_c`) plus three changes, all sound:
1. **Fixed-off fix:** a gate proven off over the window has dh = 0 = dh*, so its deviation is 0. v2 used ez there,
   which was sound but loose. v1 had it right.
2. **Norm track:** ν_s >= ||δdh_s||₂ over the window, with
   ν_s = min(||e_s||₂, ‖Wh[N_s, A_{s-1}]‖ · ν_{s-1} + ||dz*_s[F_s]||₂ + ...), where:
   - A_{s-1} is the set of non-fixed-off neurons at s−1 (δdh is 0 elsewhere);
   - N_s = fixed-on ∪ free at s;
   - F_s are the free gates;
   - ‖·‖ is the smaller of the Frobenius norm of that sub-matrix (from the precomputed Wh∘Wh, one H² matvec) and the
     global σ_max(Wh) (once per model).
3. **Per-neuron bound:** |δdz_s,i| <= min((|Wh| e_{s-1})_i, ‖Wh[i, A_{s-1}]‖₂ · ν_{s-1}). The row norms come from the
   same Wh∘Wh matvec.

All the extra work is charged: 2 more H² matvecs per step once the deviation is non-zero, plus H per step. Tried at
every flip, and with the cheapest tier-B schedule (Δ <= .05, once per row).

**Go (same as T330):** max |est − T317e| <= 1e-4 pp on all 80 DEV2 at C <= 3.0k FPE as executed.
**Stop:** if, on 4 models, it fires no earlier than v2 (same or fewer later cells skipped), report without the full run.

## T331+C1 combination pre-registration (DRI; 2026-10-18, before running)
**Variant "T317f-candidate":** T317e with three changes.
- **tmax 1.**
- **Zero-cost affine extension:** the last walked cell's correct set is solved on [t, ∞).
- **Tier C1 on all 3 lines:** stop a row at the walk start or at a flip when the frozen-gate margin is positive over
  [t, tmax]. The stopped row's piece is its current cell's correct set on [t, ∞), from one correct_interval call
  (229 FLOPs, charged). That set contains [t, tmax], since its margin is positive there.

Run on 80 DEV2, charged as executed. Reported separately: bias, RMS vs truth, C, and C1's declared error. The declared
error is (estimate) − (the same variant without C1), i.e. tmax 1 + extension, T331 (iv).
- **Go:** DEV2 MSE × C <= **.70** × T317e's.
- One variant, pre-registered, so there is no selection within this run. The idea came from T331 and tier C results on
  the same DEV2, so the forking-paths path is: T326 → T330 → T331 → this.

**Arithmetic note (DRI):** at C ≈ 4k, strict needs RMS <= .43 pp. That is below the closure-c floor (.463 pp,
T324/T304), even with every walk approximation removed. So strict hinges on the closure lane (T327). This stack is
the lenient-improvement track.

## T331+C1 combination results: GO = NO (`python run_combo.py`, `results_combo.json`, `run_combo.log`; 80 DEV2)

| variant | C (FPE) | RMS pp | bias pp | MSE × C / T317e's | C1 declared error vs (iv): RMS / mean / max |
|---|---|---|---|---|---|
| T317e | 5,490 | .794 | +.444 | 1.00 | — |
| (iv) tmax 1 + extension | 8,388 | .669 | +.104 | 1.08 | — |
| **combo (iv) + C1 on all lines** | 3,854 | 1.034 | **−.708** | **1.19** | .888 / **−.811** / 1.75 pp |

- **The go line (<= .70) is missed. The error comes from the combination's definition, as pre-registered:**
  - A C1-stopped row's piece is its *current* (early) cell's correct set on [t, ∞).
  - That cell's affine margin is positive on [t, tmax = 1], but its root beyond 1 comes far sooner than the root of
    the cell the full walk would reach at t ≈ 1. Far-gap mass is wrongly marked wrong: declared −.81 pp mean.
- So C1 and the extension do not compose as defined. A frozen early cell is a good proxy *inside* the window
  (C1's +.045 pp) but not beyond it.
- [speculative; not pre-registered] One alternative would give C1-stopped rows [t, tmax] correct and then the
  beyond-tmax gap assumed correct (no extension), keeping the extension only for rows walked to tmax. It mixes two
  gap rules, and its bias sign is unclear (the walked rows get the extension's negative push). Not run; it would need
  its own pre-registration.

## T330b results: GO = NO (`python run_t330b.py`, `results_t330b.json`, `run_t330b.log`; 80 DEV2)

| schedule | C (FPE) | max \|d\| | per-row profiles |
|---|---|---|---|
| T317e | 5,490 | — | — |
| v3 (norm track + fixed-off fix), every flip | 9,843 (v2: 7,787) | 0.0 | identical |
| v3, Δ <= .05, once per row | 5,561 (v2: 5,525) | 0.0 | identical |

- The per-gate-pattern norm track and the fixed-off fix are sound (exact on all 80) but barely tighter. On 4 models
  they skip 0.3-5% more later cells than v2, while each try costs ~2x. **No schedule beats the baseline.**
- Reason [inferred]: the binding looseness is not the |Wh| box versus the operator norm. It is the free gates, whose
  full |dz*| enters the deviation at Δ of .1-.5. Every bound in this family pays for that.
- Combined with T330's tiers A/B: **rigorous safe-to-tmax certificates of this kind do not pay on T317e** (negative,
  unverified).
  - Untried: zonotopes, judged too costly per step.
  - C1 (declared, .076 pp) remains the practical screen. Making its error rigorous would need a bound on the free
    gates' effect, which these tiers could not deliver cheaply.

## T335: T317f frozen, the HP4 submission (`cont_f.py`, `estimator_f.py`, `run_f.py`, `results_f.json`)
**What:** T317e plus screen C1 on all three lines, at tmax .5. This is exactly the configuration that gave DEV2
lenient F 3.14 at 3,328 FPE (T330 tier C, `C1_all`).
- One self-contained file. T317e's functions are copied verbatim; `walk_c1` and `est` are added.
- Imports: numpy, scipy.special.ndtr, itertools. **No Monte Carlo anywhere.**
- The model-free tables are T317e's, precomputed once at import (`TABLES`, listed above under T317d and printed by
  `table_sizes()`).
- Class: **sampling-free at runtime** (no random draws), a deductive structure plus declared approximations.

**Declared approximations:**
- R1W V1's, as audited by T320:
  - closure c (top-4 kept, with one Gauss node for v4 at the latest free position, moving with v3);
  - μ linearised;
  - blind g12 assumed correct;
  - the gap beyond tmax assumed correct;
  - product of line sets.

  The DRI counts these as six (T320's list itemises seven).
- **Plus C1:** a row stops walking when its current cell's frozen-gate margin is positive over [t, tmax], and the
  rest of its window is taken as correct. Measured on DEV2 against T317e: declared error RMS .077 pp, mean +.045 pp,
  max .34 pp. It is not bounded: later flips can create wrong islands.

**Gate (`python run_f.py`):**
- A fresh subprocess with numpy's RNGs patched to raise imports estimator_f and calls est. No error is raised, and
  none of the development modules or scipy.stats is loaded.
- On all 80 DEV2, estimates are **bitwise equal** to the T330 C1_all run (results_tierc.json) and to screen_c live.
- FLOPs equal screen_c's minus exactly 3 × (H² + 2C²H) = **1.74 FPE**. screen_c charged that for tier-B tables (|Wh|,
  |Wo_k − Wo_j|) that C1 mode never uses; cont_f does not compute them, so it charges as executed.
- **DEV2 (development):** C **3,326 FPE**, RMS .827 pp, bias +.489 pp, **lenient F 3.14**, strict F 3.08.

**Forking-paths note.** Everything below was evaluated on the same 80 DEV2 models before choosing T317f:
- T326: 80 knob configurations (tmax ↓, cap, line subsets).
- T330:
  - tier A;
  - tier B v1/v2 with 10 try schedules;
  - tier B with the norm track (T330b), 2 schedules;
  - tier C: C1/C2 × {non-blind, all lines} (4 configurations).
- T331: 5 variants (tmax 1/2, extension, both).
- The T331+C1 combination: 1.
- The choice of C1_all is the minimum-F point among the 4 tier-C configurations, and among all exact-or-declared
  variants. Its declared error is one-signed and was measured on these same models, so the DEV2 F is optimistic.
  **HP4 (sealed, single pre-registered submission) is the only claim.**

## T317f on sealed HP4 (T281 scorer, claude/hp1-scorer 84466121; hashes match; scored once)
- **HP4: C 3,302 FPE, F = 2.63 [1.93, 3.39]**, RMS .75 pp, bias +.44 pp. [measured by T281; unverified]
- Secondary (panels already spent): HP3 3.22 [2.19, 4.42]; HP2 4.41 [3.21, 5.66].
- **The scorer flagged my undercount.** safe() charges 4C + 2 for its margins but executes ~6C + 2 (m, dm, min(0, dm),
  × Δ, +, min). The L update of rows screened at a flip (2C) is also uncharged. Together that is <= +1.5% of C, giving
  **F ≈ 2.67**, the conservative figure. The frozen code is left unchanged.

## T337 pre-registration (DRI, owner's question; 2026-10-18, before any T337 run): T317f's noise breakdown curve
**Models (open only; no sealed truths).**
- Bases: the target M_16,10 and zoo seeds 0-4 (6 bases).
- Copies: W + σ · RMS(W) · N(0,1) per matrix (m16/arc_eval.py:noisy_models' formula), at relative σ ∈ {.0005, .001,
  .0025, .005}. 4 copies per base and level (96 noisy models, plus 6 clean). Seed default_rng(337000 + 100·base +
  10·level).
- **Truths:** 1e7 inputs per model, float32 forward in 250k chunks, MC SE <= .01 pp. float32 vs float64 agreement is
  checked on 1e6 inputs of 2 models.

**Estimator:** T317f exactly (estimator_f.est).

**Per level:** bias, RMS and lenient F = MSE · C / mean p(1−p) over the 24 copies, with the clean bases as level 0.

**Where the error lives (from the truth MC, all models):** the misclassified mass split by:
- (a) **in window**: g12 < tmax or g23 < tmax (tmax = .5, the region the lines cover) vs **interior** (both gaps >= .5;
  T317f treats this as correct);
- (b) the predicted rank (top-1, top-3, other);
- (c) the v2 sign (blind region v2 <= 0).

**Switch-offs** at the first level with F > 10, on 2 pre-chosen copies (copy 0 of the target and copy 0 of zoo
seed 0), one at a time from T317f:
- S1: C1 off (= T317e);
- S2: tmax .5 → 6 (removes "gap beyond tmax assumed correct");
- S3: blind g12 walked (removes "blind g12 assumed correct").

The line-set product and closure c cannot be switched off in this code within the budget. Their joint share is the
remainder.

**Hypotheses, written before running:**
- H1: the breakdown is interior error. Noise creates misclassifications far from both ties, which the lines never
  see, so the bias goes positive.
- H2: the breakdown is a closure failure. The low positions matter: errors predict a rank-4+ position.
- H3: it is the C1 screen. Wrong islands later in the window grow under noise.

The cheapest separation is (a) + (b) from the truth MC. S1 tests H3; S2 tests H1 within the line structure.

## T337 results: T317f's noise breakdown (`python noise.py truths|switch .0025`, `results_noise*.json`, `noise.log`)
Open models only: the target and zoo seeds 0-4, 4 copies per level, 1e7-sample truths. float32 and float64 predictions
differ on <= 4e-6 of identical inputs.

| rel σ | models | true acc mean (min) | C | bias pp | RMS pp | **F** | true error pp: in window / interior | errors predicting top-1 / top-3 / rank 4+ (pp) |
|---|---|---|---|---|---|---|---|---|
| 0 (clean bases) | 6 | 91.9 (88.0) | 3,365 | +.58 | .85 | **3.3** | 8.01 / .09 | 2.23 / 4.90 / .97 |
| .0005 | 24 | 91.3 (85.9) | 3,402 | +.58 | .85 | **3.1** | 8.58 / .10 | 2.52 / 5.15 / 1.01 |
| .001 | 24 | 89.4 (80.4) | 3,403 | +.75 | 1.04 | **4.0** | 10.40 / .18 | 3.60 / 5.66 / 1.32 |
| **.0025** | 24 | 80.2 (55.9) | 3,625 | +1.63 | 2.19 | **12.0** | 19.07 / .76 | 8.76 / 7.84 / 3.23 |
| .005 | 24 | 73.5 (43.5) | 3,725 | +2.94 | 3.86 | **32.4** | 25.26 / 1.25 | 11.50 / 9.59 / 5.42 |

- **The breakdown lies between rel σ .001 (F 4.0) and .0025 (F 12.0).** It is driven by how far the copy's accuracy
  falls. Per copy, the estimate's error correlates with the true error rate at r = .81-.84. Some bases are much more
  fragile: at .0025, the mean error is +.28 pp for the target but +3.3 pp for zoo1 and zoo2.
- **The bias is positive and comes from errors far from the tie faces:**
  - **Interior errors** (both gaps >= tmax = .5, which T317f treats as correct) grow from .09 pp clean to .76 pp at
    .0025: about half of that level's +1.63 pp bias. Per copy, they correlate with the estimate's error at r = .83-.84.
  - **Errors predicting a rank-4+ position** (the closure's lows) grow from .97 to 3.2 pp, r = .73-.89 with the error.

**Switch-offs at σ .0025, on 2 pre-chosen copies** (error vs truth, pp):

| copy | T317f | S1 C1 off | S2 tmax 6 | S3 blind g12 walked |
|---|---|---|---|---|
| target #0 | +.38 | +.36 | +.34 | +.37 |
| zoo0 #0 | +1.60 | +1.52 | **+.93** | +1.51 |

- **Gap beyond tmax (S2) breaks first:** it removes ~40% of zoo0's error. C1 (S1, −.08) and blind g12 (S3, −.09) are
  minor. The remaining +.93 pp belongs to the approximations not switched off here, the closure (lows at μ(v4)) and the
  line-set product. That fits the rank-4+ errors growing and the interior errors, which no line sees even at tmax 6.
- **Hypotheses:**
  - **H1 (far-from-tie errors) supported:** both the beyond-tmax gap and the truly interior mass grow with noise.
  - **H2 (closure/lows) partly supported:** rank-4+ errors grow and correlate. Its share is inferred from the
    remainder only, since the switch-off was not run.
  - **H3 (C1 screen) rejected:** C1 adds only ~.08 pp.
- **Claim [measured, open models; unverified]:** T317f holds F <= 4 up to rel σ .001, then fails between .001 and .0025.
  The failure is a positive bias from misclassified mass far from the tie faces: beyond tmax along the lines, in the
  interior, and in predictions of low-ranked positions. All of these are regions the R1W geometry assumes correct.
  Only 2 copies were switched off.

## T344: rigorous upper bounds on T340's rare events, a feasibility finding before building (2026-10-18)
**Events (T340's README, claude/t286-cheap-frontier):** E(t0) = {the network outputs the position of the 3rd-largest
input AND g23 > t0}, t0 ∈ {1, 1.25, 1.5}, on orig (the target), zoo seed 0 and zoo seed 3. T340's 2e7 magnitude scan
gives 2.5e-7 to 1.8e-4; its 1e9 truths are pending.

**What a rigorous upper bound must do.** Cover all of R^10 under N(0, I). Every region where E is not excluded is
charged its full Gaussian mass. The lows are 7 free coordinates, and no closure is allowed, since a closure is
itself a declared approximation.

**Probe 1: volume certificates by interval propagation.**
- Around 40 MC points per model that are correct and have g23 > 1, I searched for the largest uniform input box
  (half-width r in all 10 coordinates) on which interval propagation proves "the rank-3 position is not the argmax".
  The quantity is the margin L_j − L_p3 > 0 for the centre's argmax j.
- **Median certified r: <= 1e-4 on all three models** (p90 1e-4). |Wh| compounds over 10 steps (the same failure as
  T330 tier A).
- Covering the far-gap region (P(g23 > 1) = .044) to a 1e-6 remainder with 10-D boxes that small is out of reach by
  many orders of magnitude.
- [inferred, from T311] Linear relaxations (CROWN) are tighter, but certify only along 1-D lines at 2-3 FPE per line.
  For 10-D volumes the curse of dimension remains.

**Probe 2: exact linear-region (cone) enumeration.**
- The network has no biases, so every activation region is a polyhedral cone. The event inside one is an LP
  feasibility question, and g23 > t0 enters through the radial χ10 law.
- That would give an exact, rigorous method. But T218 measured the region tree's growth at **12-19x per step**
  (blind prefix), i.e. ~1e10-1e12 regions over 10 steps. Infeasible.

**The only rigorous bound computed so far: the trivial one, P(E(t0)) <= P(g23 > t0)** (exact, model-free):

| t0 | P(g23 > t0) | ratio to T340's scan: orig / seed0 / seed3 |
|---|---|---|
| 1.0 | 4.44e-2 | 10,100 / 247 / 278 |
| 1.25 | 1.69e-2 | 18,800 / 273 / 268 |
| 1.5 | 5.95e-3 | 23,800 / 297 / 259 |

**Conclusion [measured probes, inferred feasibility; negative (unverified), scoped].** A useful rigorous, declared-free
upper bound on these 1e-4..1e-6 events is not reachable with the exact-walk machinery in this budget:
- the walks resolve only 1-D lines, not volumes;
- the volume certificates available (interval propagation) certify ~1e-4 boxes;
- exact cone enumeration explodes.

**What could work, as follow-up candidates:**
- (a) A *conditional* certificate: rigorous given a declared closure of the lows. It reduces to 3-D (v2, g12, g23),
  where box or line certification is plausible. It is not declared-free.
- (b) Cone enumeration restricted by symmetry or pruning to the event's neighbourhood, if the growth there is much
  lower than T218's blind-prefix growth. Untested; the cheapest test is to count regions on the g23 > 1 sub-cone for
  the first 3-4 steps.
- (c) Certified bounds for smaller networks (H = 4-8), where the region count is manageable. This would say whether
  rigour at 1e-6 is possible at all for this architecture.

## T344 option (b) pre-registration (DRI; 2026-10-18, before counting): region growth in the far-gap sub-cone
- **Cone.** For a triple (p1, p2, p3), K = {x_p1 >= x_p2 >= x_p3 >= every low} ∩ the far-gap proxy
  {g23 >= 2 g12, g23 >= 2 (x_p3 − x_l) for every low l}. The second set says the 2-3 gap dominates. It is a
  conservative stand-in for g23 > t0, chosen to make K small and so to *favour* (b).
- **Count.** The exact activation regions (gate histories) and cells of the first k steps inside K ∩ [−1,1]^10, for
  k = 1..5, by LP cone splitting (T218's prefix_blind.py method; own copy `cone_count.py`).
- **Where.** The target (orig) and zoo seed 0, triples (0,1,2), (3,6,9), (9,5,1), with a cap of 2e5 cells.
- **Kill if** the mean per-step region growth over steps 2→5 is > 6x, **or** the projected step-10 total over the 720
  triples is > 1e6 regions. The projection is the median triple's count at k = 5 × (its mean growth)^5 × 720.
- If killed: option (a), a certificate rigorous *conditional on the declared closure*, in 3-D (v2, g12, g23).

## T344 results so far: option (b) killed; option (a) blocked at the planned method (`cone_count.py`, `cone_count.log`, `results_cone_count.json`)
**(b) Region growth in the far-gap sub-cone** (pre-registered kill line: mean growth > 6x over steps 2→5, or a
projected step-10 total > 1e6):

| triple (orig) | k = 1..5 regions | growth 2→3, 3→4, 4→5 |
|---|---|---|
| (0,1,2) | 2, 16, 113, 288, 630 | 7.1, 2.6, 2.2 |
| (3,6,9) | 2, 34, 792, 4,082, 13,067 | 23.3, 5.2, 3.2 |
| (9,5,1) | 2, 20, 212, 3,994, (stopped) | 10.6, 18.8, — |

- **KILLED.** Even the most favourable triple projects to 630 × 3.4^5 × 720 ≈ 2e8 regions at step 10, against the
  1e6 line. Growth slows once the top values have been read (2-3x per step) but never approaches 1.
- The run was stopped at seed0 (its own PID, checked) because the verdict was already clear.

**(a) Certificate conditional on the declared closure, in 3-D (v2, g12, g23).**
- **Interval propagation in 3-D:** around far-gap points (g23 in [1, 2]), the certified half-width is again
  **<= 1e-4** (median). The 3-D reduction does not help IBP: the compounding is in Wh, not in the input dimension.
- **The exact alternative is feasible in principle.** Under the closure, x is affine in (v2, g12, g23), so the
  network is piecewise linear in 3-D, and a fixed gate pattern is one convex polytope. The certificate would:
  1. enumerate the cells by grid probing;
  2. take each cell's polytope by half-space intersection;
  3. for each cell, run an event LP (argmax = p3 somewhere in it?), and charge an event-feasible cell's sub-polytope
     its volume × the max density over its bounding box;
  4. certify coverage by the volume sum, with any uncovered volume charged at the max density.
- **Measured cell counts** on the nb domain [0,6] × [0,6] × [1,6]: 21-126 / 34-162 / 65-210 distinct cells per triple
  at 20³ / 40³ / 80³ grids, so ~200-500 at saturation. That is ~4e5 cells per model.
  [inferred] ~0.5 CPU-h per model at ~5 ms per cell, **over this task's budget for 3 models.**
- **Caveat:** a closure-conditional bound is not a bound on the true probability. If far-gap rank-3 picks arise
  through the lows (T340's alternative 1), the closure removes them, and the conditional bound could sit *below* the
  truth.

**Feasibility statement [measured probes; negative (unverified), scoped].** On this architecture, rigorous volume
certificates by interval propagation fail at ~1e-4 box widths in 10-D and in 3-D alike, because |Wh| compounds over 10
recurrent steps (the same mechanism as T330 tier A). Exact enumeration is infeasible in 10-D, at ~2e8 regions per
event cone. In the closure's 3-D, exact enumeration is feasible but costs ~0.5 CPU-h per model, and the result would
only be conditional.

## T344 status: stopped, scoped feasibility negative (DRI ruling, option ii)
- A bound conditional on the closure is not the rigorous statement that gives certificates their value, and it
  could sit below the truth. Not run.
- **Final record [measured probes; negative (unverified), scoped to this architecture and these methods]:**
  - Interval certificates fail even in 3-D: certified half-width <= 1e-4. The compounding is in Wh, not in the input
    dimension.
  - Exact cone enumeration in the far-gap sub-cone grows too fast: the best triple projects to ~2e8 regions at step 10.
  - **Only the trivial bound is rigorous.** P(E(t0)) <= P(g23 > t0) = 4.4e-2 / 1.7e-2 / 5.9e-3. Against T340's orig
    1e9 truths (5.4e-6 / 1.14e-6 / 2.16e-7) that is **8,200 / 14,900 / 27,500x loose**.
- T344 released.

**Tightness of the trivial rigorous bound against T340's final truths** (T192 GPU float64: orig 1e10, seeds 1e9;
claude/t192-gpu-cost-law @63d36d38). The ratio is bound / truth:

| model | t0 = 1.0 | 1.25 | 1.5 |
|---|---|---|---|
| orig | 8,278x | 15,566x | 28,448x |
| seed0 | 250x | 274x | 302x |
| seed3 | 269x | 270x | 272x |

## T349 ("T340c") pre-registration (2026-10-18, before running; split agreed with T340's owner)
**Scope.** A sub-task of T340.
- T340 owns the events E1(t0) = {pick = p3, g23 > t0} and E2(t) = {wrong, g12 >= t, g23 >= t}, the truths (T192 GPU)
  and the verdict.
- T349 delivers estimates and C only.
- T340's code is used **read-only from a pinned checkout** (claude/t286-cheap-frontier @620eb47b, rare.py + cont_g.py),
  outside the repo tree.

**Step 1 (gate).** Reproduce T340's variant B bitwise (E1, E2 and FLOPs) on orig, seed0 and seed3 by calling
rare.est(variant="B").

**Step 2: T340c.** Identical to B except for g12. B pins g12 = E[g12 | v2] at each v2 node and weights it with the full
g12 mass. T340c instead integrates g12 with a **K-node Gauss rule for the exact law of g12 | v2**, whose density is
φ(v2 + g) / (1 − Φ(v2)) for g >= 0 (Stieltjes on a fine model-free grid, as r1w.gauss_v4).
- One line23 walk per (triple, v2 node, g12 node), with weight ω_k.
- For E2(t), the rule is for g12 | v2 restricted to g12 >= t, with total weight 1 − Φ(v2 + t), as in B. These are
  separate walks per t.
- Everything else is B unchanged: v3* = v2 − E[g23 | v2], v4 at its Gauss node, the v3-v4 link, μ(v4) linearised.
- **Primary: K = 3. Convergence check: K = 5.** K = 1 reproduces B's g12 for E1, up to the Stieltjes grid (~1e-10).
- Cost charged as executed with B's charges. It is about 3K × B's walks.

**Reported:** per (model, event), the estimate, the ratio to the truth and C, in est.json's shape for T340's table.
- Go (T340's own line): ratio within [1/2, 2] on all 9 E1 pairs.
- Hypothesis (T340's diagnostic): orig's ratio rises from .20-.29 toward 1.
- Alternative: g12 is not the cause, and the ratio stays low. That would point to the fixed v3-v4 link or the lows
  (T340's alternative 1).

## T349 (T340c) results (`python t340c.py gate|run K`, `score_t340c.py`, `results_t340c_*.json`, `t340c.log`)
**Gate:** T340's variant B reproduces **bitwise** (E1, E2 and FLOPs) on orig, seed0 and seed3, from the pinned
checkout @620eb47b. K = 1 equals B on E1 for every model, as expected (one Gauss node is the conditional mean).

**Estimate / truth** (T340's T192 GPU truths, float64: orig 1e10, seeds 1e9):

| model | variant | E1(0) | E1(1.0) | E1(1.25) | E1(1.5) | E2(.5) | E2(.75) | C (FPE) |
|---|---|---|---|---|---|---|---|---|
| orig | B (T340) | .95 | **.27** | **.20** | **.29** | 2.31 | **10.95** | 320,784 |
| orig | **K = 3 (primary)** | 1.03 | 1.73 | 1.24 | .49 | 1.30 | 1.61 | 2,738,050 |
| orig | K = 5 (check) | 1.05 | 1.58 | 1.02 | .84 | 1.29 | 1.65 | 4,528,646 |
| seed0 | B | 1.00 | .89 | .86 | .65 | 1.38 | 2.06 | 209,565 |
| seed0 | K = 3 | 1.08 | 1.24 | 1.27 | 1.10 | 1.20 | 1.52 | 1,828,935 |
| seed0 | K = 5 | 1.10 | 1.25 | 1.23 | 1.02 | 1.21 | 1.53 | 3,034,830 |
| seed3 | B | .93 | .69 | .73 | .64 | 1.89 | 5.00 | 292,841 |
| seed3 | K = 3 | .98 | 1.04 | 1.06 | 1.02 | 1.08 | 1.41 | 2,475,913 |
| seed3 | K = 5 | .99 | 1.00 | .98 | .92 | 1.12 | 1.41 | 4,094,011 |

**Verdict against T340's pre-registered line** (E1, 9 pairs):
- B: 6/9 within [1/2, 2], and the same 6/9 within [1/3, 3]. That meets T340's "partial" rule (>= 6 of 9 within
  [1/3, 3]), as T340's owner confirmed. My first write-up here wrongly said it fell short.
- **K = 3 (primary): 8/9 within [1/2, 2] and 9/9 within [1/3, 3], i.e. "partial".** orig E1(1.5) is .49.
- K = 5: **9/9 within [1/2, 2]**, which would be "useful". But K = 5 was the convergence check, not the primary.
  orig E1(1.0) is still moving (1.73 → 1.58), so K is not converged on orig.

**Diagnosis confirmed.**
- T340's diagnostic was that the transverse g12 point, not the closure, causes orig's 3.5-5x underestimate. g12
  integration lifts orig from .20-.29 to .84-1.58.
- **B's E2 overshoot (2-11x)** comes from evaluating the line at E[g12 | v2] while weighting it with g12 >= t. Putting
  the g12 nodes on the restricted law fixes most of it already at K = 1 (.95-1.47). A residual one-signed +12-65%
  remains, rising with t.

**But the cost: F vs sampling at equal compute.** F = ((ratio − 1) p)² C / (p(1 − p)); sampling has F ≈ 1. At C = 2-4.5M
FPE, sampling would itself see p·C ≈ 1-1,300 events.
- F < 1 only where p·C <= ~90: orig E1(1.25 / 1.5), and seed0 and seed3 E1(1.5). There it is .00-.58.
- Elsewhere F = 5-58, i.e. worse than sampling.
- So the transverse integration makes the estimate *accurate*, but at ~9-14x B's cost it beats sampling only for the
  rarest events. **The rare-event advantage needs the cost per line to drop, or p to be far smaller.**

Claims [measured, open models, against T340's truths; unverified]:
- g12 transverse integration fixes T340's orig underestimate and E2 overshoot. K = 3 is "partial", K = 5 is 9/9
  within a factor of 2.
- At these costs the method beats sampling at equal compute only on events with p·C ≲ 100.

## T354 pre-registration (DRI; 2026-10-18, before running): walk only the event's segment
**Change to T340c (T349)**, nothing else changed. Each line's walk starts at the event's own threshold, from an
exact restart state, instead of at the tie:
- **E1** (g23 > t0, t0 ∈ {1, 1.25, 1.5}): one walk per row from t = 1.0 to TBIG = 6, covering all three t0.
- **E2(t)** (t ∈ {.5, .75}): each family's walk starts at t.
- **The restart state is exact.** A fresh first cell (prefix tree, values and tangents) is computed at the start
  point x0 + t_start · d, and the walk proceeds as before on t' = t − t_start, with the pieces shifted back by t_start.
  The cells beyond t_start are the same cells the full walk visits, so the estimates should equal K3's and K5's up
  to rounding.
- **Not reproduced:** the E1(0) calibration row, which needs the whole line.
- **Screen for triples that cannot reach the event:** a cheap deductive certificate over Δ = 5 is not expected to
  work (T330's tier B was too loose even at Δ <= .5). So none is used, unless a check on one model shows a rigorous
  screen that fires. Reported either way.
- Charges as executed: T340's walk3 charges, unchanged.

**Pre-registered checks:**
- Estimates within 5% of K3's and K5's on all 15 (model, event) pairs (expected ~1e-12).
- **Go: C <= 1/5 of K3's** at unchanged accuracy.

Reported: est/truth, C and the equal-compute F per (model, event), handed to T340's table.

**T340's scoring of T349 (claude/t286-cheap-frontier @643c9fbf, rare_events/README.md "T340c"):** my numbers reproduce
exactly. For K3: E1 partial (8/9 within 2x; the miss is orig t0 = 1.5 at .49), **E2 useful (6/6 within 2x)**.

## T354 bug note: simultaneous gate flips in the continuity walk (`walkfix.py`)
**Found while gating T354.** The segment walk's estimates differed from T349's by up to 37% (E2) where they should
have matched to rounding. A dense check (52-60k forwards per line on 200-300 random rows) showed that **the walk itself
was wrong**:
- **Original walk** (T340's walk3, which descends from my cont.walk_cont): per-row correct length off by up to 5.3 (orig)
  and 5.5 (seed0), mean .13-.36.
- **Fixed walk:** <= 1e-4 (grid resolution).

**Cause.** At a flip t*, the walk toggles only the argmin gate (s*, j*). When several pre-activations hit 0 at the same
t*, the others have roots exactly at t*. The strict "root > t*" filter drops them, and their gates are never flipped.
- The main case: every step-0 neuron (z = x_0 Wi) crosses 0 together when the input coordinate x_0 crosses 0. The
  same holds for any step s whose input coordinate crosses 0 while h_{s−1} = 0.
- Along T340's per-node lines (t to 6, v2 nodes near 0), input coordinates such as v3 = v2 − t do cross 0. The 200
  validation rows needed ~4,700-5,200 extra simultaneous flips.

**Fix.** At each restart, re-gate every step s >= s* from the values: gate = z > 0 if |z| > tol, otherwise dz > 0
(tol = 1e-9 (1 + max|z_s|)). Tangents are then recomputed from s* with the new gates, roots from s* as before, and
near-zero pre-activations are excluded from the roots. Charged: + 2H per re-gated step.

**Scope.**
- **The T317 lineage (T317c/d/e/f, tmax .5) is unaffected.** On 10 DEV2 models, its lines have **0** simultaneous flips,
  and its pieces are identical under the fixed walk. That fits with their exact equality to V1's fresh forwards: no
  input coordinate crosses 0 inside their windows.
- **Affected:** T340's variant A/B and my T349 (T340c K1/K3/K5) and T354 numbers. They are being rerun with the fix
  (`run_fixed.py`). The T349 results section above is superseded by the fixed rerun below.

**T317f verified unaffected (`python check_t317f.py`, `results_check_t317f.json`).**
- (a) Model-free: no input coordinate of any T317f line crosses 0 inside [0, tmax = .5] (0 rows on all three lines).
  The nearest crossing is at t = .751, on nb-g23.
- (b) On 10 DEV2 models and 10 open DEV3 models (s500-s509), T317f's walk equals a dense float64 forward along every
  line (2,000 midpoints on [0, .5]): max |correct-length difference| **2.6e-4**, the grid's own resolution
  (dt 2.5e-4). The fixed walk needs **0** simultaneous flips there.
- So HP4's T317f score is not touched by this bug. The DEV3 models stand in for the sealed HP panels: the zero-crossing
  result is model-free, and (b) shows the walk resolving correctly on unseen open models.

## T354 results, and the fixed-walk reruns of B / T340c (`run_fixed.py`, `score_fixed.py`, `results_fixed_*.json`)
All rows use the fixed walk. Each cell is estimate / truth, with F vs sampling at equal compute in brackets
(F = (est − p)² · max(C, 1k) / (p(1 − p)); sampling has F ≈ 1).

| model | variant | C (FPE) | E1(1.0) | E1(1.25) | E1(1.5) | E2(.5) | E2(.75) |
|---|---|---|---|---|---|---|---|
| orig | B (fixed) | 297k | .25 (.89) | .17 (.22) | .19 (.04) | .99 (.01) | .95 (.00) |
| orig | **K3** | 2.55M | 1.77 (8.2) | 1.24 (.15) | .48 (.14) | 1.03 (.19) | 1.01 (.00) |
| orig | K5 | 4.22M | 1.60 (8.2) | 1.01 (.00) | .78 (.04) | 1.02 (.14) | 1.03 (.02) |
| orig | **S3 (T354)** | 1.97M | 1.77 (6.3) | 1.24 (.12) | .48 (.11) | 1.03 (.15) | 1.01 (.00) |
| seed0 | B (fixed) | 201k | .97 (.02) | 1.03 (.01) | .93 (.02) | 1.07 (.45) | 1.22 (.63) |
| seed0 | K3 | 1.76M | 1.32 (32) | 1.44 (21) | 1.43 (6.3) | .97 (.59) | .98 (.05) |
| seed0 | K5 | 2.92M | 1.34 (59) | 1.40 (29) | 1.33 (6.4) | .98 (.45) | .99 (.02) |
| seed0 | S3 | 1.40M | 1.32 (26) | 1.44 (17) | 1.43 (5.0) | .97 (.47) | .98 (.04) |
| seed3 | B (fixed) | 265k | .69 (4.1) | .76 (.92) | .73 (.42) | 1.40 (7.4) | 3.40 (22) |
| seed3 | K3 | 2.24M | 1.05 (1.0) | 1.10 (1.3) | 1.10 (.51) | .73 (27) | .55 (6.3) |
| seed3 | K5 | 3.70M | 1.02 (.15) | 1.01 (.02) | 1.01 (.00) | .78 (30) | .55 (11) |
| seed3 | S3 | 1.76M | 1.05 (.82) | 1.10 (1.0) | 1.10 (.40) | .73 (22) | .55 (5.0) |

Pass counts:
- **Fixed K3:** E1 8/9 within 2x, 9/9 within 3x; E2 6/6 within 2x.
- **Fixed K5:** E1 9/9 within 2x; E2 6/6.
- **Fixed B:** E1 6/9 within 2x and 6/9 within 3x; E2 5/6.

**T354 verdict: the go line is missed.**
- The segment walk is exact: **S3 = fixed K3 to every printed digit.**
- But **C falls only to 0.77-0.80 × K3's**, against the go line of 1/5. Most of each line's cells lie *beyond* t0,
  across the coarse far region up to t = 6. The dense near-tie cells skipped are only ~20-23% of the cost.
- S5's result will follow; it is expected to be the same ratio vs K5.

**What the fix changed:**
- **The bug mainly inflated E2.** Fixed K3/K5 E2 is ~1.0 on orig and seed0 (was 1.2-1.65). On seed3 it is now *low*
  (.55-.78).
- E1 hardly moved on orig and seed3. It moved up somewhat on seed0 (1.1-1.27 → 1.32-1.44).
- **Fixed B is far better than the buggy B:** E2 .95-1.40 except seed3 E2(.75) at 3.4; E1 on seed0 is 0.93-1.03.
  B's remaining failure is orig E1 (.17-.25) and seed3. **At equal compute B wins (F < 1) on 10 of 15 pairs**, because
  it is 7-14x cheaper than K3.

**Equal-compute reading:** the transverse integration buys accuracy where B is badly biased (orig E1), but its cost
loses on F almost everywhere else.

**A cheaper next step (not pre-registered; for the DRI):** end each walk at T_hi. The dropped mass is rigorously
bounded by P(g23 > T_hi), which is model-free: 2.0e-6 at 3, 7.0e-8 at 3.5, **1.7e-9 at 4** (< 1% of the rarest event,
orig E1(1.5) = 2.1e-7). The saving depends on the cell share beyond t = 4, which is unmeasured.

## T359 pre-registration (DRI; 2026-10-18, before the full runs): a T_hi cut with a rigorous model-free bound
**Probe (orig only, already run; disclosed):** fixed-walk S3 with tmax = 4.
- C 1.464M FPE vs 1.971M with tmax 6, i.e. **26% of the cost lies beyond t = 4.** That is 0.575x K3's 2.547M.
- Estimates move by < 1e-4 relative (E1(1.0): 9.5098e-6 vs 9.5098e-6).

**Arms:** S3 (fixed walk, K = 3, segment start) with the walk ended at T_hi. The dropped mass of every event is bounded
by P(g23 > T_hi) (model-free, exact quadrature), which is **added as a declared upper bound** (reported, not added to
the estimate).
- **(a) T_hi = 4:** bound 1.67e-9, < 1% of every event's truth.
- **(b) T_hi = 3.5:** bound 7.04e-8. That is 1.3% of orig E1(1.0), 6.5% of E1(1.25) and 34% of E1(1.5), and smaller
  elsewhere.

Run on orig, seed0 and seed3.

**Go (DRI):** every estimate moves by <= its bound vs the T_hi = 6 walk, **and C <= .5 x K3's**. The primary
accuracy criterion is est/truth within 2x.
**Expected from the probe:** (a) gives ~.57x K3 (a miss); (b) is near .5x.
