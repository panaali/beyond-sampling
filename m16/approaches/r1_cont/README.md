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
