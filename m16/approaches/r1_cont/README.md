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
