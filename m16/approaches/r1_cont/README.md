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
