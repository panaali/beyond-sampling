# T350 (first filed as T349): independent check of T340's rare-event result on fresh DEV3 models

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · T340 (claude/t286-cheap-frontier 620eb47b, rare_events/)

## Question
T340 found that T317's exact walks, uncut, estimate rare error probabilities (1e-4 to 1e-7) better than sampling at
equal compute in about 12 of 15 (model, event) cases on 3 models:
- variant B for the far-gap rank-3 islands E1;
- variant A for the interior error set E2.

Does this hold on ≥ 6 fresh models, with an independent implementation? I use my own walk (`r1w_repro/r1w_indep.py`,
T312) and have not read T340's code (`rare.py` etc.), only its README.

## Pre-registration (2026-10-18, before any DEV3 model or truth exists)
**Events** (as T340 defines them; v1 > v2 > v3 the top 3, g12 = v1 − v2, g23 = v2 − v3):
- **E1(t0)** = {the network outputs the position of the 3rd-largest input} ∩ {g23 > t0}, for t0 ∈ {1, 1.25, 1.5}.
- **E2(t)** = {the network is wrong (any pick)} ∩ {g12 ≥ t} ∩ {g23 ≥ t}, for t ∈ {.5, .75}.

**Estimators**, my derivation from T340's README. Both are deductive, with cost charged as executed.
- Common to both:
  - The line-23 inputs follow T317's closure: v1 = v2* + g12*, v2*, v3 = v2* − t, v4 = v3 − δ at the latest free
    position, and lows at μ(v4*) + μ'(v4*)(v4 − v4*).
  - My exact segment walk (T312) runs with **no tmax cut**, to t = 6, with no screen.
  - Each cell's affine logits give the exact interval where a given position is the argmax.
  - The masses use the exact model-free law: P = Σ_triples ∫ dv2 φ(v2) · M12(v2) · Σ_intervals
    [Φ(v2 − a)^8 − Φ(v2 − b)^8]/8. Both regions (nb: v2 ∈ (0, 6.5], b: [−6.5, 0]) use 20 Gauss-Legendre nodes each.
- **E1 by variant B.** One walk per triple per GL node, with node-specific transverse points:
  - g12 = E[v1 − v2 | v2], in closed form;
  - v3* = v2 − E[v2 − v3 | v2], by quadrature;
  - v4* = E[max of 7 below v3*], with δ = v3* − v4*.

  Intervals are where the argmax is p3, restricted to t > t0. M12 = 1 − Φ(v2), the full g12 mass.
- **E2 by variant A.** One walk per triple per region, at T317's region transverse point (the T317e TABLES constants).
  Intervals are the complement in [t, 6] of the set where p2 is the argmax. M12 = 1 − Φ(v2 + t) (g12 ≥ t;
  independence declared).
- **Cost (FPE)** is charged as executed:
  - walk cells as in T312: 2 FPE for the first cell, 2(T − s)/T for a restart at step s;
  - plus 20 FLOPs/5,948 per Φ evaluation at a weight-dependent point (interval endpoints × G nodes).
  - Model-free tables are uncharged.

**Metric**, per (model, event) case, with truth p and my executed cost C:
- relative error r = |est/p − 1|;
- the competitor's relative error s = min(1, 1/sqrt(pC)). Sampling at C forward passes has relative sd 1/sqrt(pC);
  when pC < 1, the zero answer (relative error 1) is the better competitor.
- **The case is won if r < s.**
- The truth's own MC error is propagated as a sensitivity: the case is "won robustly" if it is won at both ends of
  the truth's ±2 SE.

**Models.** The **first 6 models in DEV3's manifest order** (T342, built by the T192 child), whatever their names.
**Truths:** GPU MC with ≥ 1e9 inputs per model, all five events counted at once (requested from T192).

**Go:** won in ≥ 2/3 of the 30 cases (≥ 20). Also reported: per event family (E1: 18 cases, E2: 12), and the
ratio est/truth against T340's bands (useful within 2x, partial within 3x).

**Calibration before DEV3 (not part of the verdict):** I run the same code on T340's three models (orig, seed0,
seed3; T192's weights and GPU truths at 63d36d38, t340/) and compare my est/truth ratios with T340's table. Agreement
within about 20% would show that the two implementations match. A disagreement is reported, and the DEV3 run uses my
code as pre-registered either way.

**Alternatives first.**
1. T340's wins come from three models with a favourable transverse placement. On fresh models the far-gap islands
   (E1) may sit elsewhere, and variant B loses more often than on the seeds.
2. The metric favours estimators that are cheap relative to 1/p. If my implementation is costlier than T340's, it
   wins less often for the same accuracy. C is reported per case so that this can be separated.

## Amendment (DRI, 2026-10-18 13:47; before any DEV3 model or truth exists)
- **ID.** The task is **T350**. T349 was taken by T340c. The commits 606759a1 and b0eeade8 say T349.
- **Scoring (replaces the metric above):**
  - sampling's relative error is s = sqrt((1 − p)/(pC)) if pC ≥ 1, and s = 1 (the zero answer) if pC < 1;
  - ours is r = |est/p − 1|;
  - win if r < s − .05, loss if r > s + .05, tie otherwise. Ties are reported separately.
- **Variants scored.**
  - Each fixed variant (A, B) on all 30 DEV3 cases.
  - The **family rule fixed in advance**: B for E1 and A for E2.
  - For this, `rare.py` gains `est_A` (A for E1 too) and `est_B` (B for E2 too). The arithmetic of the original
    estimators is unchanged.
- **Go: the family rule wins ≥ 2/3 of the non-tied DEV3 cases.**
- **Models and truths.** As T192 confirmed: dev3_s500..dev3_s505, 1e9 float64 inputs each, all events in one pass,
  pushed under t349/ on claude/t192-gpu-cost-law. DEV3's own truths.json will be open on that branch. I will not read
  it, or the t349 counts, before my DEV3 estimates are committed.
- T340's handover message gave its pre-registration (c0564f03 + 716ea8b5) and its results at f9fae86e. I have still
  not read `rare.py` on its branch.

## Calibration on T340's 3 models (done before any DEV3 truth; `python3 calib.py`, `results/calib.json`)
These are my estimators against T192's GPU truths (t340/rare_mc.json @63d36d38). Each cell is est/truth, mine against
T340's:

| model | E1 t0 = 1 / 1.25 / 1.5 (variant B) | E1 t0 = 0 (calibration) | E2 t = .5 / .75 (variant A) | C, E1 / E2 (FPE) |
|---|---|---|---|---|
| orig | .25 / .17 / .19 (T340 .27 / .20 / .29) | .952 (.953) | .96 / .94 (T340 1.11 / 2.25) | 448k / 21k |
| seed0 | .97 / 1.03 / .93 (.89 / .86 / .65) | .997 (1.000) | .82 / .71 (.82 / .71) | 307k / 14k |
| seed3 | .69 / .76 / .73 (.69 / .73 / .64) | .924 (.925) | .59 / 1.23 (.61 / 1.37) | 399k / 19k |

- The two implementations agree closely on the non-rare calibration row (to .003). They agree on seed0's E2 exactly,
  and on most rare cells within about 20%.
- They differ on orig E2 at t = .75 (mine .94, T340's 2.25), on orig E1 at t0 = 1.5 (.19 against .29), and on seed0's
  E1 (mine closer to 1).
- At these probabilities the estimates are sensitive to small implementation choices. My v4* node is an exact
  quadrature; T340's is a grid Gauss rule. Thin islands can also be handled differently near a cell edge. I have
  not read T340's code to locate the difference.
- **Calibration score by the pre-registered metric: 12/15 cases won**, the same count as T340.
- **A note on the metric.** When pC < 1, a case is won whenever 0 < est < 2p, which is generous. Six of the 15
  calibration cases are of that kind. Cases with pC ≥ 1 are reported separately on DEV3.

**Calibration under the amended scoring** (`python3 run_all.py <t340 weights> results/calib_est.json`, then
`python3 score.py results/calib_est.json <t340 rare_mc.json> results/calib_score.json`), on 15 cases:

| scored | win | tie | loss | win share of non-tied |
|---|---|---|---|---|
| fixed A | 6 | 6 | 3 | .67 |
| fixed B | 8 | 3 | 4 | .67 |
| **family rule (B for E1, A for E2)** | **12** | 1 | 2 | **.86** |

For comparison, T286's scoring of T340's own estimates gave fixed variants 5/15 each, and the family rule 10 wins, 3
ties, 2 losses.

## DEV3 result
**Order of steps.**
- Estimates frozen: b736a5d0 (`results/dev3_est.json`, sha256 e4d20068…7b50), before any DEV3 truth was read.
- Truths: T192 GPU MC, claude/t192-gpu-cost-law @12e28ebb, t349/rare_mc.json, 1e9 float64 inputs per model.
- Scoring: `python3 score.py results/dev3_est.json <t349 rare_mc.json> results/dev3_score.json` (log in
  `logs/dev3_score.log`).

**Verdict: NO GO.** The family rule wins **11**, ties 6 and loses **13** of 30 cases: .46 of the non-tied cases, against
the .67 line.

| scored | win | tie | loss | win share of non-tied |
|---|---|---|---|---|
| fixed A | 10 | 10 | 10 | .50 |
| fixed B | 4 | 6 | 20 | .17 |
| **family rule** | **11** | 6 | **13** | **.46** |
| family, E1 (B) | 4 | 3 | 11 | .27 |
| family, E2 (A) | 7 | 3 | 2 | .78 |
| family, cases with pC ≥ 1 | 9 | 6 | 12 | .43 |

**What breaks.**
- **E1 (variant B) loses on cost, not because it is unusable as an estimate.** Its est/truth on the far-gap islands
  is .50–1.14 on five models, and 14/18 cases are within 2x.
  - But B costs 0.3–0.45M FPE. At that cost sampling sees 15–160 events, so its relative error is 8–25%. B's 20–50%
    bias then loses.
  - On dev3_s505, B is 6–10x low (ratio .11–.16). Claim (i) of T340, "B within ~1.6x on retrains", fails on 1 of 6
    fresh models (and is .50–.53 on dev3_s503).
- **E2 (variant A) holds up.** Its ratios are .21–2.14, with 8/12 within 2x and 11/12 within 3x. It wins 7 of 9
  non-tied cases at 14–21k FPE.
  - Claim (ii) of T340, "A within ~2x on the interior error set", holds on 8/12 cases. The misses are on dev3_s505
    (.35, .21) and dev3_s500 at t = .75 (2.14).
- **Calibration row (E1 at t0 = 0, not rare).** B's ratios are .84–.98, against .92–1.00 on T340's models.

**Comparison with the calibration on T340's 3 models** (family rule 12 wins / 1 tie / 2 losses; T340's own 10 / 3 / 2):
- the fresh models reverse the E1 result;
- the E2 result survives.

## Claims
- [measured] On 6 fresh DEV3 retrains, the pre-declared family rule beats sampling (or the zero answer) at equal
  compute in 11 of 24 non-tied cases. This is no go against the 2/3 line, and T340's ~12/15 does not replicate.
- [measured] Variant A on the interior error set (E2) beats sampling in 7 of 9 non-tied cases, at 14–21k FPE, within
  3x in 11/12.
- [measured] Variant B on the far-gap islands (E1) is within 2x in 14/18 cases, but too costly to beat sampling. It
  is 6–10x low on one model.

## Alternatives not ruled out
- My implementation differs from T340's on some rare cells by more than 20% (calibration). T340's own code on DEV3
  could score differently. This check tests the method, not T340's code.
- Six models: the E1 failure rests mainly on cost, and on one model (dev3_s505) for accuracy.

## Compute accounting
- Estimates: 6 models × (A ~14–21k FPE + B ~0.3–0.45M FPE as charged). About 25 CPU-min in total, single thread.
- Calibration: 3 models, about 12 CPU-min.
- Truths: T192 GPU.

## Check: simultaneous zero crossings (DRI, after the result)
The T218 child found a bug in the continuity walk used by T340's variant B. When an input coordinate crosses 0, all
16 step-0 pre-activations cross at the same t, and a strict root filter toggles only one gate.

My T312 walk is not a continuity walk. In every cell it recomputes the whole forward pass and all gates at
t + 1e-10, so simultaneous crossings are re-evaluated together.

Dense check (`python3 dense_check.py`, `results/dense_check.json`):
- For each of 60 random rows per line, I compared the walk's set length with a 50k-point forward grid on [0, 6].
- Lines tested:
  - E1/B at the GL nodes nearest v2 = 0 (where v3, v4 and the lows cross 0 early), and at two others, in both
    regions;
  - E2/A at both regions.
- Models: dev3_s505 and T340's seed0.

| model | max abs. length discrepancy | mean |
|---|---|---|
| dev3_s505 | 9.3e-5 | 1.1–2.0e-5 |
| seed0 | 8.2e-5 | 1.1–1.6e-5 |

This is at the grid's resolution (6/50k = 1.2e-4). **My walk does not have the bug, so the E1 and E2 numbers stand.**

## Status
**Done. NO GO** on the fixed family rule. Unverified. Waiting for the DEV3 truths (requested from T192).
