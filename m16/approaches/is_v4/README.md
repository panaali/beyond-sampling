# T456: dominating-point IS v4 (defensive hits seed descents, heavy-tail gate) on a FRESH panel

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T454 (v3)](../is_defensive/README.md) · [T453 (v2)](../dominating_is2/README.md)

## Question

T454's v3 was within 2x on 15/15 of T403's points but over-confident on 3. The worst miss came from a dominating
region (triple (7,0,1) of dev3_s595) that the pilot never found. T403's points have informed three designs, so v4
is tested on a **fresh panel**. Do discovery from defensive hits plus a heavy-tail gate give honest intervals?
**Class: sampling-based.**

## Method ([is4.py](is4.py), [run4.py](run4.py))

v3 (pilot 1e7 / 5e5 / 200 seeds; per-triple cap 3, 60 in total; cross-entropy anisotropic components; exact
defensive gap-conditional component) with:
1. **Discovery stage.** Stage 1 (2e6 draws) uses defensive weight a_d1 = **0.5**.
   - Its hits, taken in order of increasing ||x||/g23 and capped at **5 per triple and 100 in all**, seed new region
     descents.
   - The new points are merged into the set (merge distance 0.05) and the point set is re-selected with the per-triple
     cap. The CE fit and component weights then use the stage-1 hits.
2. **Stage 2 (the estimate):** a_d2 = **0.2**, 1e6 draws, one extension to 1e7 if the relative s.e. > 10% (as v3).
3. **Heavy-tail gate** (on stage-2 weights; fixed here, before any estimate on the panel). The interval is
   **flagged** if any of:
   - max w / sum w > **0.05**;
   - **fewer than 30** near-cap hits (w > 0.5 Z/a_d2) carry > **20%** of the estimate;
   - there are no hits.
   A design-stage code check on dev3_s526 (not on the panel) showed that a gate without the "fewer than 30" clause
   fires at p/Z ~ 2e-3, where capped hits are many and harmless. The clause was added then, before any panel run.
4. **Cost C:** pilot network passes, pilot and discovery descents at 4 us per FPE, stage-1 and stage-2 network passes.
   C_draws charges every draw as a pass.

## Pre-registration (2026-10-19, before any estimate on a panel model)

**Panel (model-free rule):** DEV3 models in name order, dev3_s500 ... dev3_s514 (15 models), with t0 cycling
1.25, 1.75, 2.25 (s500 1.25, s501 1.75, s502 2.25, s503 1.25, ...) -> [panel.json](panel.json).
- These models are unused by T403/T453/T454 (dev3_s564-s599) and T370 (dev3_s548-s563).
- Some were studied by earlier rare-event lanes (T340-T388); I have not read their truths.
- Seeds: pilot 5000 + I, stages 6000 + 10 I + {0, 1, 2}.

**Truths (`run4.py truth I`, own gap-conditional MC, run only after all 15 estimates are pushed):**
- Exact sampling of phi 1{g23 >= t0} / Z (the v3 defensive sampler), seed 7000 + I, blocks of 1e6.
- **Hit floor 1000**, cap 2e9 draws. p = Z hits / n, s.e. = Z sqrt(hits) / n.
- A point that reaches the cap with < 100 hits is reported but counted as NOT honest (no exclusion).

**Line:** >= 14/15 points honest. A point is honest if |z| <= 2, with z = (p_hat - truth) / sqrt(se^2 + se_truth^2),
**or** if the gate flagged it in the frozen estimate.
**Also reported:**
- the raw |z| <= 2 count, ignoring flags;
- the flag rate (a gate that fires everywhere would make the line empty);
- within 2x; F per point and panel F against strict/lenient bars, in network passes and with draws charged;
- F_pref = Z(t0), the prefiltered-MC comparator, and 0.19 + Z with draws charged;
- cost (mean, max).

**Expectations / alternatives (rule 3).**
- (a) Discovery finds the regions the pilot misses. Raw |z| <= 2 on >= 14/15, few flags.
- (b) Alternative: at p/Z ~ 1e-6 to 1e-5, stage 1's defensive half (1e6 draws) gives only ~1-10 hits, so discovery
  is weak for the rarest points. Signature: misses at t0 = 2.25 with the gate silent.
- (c) The gate fires on many points (> 5). The line then passes vacuously, and the raw count is the honest measure.
- (d) Cost rises (~2e6 more network passes in stage 1). F at t0 = 1.25 may lose to the prefiltered MC (Z = 0.017).

## Status

Done (2026-10-19). **Line PASSES: 15/15 honest, all by raw |z| <= 2 (0 gate flags).**

## Results [measured]

Estimates frozen in 91e859917, then truths computed (`run4.py truth I`) -> [score4.json](score4.json) via `python score4.py`.
F_pref = Z(t0): the gap-prefiltered MC, counting network passes only.

| # | model | t0 | truth (hits) | IS est +- se | ratio | z | flag (maxw, capshare, n_capped) | new desc. | triples before/after | C | F | F_pref |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dev3_s500 | 1.25 | 7.173e-05 (4243) | 7.087e-05 +- 5.9e-07 | 0.99 | -0.7 | - (0.001, 0.03, 36) | 100 | 11/23 | 2.94e+06 | 0.03 | 0.017 |
| 1 | dev3_s501 | 1.75 | 9.398e-06 (4869) | 9.416e-06 +- 9.6e-08 | 1.00 | +0.1 | - (0.001, 0.05, 73) | 100 | 19/23 | 3.24e+06 | 0.00011 | 0.0019 |
| 2 | dev3_s502 | 2.25 | 1.332e-06 (8375) | 1.340e-06 +- 9.5e-09 | 1.01 | +0.5 | - (0.001, 0.02, 59) | 100 | 20/22 | 3.48e+06 | 0.00016 | 0.00016 |
| 3 | dev3_s503 | 1.25 | 1.230e-04 (7279) | 1.204e-04 +- 1.1e-06 | 0.98 | -1.4 | - (0.001, 0.06, 118) | 100 | 25/25 | 3.17e+06 | 0.18 | 0.017 |
| 4 | dev3_s504 | 1.75 | 3.306e-06 (1713) | 3.202e-06 +- 3.0e-08 | 0.97 | -1.2 | - (0.003, 0.02, 6) | 69 | 9/18 | 2.88e+06 | 0.0095 | 0.0019 |
| 5 | dev3_s505 | 2.25 | 3.244e-07 (2039) | 3.139e-07 +- 3.7e-09 | 0.97 | -1.3 | - (0.002, 0.04, 22) | 81 | 14/18 | 2.81e+06 | 0.00095 | 0.00016 |
| 6 | dev3_s506 | 1.25 | 6.349e-05 (3756) | 6.223e-05 +- 5.5e-07 | 0.98 | -1.1 | - (0.001, 0.02, 26) | 100 | 11/25 | 4.48e+06 | 0.11 | 0.017 |
| 7 | dev3_s507 | 1.75 | 3.308e-05 (17138) | 3.301e-05 +- 1.8e-07 | 1.00 | -0.2 | - (0.000, 0.03, 135) | 100 | 29/31 | 3.03e+06 | 0.00042 | 0.0019 |
| 8 | dev3_s508 | 2.25 | 2.506e-07 (1575) | 2.499e-07 +- 2.8e-09 | 1.00 | -0.1 | - (0.003, 0.02, 8) | 85 | 15/16 | 2.37e+06 | 4.6e-06 | 0.00016 |
| 9 | dev3_s509 | 1.25 | 1.444e-04 (8544) | 1.449e-04 +- 1.2e-06 | 1.00 | +0.2 | - (0.001, 0.05, 128) | 100 | 16/23 | 2.99e+06 | 0.0049 | 0.017 |
| 10 | dev3_s510 | 1.75 | 1.255e-05 (6504) | 1.271e-05 +- 7.9e-08 | 1.01 | +0.9 | - (0.001, 0.01, 15) | 100 | 13/24 | 3.91e+06 | 0.0081 | 0.0019 |
| 11 | dev3_s511 | 2.25 | 1.328e-07 (1670) | 1.326e-07 +- 1.7e-09 | 1.00 | -0.1 | - (0.005, 0.01, 2) | 41 | 8/13 | 5.27e+06 | 2.4e-06 | 0.00016 |
| 12 | dev3_s512 | 1.25 | 2.597e-05 (1536) | 2.596e-05 +- 3.5e-07 | 1.00 | -0.0 | - (0.003, 0.03, 12) | 90 | 12/20 | 3.60e+06 | 1.1e-07 | 0.017 |
| 13 | dev3_s513 | 1.75 | 2.146e-05 (11118) | 2.181e-05 +- 1.3e-07 | 1.02 | +1.5 | - (0.000, 0.02, 52) | 100 | 17/21 | 3.41e+06 | 0.02 | 0.0019 |
| 14 | dev3_s514 | 2.25 | 3.876e-07 (2436) | 3.996e-07 +- 4.2e-09 | 1.03 | +1.4 | - (0.002, 0.03, 24) | 99 | 14/22 | 2.98e+06 | 0.0011 | 0.00016 |

**Summary.**
- **Line:** honest **15/15** (raw |z| <= 2 on 15/15; gate flags 0) -> **PASS** (line 14). Max |z| 1.5.
- **Accuracy:** within 2x 15/15, median ratio 0.998, ratios 0.97-1.03.
- **Cost:** C mean 3.4e6 FPE, max 5.3e6.
- **F against the bars:** panel F = **0.064** in network passes and 0.27 with every draw charged. Every point has
  F <= 1 (strict bar).
- **Against the prefiltered MC:** better on **6/15** counting network passes (F_pref = Z), and **13/15** when MC draws
  are charged 0.19 FPE.
  - IS wins clearly where the event is rare inside the gap set: t0 = 1.75 points and some t0 = 2.25 points.
  - It loses at t0 = 1.25 (Z = .017), where its ~3e6 FPE fixed cost dominates. At t0 = 2.25 it is about equal to
    the free prefilter (Z = 1.6e-4).
- **Discovery mattered:** the number of distinct triples in the mixture grew on 13/15 points (e.g. 11 -> 23,
  9 -> 18, 11 -> 25), with 41-100 discovery descents per point.
- **Truth precision:** truths have 1,536-17,138 hits (relative s.e. 0.8-2.6%); blocks of 1e6 overshoot the 1000-hit
  floor. So |z| <= 2 rules out biases above ~2-5%, not smaller ones.

## Compute accounting

Per point:
- pilot: 1e7 draws, 5e5 network passes;
- descents: pilot 200 plus discovery 41-100;
- stage 1: 2e6 draws (half defensive, all of which pass the gap filter);
- stage 2: 1e6 draws.

Charged C per point: 2.4e6-5.3e6 FPE. Estimates took ~1 h on 2 cores; truths (gap-conditional MC, 1e6-draw blocks)
took ~15 min.

## Claims

- [measured] On a fresh, model-free panel (dev3_s500-s514 x t0 1.25/1.75/2.25), IS v4 gives intervals that are
  honest at the |z| <= 2 level on 15/15 points (none needed the gate), within 3% of truth, at panel F = .064.
  Sampling-class.
- [measured] Discovery from defensive hits adds triples the pilot missed on 13/15 points. That was the failure mode of
  v3 (dev3_s595).
- [measured] IS beats the free gap prefilter only where p / Z(t0) is small. At t0 = 1.25 it does not.
- [inferred] The pass rests on 15 points; the panel's own tie structure was not measured.

## Alternatives not ruled out

- This panel may be easier than T403's tie-dominated selection: T403 picked points where the walk predicted a high
  triple-tie share, while this panel is model-free. v4 was not re-run on T403's 15 points (they informed the design).
- Truth s.e. of 1-3% limits the test: a 2% bias would pass. Higher-precision truths, or replicate IS runs, would
  tighten it.
- The gate never fired, so its usefulness is untested here.

## Failure modes

- Fixed cost of ~3e6 FPE: at moderate rarity (p >~ 1e-5 with Z(t0) >~ 1e-2) the gap-prefiltered MC is cheaper.
- A dominating region with no defensive hits in stage 1 (p / Z << 1 / 1e6) can still be missed. The defensive weight
  then bounds the damage only in variance, not in bias at finite n.
