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

Pre-registered; running.
