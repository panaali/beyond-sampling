# T453: dominating-point IS v2 (per-triple cap, adaptive anisotropic mixture) and T403's 15 blind points

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T436 (v1)](../dominating_is/README.md)

## Question

T436's dominating-point IS passed at E1(2.5) on M_16,10, but its 30-point cap filled with (9,3,2) neighbours, so
other triples were barely sampled, and its hit rate was ~2e-4 (thin tie slivers). Does a per-triple cap plus
anisotropic components fix that (E1(1.5) on the target, every triple holding >= 5% of MC hits actually sampled)?
And does the method work **without knowing where the failures are**, on T403's 15 points: DEV3 retrains, chosen
truth-blind by the walk estimator's predicted tie share? **Class: sampling-based** (unbiased IS, sampled s.e.).

## Method ([is2.py](is2.py), [run2.py](run2.py))

Same event, triple convention and scale invariance as T436: E1(t0) = K ∩ {g23 >= t0}, K a cone, since none of these
nets has biases.
1. **Cheap pilot:** 2e6 draws x ~ N(0, I). Rank them by kappa0 = g23 / ||x||, computed without the net. Run the net
   on the top 1e5 only (5% of each 5e5 block). Seeds = the top 60 cone points by kappa.
2. **Local descents** (as T436): region QPs min ||x||^2 at g23 >= 1, crossing gate boundaries. Merge points within 0.05.
3. **Point set at t0, with the per-triple cap:** take points in increasing r^2 with r^2 <= r_min^2 + 2 ln(1000)/t0^2.
   Keep at most **3 per triple** and **60 in total**.
4. **Stage 1** (1e6 draws; it sets the proposal and is NOT used in the estimate): mixture a0 N(0, I) + (1 - a0)
   sum_k w_k N(t0 x_k, I), a0 = 0.05, w_k ∝ exp(-t0^2 r_k^2 / 2). Each hit goes to its nearest centre t0 x_k.
   - A component with >= 30 hits gets the IS-weighted mean m_k and covariance C_k of its hits. The covariance is
     inflated by 1.5, eigenvalues clipped to [0.01, 4]. This is the anisotropic part: narrow across the sliver.
   - The component becomes 0.8 N(m_k, C_k) + 0.2 N(t0 x_k, I). Components with < 30 hits stay N(t0 x_k, I).
   - Component weights ∝ max(share_k, 0.01), where share_k is the share of the stage-1 estimate from component k's
     hits.
5. **Stage 2** (the estimate): n2 = 1e6 draws from the adapted mixture (defensive a0 = 0.05 kept).
   - p_hat = mean(1_E phi / q); s.e. = sd / sqrt(n2).
   - ONE extension to 1e7 draws (fresh seed, pooled) if the relative s.e. > 10%.
6. **Cost C:** pilot network passes (1e5), descents at 4.0 us per FPE, and network passes of stages 1 and 2 (the gap
   prefilter is free, as T400/T436). Also reported: C_draws, which charges every draw (pilot, stage 1, stage 2) as a
   full pass.
   - F = s.e.^2 C / (p(1 - p)), against the strict (F <= 1) and lenient (F <= 10) bars.
   - Gap-prefiltered MC comparator (T400's MC; T403's truth sampler is its conditional version) has
     F_pref = P(g23 >= t0), counting network passes only. With its measured draw + sort cost (0.19 FPE per draw)
     it is ~0.19 + P(g23 >= t0). Both are reported.
   - The T403 walk estimator's own F (from T403's table) is reported next to ours.

## Pre-registration (2026-10-19, before any v2 estimate on the target or on a T403 model)

**A. Validation, E1(1.5) on M_16,10** (`run2.py target 1.5`, pilot seed 101, stage seeds 201-203).
Truth = T436's own MC: 4e9 draws, 821 hits, 2.052e-7 +- .072e-7. Triples with >= 5% of MC hits:
(9,3,2) 61%, (8,1,2) 15%, (0,9,7) 8.0%, (9,0,7) 7.7%.
**Pass** if |p_hat - p_MC| <= 2 sqrt(se^2 + se_MC^2) AND each of those four triples has >= 30 stage-2 IS hits.

**B. T403's 15 points** (`run2.py t403 I`, I = 0..14 in the order of T403's `results/score.json` rows; models and
thresholds in [t403_points.json](t403_points.json); pilot seed 1000 + I, stage seeds 2000 + 10 I + {0, 1, 2}).
- Same settings for every point; nothing is tuned per point.
- Truths: T403's v3-conditional MC (>= 1,011 hits per point). They are read only by `score2.py` after all 15
  estimates are pushed.
- Scored per point: ratio to truth; within 2x; z = (p_hat - truth) / sqrt(se^2 + se_truth^2); F_i; F_pref_i.
- Panel scores: share within 2x, share |z| <= 2, and panel F = mean(e^2) max(C, 1e3) / mean(p(1-p)) with
  C = mean cost.
- **Pre-registered line for B: >= 12/15 within 2x AND >= 13/15 with |z| <= 2** (the s.e. must be honest).
  Also reported: strict and lenient bars, F against the gap-prefiltered MC, and against T403's walk.

**Expectations / alternatives (rule 3).**
- (a) v2 samples all four target triples and keeps the T436 result. Its F should improve, because the anisotropic
  fit raises the hit rate.
- (b) Alternative: the CE fit over-narrows on a sliver, so tails are under-covered. Signature: |z| > 2 with low
  ratios, while the s.e. looks small.
- (c) Alternative: on DEV3 models the 60-seed pilot misses a dominant region (non-tie islands, as T403's s585).
  Signature: a low ratio with few hits.
- (d) At p ~ 1e-4 to 1e-5 (t0 = 1.0 points), the fixed pilot and descent cost (~1e6 FPE) makes F poor against
  gap-prefiltered MC, whose F_pref = P(g23 >= 1) ~ 0.04. IS should win only at the rare points.

**Disclosure.** Before this pre-registration I saw one T403 truth: dev3_s576 at t0 = 1.0, 1.15e-4, while reading the
file format. The code check ran on the excluded model dev3_s526: t0 = 1.0, tiny n, 6.2e-5 +- 2.4e-5 vs T371's
8.75e-5. The seed count was cut from 200 to 60 after timing descents on dev3_s526 (8.4 s for 168 seeds), for cost
only.

## Status

Pre-registered; running.

## Results

Pending.

## Compute accounting

Pending.

## Claims

None yet.

## Alternatives not ruled out

Pending.

## Failure modes

Pending.
