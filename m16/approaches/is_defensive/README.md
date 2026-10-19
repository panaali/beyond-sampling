# T454: honest intervals for dominating-point IS: defensive mixture + larger pilot, re-tested on T403's 15 points

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T453 (v2)](../dominating_is2/README.md) · [T436 (v1)](../dominating_is/README.md)

## Question

T453's IS v2 was efficient (panel F .42) but over-confident: |z| > 2 on 3 of T403's 15 points, worst dev3_s595
t0 1.75 (0.48x, z -16.5), a region no component covered. Do a defensive component (Hesterberg 1995) and a
T436-size pilot make the intervals honest? **Class: sampling-based.**

## Method ([is3.py](is3.py), [run3.py](run3.py))

v2 (two-stage cross-entropy mixture, per-triple cap 3, as T453) with two changes:
1. **Defensive component, weight a_d = 0.2**, replacing v2's 5% N(0, I).
   - The component is the gap-prefiltered nominal law q_def(x) = phi(x) 1{g23 >= t0} / Z(t0).
   - It is sampled exactly: v3 from its density 360 phi Phi^7 (1 - Phi(v + t0))^2 / Z (tabulated inverse CDF),
     two tops iid N | > v3 + t0, seven lows iid N | < v3, uniform random positions.
   - Z(t0) by quadrature; checked against a trapezoid on 2e6 nodes (equal to 1e-15) and a 2e8-draw MC at t0 = 3
     (2.02e-6 vs 2.003e-6).
   - The event lies inside {g23 >= t0}, so every hit's weight is <= Z / a_d (reported per run as max_weight vs
     weight_cap). Per draw, the IS second moment is <= (Z / a_d) p, against the prefiltered MC's p Z per network
     pass: at most 1/a_d = 5x worse per network pass.
   - Any region the mixture misses is still sampled at rate a_d p_missed / Z per draw, so it shows in the s.e.
     (heavy weights), not silently.
2. **Pilot:** 1e7 draws, net on the top 5e5 by g23/||x||, 200 seeds (T436 coverage; v2 used 2e6 / 1e5 / 60).

Everything else as T453: per-triple cap 3 and 60 in total; stage 1 = 1e6 draws (not in the estimate); stage 2 =
1e6 draws, one extension to 1e7 if the relative s.e. > 10%. Cost C counts network passes (the defensive draws all
pass the gap filter), pilot passes and descents at 4 us per FPE; C_draws charges every draw. No global MIQP (T436:
no incumbent in 900 s, cold); not retried.

## Pre-registration (2026-10-19, before any v3 estimate)

- **Points:** T403's 15 (model, t0) in the order of [t403_points.json](t403_points.json).
  - Point I uses pilot seed 3000 + I and stage seeds 4000 + 10 I + {0, 1, 2}.
  - Run order: **#7 (dev3_s595, t0 1.75) first**, then the rest in two lanes.
- **Freezing:** all 15 estimates are pushed before `score3.py` reads T403's truths.
- **Pass line:** **>= 14/15 with |z| <= 2**, z = (p_hat - truth) / sqrt(se^2 + se_truth^2), at the stated cost
  (mean and max C).
- **Also reported:**
  - within 2x;
  - F per point and panel F, against the strict (<= 1) and lenient (<= 10) bars, in network passes and with draws
    charged;
  - the prefiltered-MC comparator F_pref = Z(t0) (exact; T453 used a 1e8-draw MC estimate, ~2 sigma high at 2.5-3)
    and 0.19 + Z when draws are charged;
  - T453 v2 and T403's walk side by side.
- **Secondary (not in the line):** E1(1.5) and E1(2.5) on M_16,10 (pilot seed 301, stages 401-403) vs T436's MC and
  T400's MC.

**Expectations / alternatives (rule 3).**
- (a) The defensive component makes s595's missing mass show up as a few heavy-weight hits. The estimate becomes
  noisier but honest (|z| <= 2), and F rises from v2's 0.001 at that point.
- (b) Alternative: with p_missed / Z ~ 1e-6, 1e6-1e7 draws give ~0-2 defensive hits there, so the missing mass may
  still not appear. Then z stays large; the honest fix would be more draws. Signature: s595 still ~0.5x with a small
  s.e.
- (c) The bigger pilot finds s595's missing region directly. Signature: ratio ~1 with many hits.
- (d) Cost: the larger pilot plus descents add ~2-3e6 FPE per point. F at the t0 = 1.0 points rises and loses more
  often to the prefiltered MC.

## Status

Pre-registered; running.
