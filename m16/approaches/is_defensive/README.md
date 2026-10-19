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

Done (2026-10-19). Line FAILS (12/15 |z| <= 2, line 14); 15/15 within 2x; panel F .145.

## Results

### T403's 15 points: pre-registered line FAILS (12/15 with |z| <= 2; line 14), but every point is within 2x [measured]

Estimates frozen in 86c32c368 before `score3.py` read any truth -> [score3.json](score3.json).
F_pref = Z(t0) (exact), the gap-prefiltered MC counting network passes only. "z v2" is T453's z at the same point.

| # | model | t0 | stratum | truth | IS est +- se | ratio | z | hits | C | F | F (draws) | F_pref | F_walk | z v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | dev3_s576 | 1.0 | A | 1.150e-04 | 1.119e-04 +- 1.7e-06 | 0.97 | -1.8 | 83060 | 1.75e+06 | 0.14 | 1 | 0.044 | 12 | +1.6 |
| 1 | dev3_s564 | 2.75 | A | 2.983e-08 | 2.977e-08 +- 3.1e-10 | 1.00 | -0.2 | 102055 | 7.60e+06 | 1.2e-06 | 2.8e-06 | 9.4e-06 | 3.3e-05 | -0.0 |
| 2 | dev3_s580 | 1.25 | A | 9.586e-05 | 9.571e-05 +- 8.0e-07 | 1.00 | -0.2 | 94097 | 2.45e+06 | 0.00061 | 0.0033 | 0.017 | 0.058 | +1.0 |
| 3 | dev3_s583 | 1.5 | A | 4.108e-06 | 4.041e-06 +- 7.8e-08 | 0.98 | -0.8 | 79466 | 1.73e+06 | 0.0019 | 0.014 | 0.0059 | 0.0026 | -0.6 |
| 4 | dev3_s596 | 2.5 | A | 5.359e-08 | 5.353e-08 +- 8.7e-10 | 1.00 | -0.1 | 86422 | 3.46e+06 | 3e-07 | 1.2e-06 | 4e-05 | 0.0026 | -4.0 |
| 5 | dev3_s565 | 2.0 | A | 9.274e-07 | 9.028e-07 +- 1.1e-08 | 0.97 | -2.0 | 155360 | 5.06e+06 | 0.0033 | 0.01 | 0.00058 | 0.0025 | -2.8 |
| 6 | dev3_s571 | 1.0 | A | 6.497e-06 | 6.290e-06 +- 3.9e-07 | 0.97 | -0.5 | 28208 | 1.62e+06 | 0.011 | 0.082 | 0.044 | 0.042 | +0.6 |
| 7 | dev3_s595 | 1.75 | A | 5.913e-09 | 4.849e-09 +- 1.9e-10 | 0.82 | -4.0 | 948658 | 5.90e+06 | 0.0011 | 0.004 | 0.0019 | 0.31 | -16.5 |
| 8 | dev3_s590 | 1.5 | A | 4.040e-06 | 4.011e-06 +- 9.5e-08 | 0.99 | -0.3 | 67917 | 2.05e+06 | 0.00043 | 0.0027 | 0.0059 | 2.3 | +0.9 |
| 9 | dev3_s593 | 2.75 | A | 9.593e-09 | 9.534e-09 +- 1.4e-10 | 0.99 | -0.4 | 133728 | 2.00e+06 | 7.3e-07 | 4.7e-06 | 9.4e-06 | 6.3e-05 | -0.2 |
| 10 | dev3_s596 | 1.0 | B | 5.632e-05 | 5.459e-05 +- 1.2e-06 | 0.97 | -1.4 | 51455 | 4.79e+06 | 0.26 | 0.83 | 0.044 | 3.4 | -1.4 |
| 11 | dev3_s583 | 1.0 | B | 3.018e-05 | 3.215e-05 +- 8.4e-07 | 1.07 | +2.3 | 36724 | 1.78e+06 | 0.23 | 1.6 | 0.044 | 0.81 | +1.5 |
| 12 | dev3_s593 | 1.0 | B | 2.669e-05 | 2.656e-05 +- 2.9e-07 | 0.99 | -0.4 | 121789 | 2.21e+06 | 0.0015 | 0.0089 | 0.044 | 0.76 | +1.0 |
| 13 | dev3_s585 | 1.25 | B | 1.139e-06 | 1.105e-06 +- 3.4e-08 | 0.97 | -1.0 | 49178 | 1.30e+06 | 0.0013 | 0.012 | 0.017 | 2.9 | +1.3 |
| 14 | dev3_s570 | 3.0 | B | 2.708e-10 | 2.792e-10 +- 5.4e-12 | 1.03 | +1.3 | 36466 | 1.35e+06 | 3.5e-07 | 3.2e-06 | 2e-06 | 0.00017 | +0.9 |

**Summary.**
- **Line:** |z| <= 2 on **12/15** (line 14: **FAIL**).
- **Within 2x:** 15/15, median ratio 0.993.
- **Cost:** C mean 3.0e6 FPE, max 7.6e6.
- **F against the bars:** panel F = **0.145** in network passes, 0.68 with every draw charged; strict bar met both
  ways. Every point has F <= 1.
- **Against the comparators:** better F than the gap-prefiltered MC on 11/15 points (12/15 with draws charged), and
  better than T403's walk on 14/15.
- **The three misses:**
  - **#7 dev3_s595 t0 1.75:** 0.82x, z -3.97 (v2: 0.48x, z -16.5).
  - **#11 dev3_s583 t0 1.0:** 1.065x, z +2.31.
  - **#5 dev3_s565 t0 2.0:** 0.974x, z -2.02.
  - At #11 and #5 the largest weight reached the defensive cap Z/a_d (2.21e-1 of 2.22e-1; 2.84e-3 of 2.89e-3).
    Rare full-weight defensive hits make the s.e. itself noisy, the classic heavy-tail case. With 15 points, ~0.7
    points with |z| > 2 would be expected by chance even with honest intervals.
- **Against v2 (T453):** #4 is fixed (z -4.0 -> -0.1), #5 is nearly fixed (-2.8 -> -2.02), and #7 moves from
  -16.5 to -4.0.

### Post-hoc diagnosis of #7 (not pre-registered; run after scoring) [measured]

`python diag_s595.py 2e8` -> [diag_s595.json](diag_s595.json): defensive-only, i.e. gap-conditional, sampling of
dev3_s595 at t0 1.75. 630 hits, p ~ 6.08e-9, consistent with T403's 5.91e-9.

| triple | share of mass | min normalised r^2 of hits | in v3's mixture? |
|---|---|---|---|
| (9,5,4) | 38% | 4.98 | yes |
| (0,9,7) | 27% | 5.77 | yes |
| (9,0,7) | 21% | 5.73 | yes |
| **(7,0,1)** | **7%** | **4.79** | **NOT FOUND by pilot + descents** |
| (6,3,2) | 4% | 5.04 | yes |
| (9,1,3) | 2% | 8.22 | yes |
| (9,0,1), (5,9,6), (1,9,3), (8,4,3) | ~1% together | 6-11.5 | not found |

The pilot ranks N(0, I) draws by g23/||x|| and descends from the top 200. It missed triple (7,0,1), whose hits reach
a lower normalised r^2 (4.79) than the best point it found (4.98 on (9,5,4)). The local search therefore missed the
global dominating region. The defensive component did sample (7,0,1), but at 1.1e7 draws only a few hits arrive
there, too few to fix the estimate. That is expectation (b) of the pre-registration. About 8% of the mass sits on
unfound triples; the remaining ~10% shortfall is under-coverage within found triples.

**Obvious fix (untested, a follow-up):** take seeds from the gap-conditional law itself, i.e. defensive draws ranked
by kappa, or from stage-1 defensive hits. Each defensive hit is an event point and can seed a descent in its own
triple. That turns the defensive component from a variance guard into a discovery mechanism.

## Compute accounting

Per point:
- pilot: 1e7 draws, 5e5 network passes (~12 s);
- 200 descents: ~5-20 s, i.e. 1.3-5e6 FPE;
- stage 1: 1e6 draws; stage 2: 1e6 draws (+9e6 on extension).

Charged C per point: 1.3e6-7.6e6 FPE (mean 3.0e6). 15 points took ~1 h on 2 cores. The s595 diagnosis used 2e8
gap-conditional draws (~25 min).

## Claims

- [measured] A defensive gap-conditional component (a_d = .2) and a T436-size pilot fix 2 of v2's 3 over-confident
  points and move s595 from z -16.5 to -4.0. All 15 points are within 2x and panel F is .145 (strict bar). But the
  honest-interval line (>= 14/15 with |z| <= 2) still fails, at 12/15.
- [measured, post hoc] s595's residual bias comes from a dominating region (triple (7,0,1), 7% of the mass) that the
  pilot never found. A defensive component guarantees a bounded weight, not discovery at this sample size.
- [inferred] Weight-capped hits at #5 and #11 show that the s.e. is unreliable whenever defensive hits carry
  material weight. A heavy-tail diagnostic (e.g. max weight / sum of weights) should gate the reported interval.

## Alternatives not ruled out

- The two marginal misses (#11 z 2.31, #5 z -2.02) may be chance: about 0.7 such points are expected among 15 honest
  ones. Separating that needs replicate runs, which were not done.
- A larger a_d or many more draws would make s595's missed triple visible in the s.e. but raise F; not tested.

## Failure modes

- Local seed search: a dominating region far from every pilot seed stays uncovered. The defensive component then
  contributes only rare, capped-weight hits, and the s.e. understates the error.
