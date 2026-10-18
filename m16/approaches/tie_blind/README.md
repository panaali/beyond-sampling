# tie_blind — T223: tie-slicing with the blind region split off (best honest deterministic number for M_16,10)

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [sci_tao (E25)](../sci_tao/README.md) · T218 (branch
claude/t218-tie-tail-labels, tie_tail_labels/) · [width16_probe](../width16_probe/README.md)

**All claims are unverified** (one agent; no independent check).

## Question
T218 found that the blind region B = {at most one positive input} (P(B) = 11/1024 exactly at T = 10) carries 60% of
the second moment of sci_tao's tie-slicing face integrand on M_16,10 (8-25% on noisy copies). If its contribution is
computed separately, does the best deterministic tie-slicing estimator improve on leaderboard v3?

## Method
Quadrature class (fixed, model-independent node sets; `rng` ignored; passes the leaderboard determinism check).
- `core_nb.py`: sci_tao's exact co-area walk (`sci_tao/core.py`, imported read-only), restricted to the non-blind set
  N = {x_(2) > 0}. On a face segment the 2nd-largest value is u + |t|/sqrt2 (R23) or u - |t|/sqrt2 (R12), so N's part
  of each segment is a closed-form t-interval. P(wrong) = Z23 E23[F_N] + Z12 E12[F_N] + P(B) E_B[1 wrong], exactly.
- **The blind part cannot be computed exactly.** Inside B the target's prefix regions still grow 12-19x per step
  (T218 blind sub-test), so exact enumeration costs about as much as on the full space (T5/T222: infeasible). The only
  free bound, [0, P(B)] = 1.07 pp, is about 13x the sampling SE at 64k. So B gets a third fixed stratum instead: a
  scrambled-Sobol rule mapped into its 11 equally likely orthants (half-normal magnitudes), one forward pass per node.
  Its per-node sd is P(B) sqrt(q(1-q)) ~ 0.0045 (q = 0.78 wrong in B on orig), against 0.21 for plain sampling.
- `estimator.py`: sci_tao's allocation (25% pilot spread evenly, then Neyman by sd x weight / sqrt(cost)) over three strata
  (R23 faces, R12 faces, blind nodes). Face walks are charged **2.23x** their counted forward-pass equivalents inside
  the estimator, the DRI audit's measured factor for the identical sci_tao walk (`OVERRIDES["sci_tao"]`). Blind nodes cost
  1 FPE each.
- `score.py`: leaderboard v3 (`m16/leaderboard.py`, imported unchanged: same 30 models, 32M-sample truths, scoring,
  cluster bootstrap) with the raw file and report redirected into `lb/`, so no shared file is touched. sci_tao's
  committed raw rows (results/leaderboard_v3_raw.jsonl, budgets 1k-256k) are copied in, so both rows go through the same
  `table()`. `seeds.py`: control over 16 independent rule scrambles.

**Checks.** On orig with 2^17 faces, the non-blind part is 0.038465 against 3.2e7-sample MC 0.038545 ± 0.000034 (difference
within the rule's own ~1.2e-4 error). The blind rule with 2^20 nodes gives 0.008368 against MC 0.008371 ± 0.000016. The split
cuts the R23 face second moment 2.5x on orig (sd 0.072 -> 0.042 per face) at the same cost per face.

## Status
done (2026-10-18). Class: **quadrature** (assigned here pending DRI review, as for sci_tao). Not mechanistic.

## Results
### Leaderboard v3, every budget, per group (MSE / sampling MSE at equal charged cost; below 1 beats sampling)
"budget" is the leaderboard's budget argument. sci_tao spends it in uncorrected walk units, so its charged cost is ~2.23x
the budget, while tie_blind charges the factor itself. Ratios are at each row's own charged cost. Reproduce:
`m16/approaches/tie_blind/run.sh` (writes lb/raw.jsonl, lb/table.json). T196 units: every ratio x
fpe_flops/c_s = 6000/5948 = 1.009 (`ratio_T196` in lb/table.json); no verdict changes.

| group (n) | budget | tie_blind ratio [95% CI] | sci_tao ratio [95% CI] | tie_blind cost / sci_tao cost (FPE charged) | bias pp tie_blind / sci_tao |
|---|---|---|---|---|---|
| orig (1) | 1k | 4.89 [4.89, 4.89] | 0.28 [0.28, 0.28] | 1163 / 2740 | -1.370 / -0.215 |
| orig (1) | 4k | 2.54 [2.54, 2.54] | 1.57 [1.57, 1.57] | 3880 / 8651 | -0.541 / -0.285 |
| orig (1) | 16k | 0.48 [0.48, 0.48] | 2.16 [2.16, 2.16] | 16062 / 36023 | -0.115 / -0.164 |
| orig (1) | 64k | 2.62 [2.62, 2.62] | 0.05 [0.05, 0.05] | 64146 / 142391 | -0.135 / -0.013 |
| orig (1) | 256k | 4.63 [4.63, 4.63] | 0.03 [0.03, 0.03] | 254870 / 568620 | -0.090 / +0.006 |
| seed (5) | 1k | 2.19 [0.40, 4.61] | 0.17 [0.03, 0.31] | 1164 / 2590 | -0.748 / -0.005 |
| seed (5) | 4k | 0.28 [0.07, 0.52] | 4.03 [1.10, 8.41] | 3841 / 8653 | +0.065 / -0.275 |
| seed (5) | 16k | 2.11 [0.88, 3.20] | 1.92 [0.77, 3.21] | 16016 / 35300 | -0.081 / +0.045 |
| seed (5) | 64k | 2.55 [0.14, 5.14] | 3.11 [0.93, 5.93] | 64249 / 143101 | +0.097 / +0.111 |
| seed (5) | 256k | 0.68 [0.22, 1.29] | 2.17 [0.36, 5.15] | 254848 / 568394 | +0.014 / +0.044 |
| s0.0025 (6) | 1k | 6.15 [2.65, 9.86] | 0.07 [0.03, 0.13] | 1167 / 2630 | -1.618 / +0.074 |
| s0.0025 (6) | 4k | 0.59 [0.18, 1.07] | 1.15 [0.74, 1.57] | 3862 / 8592 | -0.262 / -0.219 |
| s0.0025 (6) | 16k | 0.39 [0.08, 0.78] | 0.82 [0.12, 1.44] | 16118 / 36074 | -0.088 / -0.084 |
| s0.0025 (6) | 64k | 0.31 [0.12, 0.51] | 0.98 [0.06, 2.49] | 64351 / 142442 | -0.042 / -0.032 |
| s0.0025 (6) | 256k | 3.24 [1.15, 6.00] | 0.19 [0.06, 0.33] | 254453 / 567301 | -0.076 / +0.005 |
| s0.005 (6) | 1k | 3.56 [1.02, 6.58] | 0.71 [0.12, 1.50] | 1141 / 2651 | -1.313 / +0.211 |
| s0.005 (6) | 4k | 1.47 [0.72, 2.63] | 0.83 [0.21, 1.73] | 3915 / 8620 | -0.264 / -0.075 |
| s0.005 (6) | 16k | 1.30 [0.47, 1.97] | 1.19 [0.42, 1.97] | 16063 / 35779 | +0.034 / +0.054 |
| s0.005 (6) | 64k | 0.77 [0.38, 1.09] | 0.42 [0.03, 0.72] | 64327 / 142490 | +0.067 / +0.005 |
| s0.005 (6) | 256k | 2.74 [0.17, 6.29] | 1.06 [0.03, 2.39] | 254553 / 567448 | -0.062 / -0.011 |
| s0.01 (6) | 1k | 4.81 [1.55, 7.33] | 2.31 [0.37, 4.93] | 1155 / 2665 | -0.165 / +0.100 |
| s0.01 (6) | 4k | 3.78 [0.17, 9.51] | 2.13 [0.89, 3.24] | 3868 / 8543 | -0.384 / -0.188 |
| s0.01 (6) | 16k | 5.51 [2.26, 8.10] | 3.40 [1.06, 5.97] | 16019 / 35515 | +0.540 / +0.301 |
| s0.01 (6) | 64k | 4.13 [1.71, 6.27] | 2.54 [0.79, 5.04] | 64370 / 142296 | +0.084 / +0.067 |
| s0.01 (6) | 256k | 4.53 [1.25, 8.28] | 2.19 [0.60, 3.84] | 254435 / 567920 | -0.040 / -0.046 |
| s0.02 (6) | 1k | 10.52 [2.44, 20.55] | 5.19 [0.85, 12.34] | 1188 / 2697 | +0.432 / -0.665 |
| s0.02 (6) | 4k | 9.15 [2.21, 20.48] | 6.49 [2.59, 10.29] | 3934 / 8649 | -1.582 / -0.788 |
| s0.02 (6) | 16k | 5.65 [3.06, 8.01] | 5.52 [2.50, 8.93] | 15947 / 35392 | +0.499 / -0.061 |
| s0.02 (6) | 64k | 4.96 [2.22, 8.00] | 4.61 [1.38, 7.97] | 64082 / 142535 | +0.159 / +0.053 |
| s0.02 (6) | 256k | 5.09 [2.57, 7.15] | 1.24 [0.16, 3.02] | 254934 / 569355 | -0.083 / -0.050 |

### Pooled over all 30 models (cluster bootstrap over models; `lb/table_pooled.json`)
| budget | tie_blind | sci_tao |
|---|---|---|
| 1k | 6.54 [3.25, 10.62] | 2.63 [0.73, 5.48] |
| 4k | 4.67 [1.42, 9.27] | 3.64 [1.82, 5.49] |
| 16k | 3.98 [2.48, 5.36] | 3.42 [1.86, 5.04] |
| 64k | 3.36 [1.97, 4.77] | 2.86 [1.45, 4.42] |
| 256k | 3.91 [2.21, 5.51] | 1.44 [0.63, 2.29] |

### Control: 16 independent rule scrambles per model (`seeds.py` -> seeds.json; MSE over scrambles, no CI)
| model | 16k: tie_blind / sci_tao | 64k: tie_blind / sci_tao |
|---|---|---|
| orig | 0.75 / 0.99 | 0.89 / 1.56 |
| seed0 | 0.86 / 1.16 | 0.93 / 1.07 |
| s0.0025_0 | 1.49 / 1.16 | 0.83 / 1.50 |
| s0.005_0 | 3.40 / 1.59 | 0.92 / 2.15 |
| s0.01_0 | 2.16 / 2.01 | 4.91 / 4.16 |
| s0.02_0 | 3.41 / 1.88 | 3.10 / 3.38 |

**Reading.**
1. **No honest deterministic win on M_16,10.** Pooled over the 30 leaderboard models, every point estimate of both methods is
   above 1; only sci_tao's 1k and 256k CIs (0.73-5.48, 0.63-2.29) reach below 1. tie_blind's pooled ratios are
   3.4-6.5. The headline sci_tao "0.03 @ 256k on orig" (leaderboard summary) is one deterministic run at the luckiest
   budget; the same rule gives 2.16 at 16k, and over 16 scrambles 0.99 (16k) and 1.56 (64k).
2. **Splitting off B helps where T218 said it would**, on low-noise models. Over 16 scrambles, orig goes 0.99 -> 0.75 (16k)
   and 1.56 -> 0.89 (64k), and seed0 1.16 -> 0.86 / 1.07 -> 0.93. It does not help on sigma >= 0.005 copies (blind share 8-25%), and the
   pilot-based Neyman allocation over three strata is noisier at small budgets (1k-4k bias -0.5..-1.6 pp).
3. Best honest deterministic number on the target itself: **tie_blind on orig, MSE ratio 0.75 (16k) / 0.89 (64k) over 16
   rule scrambles**, i.e. 1.1-1.3x better than sampling, a quadrature-class result. Single-run leaderboard rows on orig are
   0.48 (16k) and 2.62 (64k), within the scramble-to-scramble spread.

## Compute accounting
Cost unit: FPE (1 FPE = 6000 FLOPs at (16,10)); face walks charged 2.23x as audited; blind nodes 1 FPE. Total CPU: leaderboard
run 30 models x 5 budgets ~2 min; validation MC 3.6e7 forward passes ~3 min; scramble control 6 models x 2 budgets x 16 x
2 methods ~5 min. Well under the 2 h cap; no reduced budget set needed.

## Claims (all unverified)
- [measured] The blind/non-blind split is exact (checks above); it reduces orig's R23 face second moment 2.5x.
- [measured] Leaderboard v3 table above: tie_blind does not beat sampling pooled over 30 models at any budget (3.4-6.5); neither does sci_tao
  except CIs straddling 1 at 1k/256k.
- [measured] Over 16 rule scrambles tie_blind beats sci_tao on orig and seed0 (1.15-1.75x) but not on sigma >= 0.005 copies.
- [inferred] Removing the blind region is a variance reduction of at most ~1.5x on the clean target, not a change of class;
  the remaining non-blind tail (T218: top 5% = 72% of its second moment) still limits deterministic tie-slicing.

## Alternatives not ruled out
- Allocation: a fixed (non-pilot) allocation across the three strata, or more pilot nodes, may remove the small-budget
  bias. Not tuned, to avoid fitting on test models (DEV models exist for that).
- The walk still traverses the blind middle of R23 segments (charged, not used); skipping it would cut cost a little.
- Exact blind mass via a cheaper structure (e.g. the all-negative orthant alone, 1/1024, where every gate starts off) was
  not attempted.
- 16 scrambles give noisy per-model MSEs (heavy-tailed integrand); no CI on the control.

## Failure modes
- Small budgets (<= 4k): the three-way pilot leaves too few faces per stratum, and the Neyman step amplifies pilot noise.
- Noisy copies: B's share of the error and of the second moment shrinks, so the split buys nothing.
