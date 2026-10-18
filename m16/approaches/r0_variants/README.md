# T298: R0-family oracle floors — which point closure of the low inputs has the lowest bias floor across retrains?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · T292 (branch claude/t292-oracle-floor, `oracle_floor/`)

## Question
T292 found that the R0 point closure has RMS bias .51 pp over the 80 DEV2 retrains: keep the top-3 skeleton (positions
and values) exact and pin the 7 low inputs at their conditional mean μ3 = E[X | X < v3] = −φ(v3)/Φ(v3). Does another
point closure do clearly better?

## Pre-registration (2026-10-18, before any run)
**Protocol (diagnostic sampling, inductive; not an estimator).**
- Per model, draw x ~ N(0, I_10) (N = 500k, seed 298000 + model index). For each closure c, build the closure point
  x_c from x's skeleton and score ok_c = [network on x_c returns x's rank-2 position].
- Pair it with ok = [network on x is correct].
- b_m(c) = mean(ok_c − ok). This is unbiased for (closure accuracy − true accuracy), the same quantity as T292's
  E_S[ok_c(S) − q(S)], with s.e. sqrt(P(disagree)/N).
- This replaces T292's 256 inner low draws per skeleton by pairing on the same x. The difference is a variance
  reduction only; the target quantity is the same.
- Truths for the slope: DEV2 1e8-sample truths (claude/t192-gpu-cost-law hp2/dev2/truths.json, sha256 582c21cf..., as
  T292). Weights: hp2/dev2/weights.npz (sha256 of arrays 22d1c7c2..., as T292).

**Variants** (skeleton dimension = number of continuous values kept exact; positions are discrete in all cases):
- (a) **R0**: top-3 exact, 7 lows = μ3. Dimension 3.
- (b) **top-4**: top-4 exact, 6 lows = μ4 = −φ(v4)/Φ(v4). Dimension 4.
- (c) **median**: top-3 exact, 7 lows = Q(Φ(v3)/2). Dimension 3.
- (d) **time-aware conditional means**: under the exact law the 7 low inputs are i.i.d. N(0,1)|<v3 whatever the
  skeleton's positions, so every per-position conditional mean equals μ3, and (d) ≡ (a) exactly. It is reported as
  such. As a substantive time-aware alternative, **(d')**: the 7 lows at canonical spread values
  Q(Φ(v3)·(8−j)/8), j = 1..7 (T274a's v5a lower-rank rule). Two orientations, descending and ascending in time, give
  2 points per skeleton; ok_c is averaged over them. Dimension 3.
- (e) **R0 + latest low exact**: the free position with the largest time index keeps its true value; the other 6
  are at μ3. Dimension 4.

**Reported per variant:**
- RMS_m b_m (and noise-corrected), mean b, OLS slope of b_m (pp) on the truth accuracy (pp), max |b|;
- the bias split by region of x, as T292: blind = M2 <= 0; else soft tail = min(g12, g23) >= .05; else the 2~3 band
  if g23 <= g12, otherwise the 1~2 band.

**Go line (DRI):** the best variant must beat R0 by >= 20%, i.e. RMS <= 0.8 × RMS(a) from this run (≈ .41 pp if
(a) reproduces .51), **and** |slope| <= |slope(a)| from this run.
Sanity check: (a) should reproduce T292's R0 (RMS .51, mean +.22, slope −.046) within noise.

**Alternatives written first.**
1. R0's floor comes from the soft tail, where low inputs matter through crosstalk. No point closure of the lows
   fixes that, so all variants land within ±20%.
2. Pinning all 7 lows at one value (ties among the lows) is artificial. A spread closure (d') may be better, or worse,
   as v5a's +0.3..1.7 pp bias on LB3 suggests.
3. "Picks rank >= 4" errors (.88 pp mean, T292) need the 4th value: (b) should fix that part.

## Status
**Done. Go line PASSED by (b) top-4** (pre-registered). A post hoc combination (top-4 + spread lows) does much
better on DEV2 and on an out-of-panel check (LB3). Its selection is post hoc, so it needs an independent re-check.
All claims unverified.

## Results
Reproduce:
- `python3 run.py 500000` -> `results/biases.json`;
- `python3 summarize.py` -> `results/summary.json`;
- exploratory: `python3 run.py 500000 explore` and `python3 summarize.py results/biases_explore.json`;
- LB3 check: `PYTHONPATH=<alg_zoo shim> python3 lb3_check.py` -> `results/lb3_check.json`.

About 12 CPU-min in total. Per-model s.e. of b ≈ .03 pp.

### Pre-registered variants, 80 DEV2 retrains (truth mean 92.01%)
Region parts are mean / RMS over models.

| variant | skeleton dim | RMS pp | mean pp | slope pp/pp | max \|b\| | blind | 2~3 | 1~2 | tail |
|---|---|---|---|---|---|---|---|---|---|
| (a) R0 | 3 | **.515** | +.215 | −.043 | 1.89 | +.04 / .05 | +.12 / .19 | +.00 / .08 | +.05 / .31 |
| (b) top-4 | 4 | **.389** | +.291 | −.042 | 1.20 | +.02 / .03 | +.08 / .11 | +.04 / .06 | +.14 / .22 |
| (c) median | 3 | .561 | +.249 | −.047 | 2.00 | +.03 / .04 | +.14 / .22 | +.02 / .09 | +.06 / .34 |
| (d) per-position cond. mean | 3 | ≡ (a) (exchangeability) | | | | | | | |
| (d') spread lows | 3 | .453 | +.373 | −.027 | 1.04 | +.01 / .02 | +.17 / .19 | +.05 / .07 | +.15 / .21 |
| (e) R0 + latest low | 4 | .448 | +.114 | −.035 | 1.76 | +.04 / .04 | +.08 / .15 | −.01 / .08 | +.01 / .29 |

- (a) reproduces T292's R0: RMS .51, mean +.22, slope −.046.
- **Go line:** RMS <= .8 × .515 = .412 and |slope| <= .0433. **(b) passes**: .389, 24% better, slope .0416.
- Its spread across models falls from sd .47 to .26 pp; its mean offset grows a little (+.29).
- No other pre-registered variant passes. (d') and (e) are 12–13% better.

### Post hoc combinations (exploratory, chosen after seeing the table above)
| variant | dim | DEV2 RMS | mean | slope | max \|b\| | LB3 30 models: RMS / mean / slope / max |
|---|---|---|---|---|---|---|
| top-4 + latest low exact | 5 | .308 | +.210 | −.034 | 1.08 | .603 / +.472 / −.019 / 1.39 |
| **top-4 + spread lows** | **4** | **.244** | +.210 | **−.011** | **.66** | **.377 / +.290 / −.014 / .89** |
| for reference: R0 / top-4 on LB3 | | | | | | 1.318 / +.927 / −.063 / 2.65 ; .751 / +.592 / −.022 / 1.66 |

**top-4 + spread** has the 4th value exact and the 6 lowest at canonical spread values Q(Φ(v4)·(7−j)/7), j = 1..6. It
uses two time orientations, so 2 points per skeleton.
- On DEV2 its RMS is half R0's, the slope is 4x flatter, and the cross-model sd is only ~.12 pp. A systematic offset
  of +.21 pp remains.
- On LB3, an independent panel that includes noisy copies down to 57% accuracy (diagnostic only), the ordering holds:
  .38 vs R0 1.32 pp.

## Context added after completion (DRI, T297)
T297 (claude/t218-tie-tail-labels, `r0_integral/`) found that R0's error set is not a per-triple threshold set: its
boundary also moves with v3 and g12. So a deterministic R0 integral needs >= 20–40k FPE, far outside the cheap
regime. The DRI then stopped this line; its message arrived after these runs were complete.
- Consequence: a better point closure helps only if it also keeps the integral cheap.
- The winning variants here add an exact continuous input (top-4: dimension 4) and, for the post hoc best, 2 points
  per skeleton. That makes the T297 integral costlier.
- So these results lower the oracle bias floor but **do not rescue the R0 integration route** in the cheap regime.

## Claims
- R1 [measured] R0's floor reproduces with independent, paired code: .515 vs T292 .51.
- R2 [measured] Keeping the 4th value exact (b) beats R0 by 24% at the same slope. Pre-registered go: PASS.
- R3 [measured, post hoc] Top-4 exact + spread lows: RMS .244 (DEV2), .377 (LB3), slope ≈ −.01. Two point evaluations
  per skeleton, skeleton dimension 4.
- R4 [inferred] Most of R0's cross-model spread comes from collapsing all low inputs to one value (ties among the
  lows) plus losing the 4th value. Spreading the lows and keeping v4 removes about 75% of the spread. The remaining
  ~+.2 pp is a systematic offset: the closure point is slightly "easier" than real inputs (soft tail).
- **For T297:** the cost is one extra continuous skeleton dimension (3 → 4) and 2 points per skeleton.

## Alternatives not ruled out
- The post hoc pick (R3) is one of 2 combinations tried after the table. An independent re-check on another panel or
  fresh retrains (e.g. by T296's agent) is needed before using it.
- The +.2 pp offset is systematic. Correcting it deductively (not by fitting) is open.
- Other spread rules (expected order statistics instead of quantile midpoints, more orientations) were not tried.

## Compute accounting
Network forward passes only: per model 500k true + 6–8 closure points per input (pre-registered run) or 3 (explore).
About 4e8 forwards in total, about 12 CPU-min single-threaded. Diagnostic sampling; no estimator cost claim.
