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
Pre-registered.
