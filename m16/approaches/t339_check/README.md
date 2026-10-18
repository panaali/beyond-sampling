# t339_check — T343: independent check of T339's spread-lows closure on a fresh panel (DEV3)

↑ [TASKS](../../../TASKS.md) · T339 (claude/t286-cheap-frontier c0564f03, closure_floor/README "T339") · T317f (claude/t218-tie-tail-labels @88453669)

**All claims unverified. A different agent on different data; own scoring code (`check.py`), frozen estimator sources.**

## Question
T339 replaced T317f's flat low pattern (all lows = mu) with a spread pattern, low_j = mu + s(-1)^j sigma. On DEV2 at s = -1:
- the model-to-model error sd fell .667 -> .455 pp;
- corr(est, truth) rose .978 -> .991;
- the bias rose by a near-constant +.27 pp (to +.758).

So the variance-only F was ~1.06 at 3.7k FPE.

Does this replicate on DEV3, a fresh open panel (T342, 100 models) that neither T339 nor I have seen?

## Pre-registration (written 2026-10-18, before DEV3's truths exist or are opened)
**Sources.** Hashes are checked in code; the run aborts on mismatch.
- `cont_f_t317f.py` sha256 327dbb4a… = claude/t218-tie-tail-labels @88453669 r1_cont/cont_f.py; estimator_f.py 3afa3912…
  also matches.
- `cont_g.py` sha256 d34dde36…, from claude/t286-cheap-frontier @c0564f03.

**Arms:** T317f (frozen); cont_g with s = 0, +1, -1. Check: s = 0 is bitwise T317f on every model.

**Per arm on all 100 DEV3 models:**
- RMS (noise-corrected by the truths' SE^2, as T339; raw also shown), bias, error sd, corr(est, truth);
- C = mean FLOPs as executed / 5948, with its range;
- F = MSE · max(C, 1000) / mean p(1-p), and F_var = var(error) · max(C, 1000) / mean p(1-p).

**Shift** (s = ±1 minus s = 0):
- mean, sd and corr with (1 - p);
- the affine regression on (1 - p) and its R^2;
- the regression on (1 - p) plus T317f's error e0, with R^2 and corr(shift, e0);
- the mean by kernel.

**Replication criteria (for s = -1, the T339 headline), each reported pass/fail:**
- R1: error sd(s=-1) / error sd(s=0) <= 0.80 (DEV2 0.68).
- R2: mean shift in [+0.15, +0.40] pp (DEV2 +0.27) and abs(corr(shift, 1-p)) < 0.3 (DEV2 -0.03), i.e. near-constant.
- R3: corr(est, truth) at s = -1 > corr at s = 0.
- R4: F_var(s = -1) <= 1.5 (DEV2 ~1.06).
- R5: corr(shift, e0) <= -0.4 (DEV2 -0.73): the shift corrects T317f's own per-model error.

The T339 finding "replicates" if R1–R3 pass. R4 and R5 are reported only. Also run on DEV2 as a harness check: the T317f and
s = ±1 estimates must equal T339's results/t339.json rows exactly.

**Alternatives written first.**
- (a) T339's selection of s = -1 over +1 was made on DEV2. On fresh data, the s = -1 advantage over +1 may shrink: a
  regression to the mean.
- (b) The near-constant +.27 pp shift could depend on the panel mix (fused/torch kernels, training lengths). If DEV3's mix
  differs, the shift may differ; reported by kernel.
- (c) The error-sd reduction could come from DEV2's particular hard models. If so it shrinks on DEV3.

## Status
Done: **does not replicate** under the pre-registered rule (R1 fails). Unverified.

## Results (unverified)
Data:
- DEV3 = 100 open models, dev3_s500..599. Weights claude/t192-gpu-cost-law @e2322a6b, sha256 62c95203…. Truths @8134ab47,
  sha256 abcb7d6d…, 1e8 samples each.
- All 400 estimates were computed and committed (b72fcc06, dev3.json sha256 7882918a…) before the truths were published.
- s = 0 is bitwise T317f on 100/100.
- Commands: `python check.py SRC dev3/weights.npz - dev3`, then `python check.py SRC dev3/weights.npz dev3/truths.json dev3`.
  This produces dev3.json and dev3_summary.json.

| arm (DEV3, 100) | RMS pp | bias pp | error sd pp | corr | C (FPE, range) | F | F_var |
|---|---|---|---|---|---|---|---|
| T317f = s 0 | **.721** | +.425 | .583 | .9775 | 3,307 [2,667, 4,216] | **2.49** | 1.63 |
| s = +1 | .791 | +.600 | .515 | .9828 | 3,945 [2,820, 5,939] | 3.58 | 1.52 |
| s = -1 | .836 | +.645 | .532 | .9814 | 3,722 [2,737, 5,574] | 3.77 | 1.53 |
| *DEV2 s = 0 / -1 (T339, reproduced)* | .827 / .884 | +.489 / +.758 | .667 / .455 | .978 / .991 | 3,326 / 3,748 | 3.14 / 4.04 | 2.04 / 1.07 |

**Shift** (s minus s=0), DEV3:

| s | mean | sd | corr(shift, 1-p) | affine on (1-p) | corr(shift, e0) | regression on (1-p, e0), R^2 |
|---|---|---|---|---|---|---|
| -1 | +.221 pp | .527 | +.01 | +.206 + .002 (1-p), R^2 .00 | -.545 | .313 |
| +1 | +.176 pp | .578 | +.06 | +.075 + .013 (1-p), R^2 .00 | -.606 | .406 |

**Pre-registered criteria (s = -1):**

| criterion | DEV3 | DEV2 (T339) | verdict |
|---|---|---|---|
| R1 error-sd ratio <= .80 | **.912** (bootstrap 95% CI [.76, 1.11]) | .68 | **FAIL** |
| R2 mean shift in [.15, .40] pp and abs(corr with 1-p) < .3 | +.221, +.01 | +.269, -.03 | pass |
| R3 corr rises | .9775 -> .9814 | .978 -> .991 | pass |
| R4 F_var <= 1.5 (reported) | 1.53 | 1.07 | fail (marginal) |
| R5 corr(shift, e0) <= -.4 (reported) | -.545 | -.73 | pass |

## Claims
- [measured] **The bias half of T339 replicates.** The spread closure adds a near-constant positive shift (+.22 pp at s = -1,
  uncorrelated with 1-p) that partly corrects T317f's own per-model error (corr -.55).
- [measured] **The variance half does not replicate at the pre-registered size.** The error sd falls only 9% at s = -1
  (.583 -> .532), against 32% on DEV2. The CI includes no reduction. F_var is 1.53, not ~1.06. So "variance alone gives
  F ~ 1" was a DEV2 finding, not a property of the method.
- [measured] **The sign choice did not carry over.** On DEV3, s = +1 is as good as or slightly better than s = -1 (sd .515 vs
  .532, F 3.58 vs 3.77). This fits alternative (a), selection on DEV2.
- [measured] **T317f (s = 0) stays the best arm on DEV3** by RMS and F (.721, F 2.49), as T339 found on DEV2.
- [inferred] DEV2's large sd reduction was partly panel-specific: DEV2's s=0 error sd was larger (.667 vs .583), leaving more
  to correct. The strict route still needs a deductive bias fix. A bias fix alone would leave F_var ~1.5–1.6 on DEV3 for
  every arm, so a variance cut is also needed.

## Alternatives not ruled out
- **DEV3 is one config:** 2^30 zoo, a single trainer/kernel (names dev3_*), unlike DEV2's fused+torch mix. Kernel-specific
  effects could not be separated.
- **Uncertainty:** 100 models gives the sd ratio a ±.17 CI. A ratio of ~.8 is not excluded at the upper end of T339's
  effect.

## Compute accounting
DEV3 estimates: 100 models x 4 arms, about 6 CPU-min. DEV2 harness reproduction about 2 CPU-min. Scoring is negligible.
