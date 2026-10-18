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
