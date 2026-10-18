# T332: paradigm-level information ceiling from T317e's own per-row outputs (Fable r5 settling test)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T314](../info_ceiling2/README.md)

## Question
Fable r5 §3 (`review/fable_review_r5/README.md` on claude/project-handoff-priorities-zpzfm8) asks whether any function
of what the R1 geometry measures can reach the strict bar. The test: take T317e's deductive estimate as a fixed
offset, and re-weight T317e's own per-row outputs with a fitted ridge. If the held-out RMS reaches ≤ .50 pp, the
information is present (F .79 at the 2.2k-FPE oracle cost), and a deductive mechanism behind the re-weighting is
worth looking for. If it stays ≥ .62 pp (F 1.22 at 2.2k), no function of these measurements reaches strict even at
the oracle cost; together with T326, strict is then closed for this paradigm (scoped to the measurement class).
**Diagnostic, and inductive by construction:** a fitted predictor is not an estimator ARC would accept.

## Pre-registration (2026-10-18, before any feature is computed)
**Code.**
- `cont_e.py` is a verbatim copy of T317e (claude/t218-tie-tail-labels @01906d33, sha256 `50dd1c9b…fda2b`).
- `features.py` reruns `cont_e.est`'s loop with read-only hooks and asserts its total equals `cont_e.est(model)[0]`
  bitwise. No per-row hook was needed from T218.
- The tie-margin readout recomputes `first_cell` on each line start: an extra prefix-tree pass, which already
  belongs to the walk's own cost.

**Per-row outputs.** Rows are the 720 position triples (p1, p2, p3), each with weight 1/720. Line types:
- 23nb: the g23 line, nonblind region;
- 23b: the g23 line, blind region;
- 12nb: the g12 line, nonblind region (the blind g12 is model-free and has no per-row output).

For each row and line type:
- **deficit** d = (full − actual) contribution to the estimate. For 23nb/23b, full is the integral with I23 at its
  all-correct value (Φ^8 profile) and the row's actual I12. For 12nb, full uses I12 all-correct (1 − Φ) and the row's
  actual I23. Each deficit is integrated against the region's density and Gauss-Legendre weights, exactly as in `est`.
- **tie margin** m = L_k − max_{j≠k} L_j at the line start (the tie point, t = EPS), with k = p2.
- **slope** s = dL_k − dL_{j*} there, with j* the rival at the tie.
- **cells** c = the number of walk cells the row used (≤ 400).

**The 30 features** (per model):

| group | features | count |
|---|---|---|
| deficit by line type × age of p2 | age(p2) = 9 − p2 in bins {0–1}, {2–4}, {5–9}; mean over rows of d (bin rows contribute, others 0) | 9 |
| deficit by line type × age of p3 | the same bins for p3 | 9 |
| tie margin, per line type | mean clip(m, −3, 3); fraction m < 0 | 6 |
| slope, per line type | mean clip(s, −30, 30) | 3 |
| cells, per line type | mean c | 3 |

**Learner (primary, decides the verdict).**
- Target: y = truth − est_T317e, in pp.
- Ridge with intercept on standardized features. The intercept is the fitted offset.
- Alpha from logspace(−3, 9, 13) by inner `KFold(5, shuffle=True, random_state=0)` on the 40 training models only.

**Data (T314 protocol).**
- Train: DEV2 fused_s0..39 (truths 1e8).
- Test: torch_s0..39 (truths 1e8) + LB3 seeds 0–4 (truths 32M), 45 models.

**Reported, not deciding:**
- the offset alone (train-mean residual), which Fable expects to give ~.69;
- raw T317e;
- the torch and seeds parts separately;
- a depth-2 GBT on the same 30 features (T314's settings: lr .05, n_est in {50, 100, 200, 400}, subsample .8, rs 0).

**Verdict (DRI / Fable).** Primary held-out RMS over the 45 test models:
- **≤ .50 pp → GO** (the information exists);
- **≥ .62 pp → STOP** (strict is closed for the paradigm, scoped);
- in between: neither.

**Alternatives first.**
1. With 40 training models, 30 features may overfit. Ridge's nested alpha guards against that, and the offset-only
   row is the floor to beat.
2. A fused → torch kernel shift may cap the held-out RMS regardless of the information present. The seeds row and
   the inner-CV RMS separate the two.
3. A GO could come from a feature that proxies accuracy itself (the bias is concentrated in low-accuracy models,
   T281 subset F). A GO would therefore be followed by an ablation by feature group before any mechanism is claimed.

## Status
Pre-registered; not run.
