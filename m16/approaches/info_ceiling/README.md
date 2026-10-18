# T308: information-ceiling test (Fable r3 B1): how much do 1–2 evaluations per class know about accuracy?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · T300/T302 (claude/t218-tie-tail-labels) · T286 (claude/t286-cheap-frontier)

## Question
Every cheap evaluation-based design sees a model only through a few network evaluations per position class. Fable r3
B1: build <= 10 universal summaries of what the T300/T302 anchors compute, fit a <= 5-parameter linear predictor of
accuracy on 40 DEV2 models, and test it on the other 40 and on the 5 LB3 seeds. A low held-out RMS means the
information is present (worth deriving deductively). A high RMS bounds every such design.
**Diagnostic and inductive by construction** (a regression fitted on models).

## Pre-registration (2026-10-18, before any feature is computed)
**Per class** (the 720 ordered top-3 triples), at T300's model-free anchor: v2a = median of v2 | v2 > 0,
v1 = v2a + median g12 (`tb.MF`, imported unchanged from claude/t218-tie-tail-labels `r0_integral/tb.py`).
- `m_tie`, `dm_tie`: R0 margin L_p2 − max_{j≠p2} L_j and its g23 tangent at g23 = 1e-6 (as `tb.root23`'s first step).
- `tau`: T300's one-Newton-step threshold (`tb.est(n23=1, newton=1, nblind=0, use12=False)`), and its per-class
  correct mass M.
- `m_med`: R0 margin at the median g23 (v3 = v2a − median g23). **R0 verdict** = m_med > 0.
- `J`: norm of ∂m_med/∂(7 low inputs), from 7 forward-mode tangents with frozen gates (new code; T300/T302 do not
  compute it).
- gate pattern: number of active gates over the 10 steps at the median anchor (recorded; not used in summaries).

**The 10 universal summaries (fixed now):**
- F1 = T300's accuracy estimate (threshold-mass sum, blind assumed wrong);
- F2, F3, F4 = mean over classes of Φ(−m_med/s), for s = 0.5, 2, 8 logits;
- F5 = fraction of classes with m_med < 0 (R0 verdict wrong at the median anchor);
- F6 = fraction of classes with m_tie < 0;
- F7 = mean over classes of log(1 + J/|m_med|);
- F8 = mean over classes of Φ(−|m_med| / (J·σ_low)) · sign-adjusted (the probability that the low inputs flip the
  verdict, with σ_low the model-free s.d. of a low input given v3 at the anchor); precisely
  Φ(−m_med/(J·σ_low)) if m_med > 0 else 1 − Φ(|m_med|/(J·σ_low));
- F9 = rank_exact output (T286 frozen config rx Q=8, stride 1, kgroups 3; stored per model in
  `results/sweep_raw_dev2.jsonl` / `sweep_raw.jsonl`);
- F10 = probe-v1 output (T286 probe version 1, iters 8, stride 1; stored the same way).

**Predictor:** accuracy (pp) = β0 + Σ βi Fi with <= 4 features (<= 5 parameters).
- The subset is chosen among all 385 subsets of size 1–4 by leave-one-out CV RMS on the **training models only**
  (DEV2 indices 0–39 = fused_s0..39).
- Then OLS on all 40.
- Test on DEV2 40–79 (torch_s0..39) and the 5 LB3 seeds (truths: DEV2 1e8 samples; LB3 32M).
- Also reported: the pre-specified 2-feature model (F9, F10), i.e. the two opposite-sign estimators.

**Go line (DRI):** held-out RMS (all 45 test models) <= .6 pp → go (fund a deductive derivation). **Stop:** > 1.0 pp →
close cheap strict for evaluation-based designs (scoped to these summaries). In between: neither.

**Alternatives first.**
1. The DEV2 train/test split is by trainer (fused vs torch), so trainer differences could inflate the held-out error
   without any lack of information. Check: the RMS on torch vs on the seeds, and the train-LOO RMS.
2. The two estimators' errors may be highly correlated (both ρ ≈ .96 with the truth), so combining them helps
   little (Fable: the combined sd sqrt(.82 + .77ρ) reaches .43 only if ρ <= −.8).
3. Overfitting by subset selection: guarded by train-only LOO selection.

## Status
**Done. Pre-registered GO:** held-out RMS .477 pp, under the .6 line. But the selected predictor uses rank_exact
(25.8k FPE). Restricted to cheap features (post hoc), the RMS is .72 pp: neither go nor stop. All claims unverified;
diagnostic and inductive by construction.

## Results
Reproduce:
- `PYTHONPATH=<alg_zoo shim> python3 features.py` (85 models, about 3 CPU-min) -> `results/features.json`;
- `python3 fit.py` -> `results/fit.json`;
- `python3 cheap_only.py` (exploratory).

The anchor is v1 = 1.62, v2 = 0.96, v3 = 0.62 (T300's model-free medians); σ_low = .70.

| predictor | features | params | train LOO RMS | train fit | **held-out RMS (45)** | torch_s (40) | LB3 seeds (5) | held-out mean err |
|---|---|---|---|---|---|---|---|---|
| **selected (pre-registered rule)** | F4 Φ(−m/8), F8 low-flip, F9 rank_exact, F10 probe-v1 | 5 | .523 | .474 | **.477** | .473 | .513 | +.07 |
| pre-specified two estimators | F9, F10 | 3 | .630 | .595 | .631 | .617 | .736 | +.15 |
| best single feature | F10 probe-v1 | 2 | .871 | .833 | .848 | .870 | .645 | +.06 |
| cheap only (post hoc, no rank_exact) | F1 T300, F4, F10 | 4 | .754 | .690 | .720 | .746 | .466 | +.06 |
| raw, unfitted: T300 / rank_exact / probe-v1 | – | 0 | | | 1.54 / 1.58 / 3.67 | | | +.35 / −1.25 / +3.28 |

The other single features have held-out RMS 1.34–3.12 pp. The full table is in `logs/fit.log`.

**Correlation of the rank_exact and probe-v1 errors** (85 models): **−.26**. They are not strongly anti-correlated, so
the two alone give .63 (Fable predicted reaching .43 needs ρ <= −.8). The anchor margin features add the rest.

**Verdict against the pre-registration:**
- **GO** (.477 <= .6).
- No sign of overfitting: train LOO .523 ≈ test .477, and the selection used the training models only.
- The trainer split (fused → torch) does not hurt (alternative 1): torch .47, seeds .51.

**Scope and cost of the information (important for a deductive follow-up):**
- F9 rank_exact costs 25.8k FPE; F10 probe-v1 ~3.9k FPE.
- F1 (T300 thresholds) ~1.8k FPE; F2–F6 (one anchor verdict / margin per class) 720 FPE.
- F7/F8 (7 low tangents per class) ~5k FPE.
- Without rank_exact the best is .72 pp at about 6.4k FPE (F1, F4, F10). That lies between the lines, so the
  information within the cheap regime is **not** shown to reach .6 pp.
- What reaches .48 pp is a mix that includes a 25.8k-FPE estimator.

## Claims
- I1 [measured] A <= 5-parameter linear combination of four universal summaries (anchor margin spread, low-flip
  probability, rank_exact, probe-v1), fitted on 40 fused retrains, predicts the accuracy of 40 torch retrains and the
  5 LB3 seeds with RMS .477 pp. Pre-registered go.
- I2 [measured] The two opposite-sign estimators alone give .63 pp; their errors correlate −.26.
- I3 [measured, post hoc] With features costing <= ~6.4k FPE only (no rank_exact), the best is .72 pp.
- I4 [inferred] The model-specific accuracy information is largely present in a few evaluations per class plus one
  costlier estimator. A deductive derivation of the 5 coefficients is worth funding **only** if rank_exact's cost can
  be cut, or its information obtained from cheaper anchor quantities.

## Alternatives not ruled out
- The coefficients are fitted (inductive). Whether they are stable across trainers is supported (fused → torch),
  but not across noise families (LB3 noisy copies were not tested).
- Other summary choices (scales, margin cutoffs) were fixed in advance and not tuned. Tuning would overfit.

## Compute accounting
Features: T300 `tb.est` (~1.8k FPE) + 720 × (1 + 7 tangents + 1 tie) evaluations per model, 85 models, about 3 CPU-min.
rank_exact and probe-v1 values were read from T286's stored rows (not recomputed). Fits are negligible.
