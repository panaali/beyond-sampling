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
Pre-registered.
