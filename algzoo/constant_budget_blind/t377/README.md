# T377: is it model size or readout fragility? (T24 follow-up)

↑ [T24 README](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

## Question

T24 found that labels and reconstruction come apart. M_16,10 had the most legible and most agreed labels
but the worst reconstruction: replacing one true neuron by its formula cost about 12 accuracy points there,
against about 0.3 on M_32,10. **Across several trained seeds per size, is the reconstruction a neuron account
achieves explained by model size, or by a model-intrinsic measure of how fragile the readout is to small
activation errors?** A second question: do T24's label findings replicate on new seeds?

## Method (pre-registered 2026-10-19, before any T377 investigator ran and before any fragility number was computed)

### Models

Rule: the two lowest-numbered non-default AlgZoo seeds at each size H = 16, 32 (T = 10), plus ARC's challenge
model `example_2nd_argmax()` (H = 16) as a third new M16. Accuracies (measured while choosing, 100,000
sequences, seed 5): M16 s0 91.05%, s1 87.94%; M32 s1 99.34%, s2 99.10%; example about 95.3%.

- **No accurate small-H model exists at T = 10 in AlgZoo.** The best 8-neuron seed reaches 42.6% and the best
  6-neuron seed 37.4%. Retraining H = 8 is out of scope; AlgZoo's best-of-5 suggests 8 neurons cannot do this
  task well. M_8,10 (T24) stays in the analysis as a separate low-accuracy regime, and every result is also
  reported excluding it.
- Analysis set: **8 models**. T24's M8 s4, M16 s2 and M32 s0 (the defaults, 3 accounts each), plus the 5 new
  models (2 accounts each).

### Investigators and judges (frozen T24 protocol, unchanged)

- **2 fresh investigators per new model** (10 in total), each given the identical
  [../prompts/investigator.md](../prompts/investigator.md) prompt, 80 tool calls, probe-only access through
  [../probe_template.py](../probe_template.py) (only the model loader line changes), and blinding and audit as in
  T24. Random 4-letter codes; all run concurrently.
- **Judges:** the same frozen role and pair prompts. Items come from the 5 new models only, pooled across sizes
  and shuffled (seed 3770). One matched pair per neuron (the two investigators) plus one derangement control
  per matched pair, in batches of 28, each batch judged by 2 independent calls.
- **Scoring:** the unchanged shared scorer `algzoo/score.py`. For the example model it is run without
  `--zoo`, which loads `example_2nd_argmax`. Each zoo seed is scored on the same 200,000 held-out sequences
  through a wrapper ([score_seed.py](score_seed.py)) that only swaps in the seeded model, so the formula and
  metric code is unchanged.

### Fragility (model-intrinsic, no investigator involved) — [fragility.py](fragility.py)

For each of the 8 models, on 200,000 fresh sequences (seed 37701):
- **F** = gap recovered when every neuron's true activation gets independent Gaussian noise at the R² = 0.9 level
  (noise variance 0.1·var(h_i)), averaged over 3 draws. F80 is the same at R² = 0.8.
- **S1** = mean single-neuron accuracy drop under that noise.

### Pre-registered analysis: separating size from fragility

Unit of analysis: the model (n = 8; n = 7 excluding M8). Outcome: **ḡ**, the mean gap recovered over that
model's fresh accounts (shared scorer).

1. **Leave-one-model-out prediction.** Fit three linear models of ḡ: (a) on log2 H, (b) on F, (c) on both.
   Report LOO RMSE for each.
   - **Fragility dominates** if RMSE_b ≤ 0.7·RMSE_a.
   - **Size dominates** if RMSE_a ≤ 0.7·RMSE_b.
   - Otherwise **not separable**.
   The primary verdict uses the 7 models excluding M8, where all models are at least 88% accurate. The 8-model
   version is secondary.
2. **Within-size test (the decisive one for separation).** Among the 4 M16 models (size fixed), the Spearman
   correlation between F and ḡ, and the same among the 3 M32 models (descriptive, n = 3). If the spread of F
   within a size is < 0.05, separation is declared **underpowered** whatever step 1 says.
3. **Account efficiency.** Per account, e = gap recovered / F. If the accounts are equally good at every size
   once fragility is removed, e should not depend on log2 H. Report the slope with a bootstrap CI over accounts.

Label replication (secondary): per new model, D (matched minus control pair score) and legibility as in T24.
T24's dissociation *replicates* if at least 2 of the 3 new M16 models have D ≥ 0.5 and judge legibility
≥ 0.75, and mean gap recovered < 0.6.

### Arm (2): equal-per-neuron budget (within the cap)

2 fresh investigators on T24's M32 default with a **160**-call budget (twice T24's; the same per-neuron budget as
80 calls on M16). Reconstruction is reported against T24's 80-call M32 accounts (0.778 / 0.824 / 0.928).
Descriptive only; these accounts are not judged and do not enter the analysis above.

Arm (3), a stricter pair rubric, is **not run** (kept for the cap; S matched near ceiling remains a T24 caveat).

### Sub-agent budget

10 + 2 investigators, about 32 judge calls (role: 224 labels → 8 batches × 2; pair: 224 items → 8 × 2), and 1
independent rescorer: **about 45 calls**, plus up to 15 for reruns. Cap 80. No Fable.

## Status

Pre-registered; nothing run yet.

## Results / Compute accounting / Claims

To be filled.

## Alternatives not ruled out (declared before running)

1. F is model-intrinsic Gaussian noise; real formula errors are structured (correlated across neurons and
   inputs), so F may mis-rank the models. The secondary F80 and the per-account R² partly address this.
2. Size and fragility may stay correlated across seeds, so n = 7–8 models cannot separate them (handled by
   the "underpowered" rule).
3. 2 investigators per new model gives a noisy ḡ.
4. The example model was studied extensively elsewhere in this project, but fresh investigators are blinded to
   that.

## Failure modes

Same as T24. The Gaussian-noise fragility proxy assumes errors are independent across neurons.
