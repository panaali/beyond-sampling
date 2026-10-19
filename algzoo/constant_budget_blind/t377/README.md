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

Done (2026-10-19). All 12 investigators and 32 judge calls completed: no missing or invalid judgments, no
run excluded. An independent rescorer reproduces every number and verdict (see below). Results are supported, not
confirmed (rule 4).

## Results

**Primary pre-registered verdict: fragility dominates. The margin is thin and the within-size evidence is mixed.**

### Models: fragility and reconstruction

Gap recovered comes from the shared scorer (200,000 held-out sequences, seed 20261016). Fragility comes from
[fragility.py](fragility.py) (seed 37701).

| Model | H | Model acc. | F (R² = 0.9 noise) | S1 | Account gaps | ḡ |
|---|---:|---:|---:|---:|---|---:|
| M8 s4 (T24) | 8 | 42.6% | 0.564 | 0.038 | 0.585, 0.560, 0.603 | 0.582 |
| M16 s2 (T24) | 16 | 94.9% | 0.173 | 0.202 | 0.305, 0.335, 0.361 | 0.333 |
| M16 s0 | 16 | 91.0% | 0.245 | 0.220 | 0.396, 0.187 | 0.291 |
| M16 s1 | 16 | 88.0% | 0.260 | 0.192 | 0.287, 0.262 | 0.274 |
| M16 example | 16 | 95.3% | 0.249 | 0.145 | 0.322, 0.523 | 0.423 |
| M32 s0 (T24) | 32 | 99.4% | 0.646 | 0.012 | 0.824, 0.778, 0.928 | 0.843 |
| M32 s1 | 32 | 99.3% | 0.613 | 0.015 | 0.831, 0.734 | 0.782 |
| M32 s2 | 32 | 99.1% | 0.452 | 0.034 | 0.492, 0.737 | 0.614 |

No account on any new model reaches the 90% criterion (best: `htmm`, M32 s1, 0.831). Every M16 account across
the 4 models stays below 0.53. Every M32 account except one (`incg`, 0.49) is at least 0.73.

### Pre-registered analysis

1. **Leave-one-model-out prediction of ḡ** (`results/summary377.json`):

   | Model set | RMSE, size only | RMSE, F only | RMSE, both | Verdict (0.7 rule) |
   |---|---:|---:|---:|---|
   | 7 models, M8 excluded (**primary**) | 0.111 | 0.075 | 0.104 | **fragility dominates** (ratio 0.67) |
   | all 8 | 0.293 | 0.093 | 0.070 | fragility dominates (ratio 0.32) |

   Including M8 is what makes the margin large. M8 is small but robust (F = 0.56, ḡ = 0.58), which a size
   model cannot fit. Without M8 the margin is just inside the 0.7 threshold.
2. **Within size:** the F spread is 0.087 among the M16 models and 0.195 among the M32 models, so neither is
   underpowered by the pre-registered rule. Spearman(F, ḡ) is **−0.40 among the 4 M16 models** and **+1.00
   among the 3 M32 models**. Within M32, the model with the lowest F (s2, 0.45) has the worst accounts.
   Within M16, F does not order the models; all of them are fragile (F 0.17–0.26), and with 2 accounts per
   model the account-to-account spread (up to 0.21 on one model) swamps the between-model differences.
3. **Account efficiency** e = gap / F: mean e is 1.03 (H = 8), 1.52 (H = 16) and 1.31 (H = 32). The slope per
   doubling of H is +0.07 (95% CI −0.21 to +0.21); without M8 it is −0.21 (−0.54 to +0.14). Once fragility is
   accounted for, there is **no detectable size trend** in how good the accounts are. Real accounts beat
   R² = 0.9 random noise (e > 1), because formula errors are not independent noise.

### Labels (secondary): T24's split between clean labels and poor reconstruction replicates

| New model | S matched | S control | D | Legible (judge) | ḡ | Split? |
|---|---:|---:|---:|---:|---:|:--:|
| M16 example | 0.969 | 0.188 | 0.781 | 84% | 0.423 | yes |
| M16 s0 | 0.875 | 0.188 | 0.688 | 70% | 0.291 | no (legibility 70% < 75%) |
| M16 s1 | 0.969 | 0.203 | 0.766 | 75% | 0.274 | yes |
| M32 s1 | 0.930 | 0.063 | 0.867 | 78% | 0.782 | — |
| M32 s2 | 0.961 | 0.063 | 0.898 | 75% | 0.614 | — |

2 of 3 new M16 models meet the frozen replication rule, so **the split replicates**. Independent labels agree
well at both sizes, and the control score is again lower for M32 (0.06) than for M16 (0.19–0.20).

### Arm (2): equal per-neuron budget on M32 s0

Two investigators with 160 calls: gaps 0.828 and 0.818, using about 53 and 45 tool calls (harness counts).
The T24 80-call accounts on the same model scored 0.824, 0.778 and 0.928. **Doubling the cap did not improve
reconstruction**, and neither investigator used more than about a third of its 160 calls.

### Per-account single-neuron sensitivity (shared scorer)

Replacing one true neuron by its formula costs on average 6–20 accuracy points on every M16 account (lowest on
the example model, 6.3 and 8.1) and 0.3–1.1 points on every M32 account. The worst M32 account (`incg`) also
has the largest single-neuron cost among M32 accounts (1.05).

### Independent rescoring

A separate sub-agent wrote its own code ([independent/](independent/REPORT.md)). It used input seed 4242 and
noise seed 777, and opened none of my scripts or results until its numbers were saved. Gaps, ḡ and F agree
within 0.005. All 5 D values are identical. The LOO verdicts agree: the 7-model ratio is 0.677 in its run vs
0.670 in mine, and the 8-model ratio 0.32 in both. The within-size Spearman values agree (−0.40 and +1.00). It
flags the same caveats: the 7-model margin is marginal, the 8-model result is driven by M8, and the within-M16
sign does not support fragility. It checks the scoring, not the investigators or judges.

### Audit

All 12 runs delivered complete `labels.py` files, with every declared constant count at most 5 and none of the
banned strings. The regex hits were benign: "weight" in comments or prose, `check.py` files searching
`labels.py` for the banned strings, and a numpy warning path in `phwb/s10.txt`. Self-reports: zero-padding
was used by several runs (allowed). `vnna` read its own background-script output in the harness task folder,
as `wfpn` did in T24, with no blinding leak. `htmm` uses an intercept (allowed; it breaks positive
homogeneity). Two M32 s2 neurons (16, 25) are near-silent, so their R² is poor in the shared scorer
(`vbya`: −0.001 for both; `incg`: 0.12 and 0.67). This lowers the R² ≥ 0.8 counts, not the gaps.

### Reproduce (from this folder, `source ../../../.venv/bin/activate`)

```bash
OMP_NUM_THREADS=1 python fragility.py
python pipeline.py <scratch>/t377 <scratch>/t377_assign.json collect   # needs the scratch dirs; outputs committed
OMP_NUM_THREADS=1 python pipeline.py <scratch>/t377 <scratch>/t377_assign.json score
python pipeline.py <scratch>/t377 <scratch>/t377_assign.json judging
python analyze377.py
```

## Compute accounting

| Role | Calls | Tool calls per call (harness) | Tokens per call |
|---|---:|---:|---:|
| Investigators (80-call cap) | 10 | 27–46 | 124k–172k |
| Investigators (160-call cap) | 2 | 45, 53 | 165k–174k |
| Role judges | 16 | 3 | ~54k |
| Pair judges | 16 | 3 | ~56k |
| Independent rescorer | 1 | — | — |

**Total: 45 sub-agent calls** (cap 80). No Fable, no reruns. Scoring: 12 shared-scorer runs. Fragility: 8
models × (200,000 sequences + 3 × (H + 1) noisy readouts at R² = 0.9 + 3 at R² = 0.8). All CPU, under 2 GB.

## Claims

- [measured] Across 8 AlgZoo 2nd-argmax models, a model-intrinsic fragility score F predicts the mean
  reconstruction of fresh blinded neuron accounts better than model size does. In leave-one-out tests, RMSE
  is 0.075 vs 0.111 without M8 and 0.093 vs 0.293 with it.
- [measured] Among the 3 M32 models, ḡ follows F exactly (Spearman 1.0). Among the 4 M16 models it does not
  (−0.4). All 4 M16 models are fragile (F ≤ 0.26), and none of their 10 accounts exceeds 0.53 gap recovered.
- [measured] T24's split between clean, agreed labels and poor reconstruction replicates on 2 of 3 new M16
  models.
- [measured] Doubling the investigation cap on M32 s0 (160 calls) gave gaps of 0.82–0.83, no better than
  80 calls.
- [inferred] Removing fragility leaves no detectable size trend in account quality (efficiency slope CI
  includes 0). So the "neuron account quality vs size" question in T24 is mostly a question about how
  fragile each trained model's readout is. That differs a lot between seeds of the same size (M32: F 0.45 to
  0.65).
- [inferred] For the essay question, naming neurons well is not enough. The same quality of naming yields
  reconstructions ranging from about 0.2 to 0.9 depending on a property of the readout, not of the labels.
- [speculative] Fragility is mostly a property of the training outcome (M16 models are fragile and M32
  robust here), so larger models may often be easier to account for neuron by neuron. Showing that needs more
  sizes and seeds.

## Alternatives not ruled out (declared before running)

1. F is model-intrinsic Gaussian noise; real formula errors are structured (correlated across neurons and
   inputs), so F may mis-rank the models. The secondary F80 and the per-account R² partly address this.
2. Size and fragility may stay correlated across seeds, so n = 7–8 models cannot separate them (handled by
   the "underpowered" rule).
3. 2 investigators per new model gives a noisy ḡ.
4. The example model was studied extensively elsewhere in this project, but fresh investigators are blinded to
   that.

Added after the results:

5. **Investigator variance:** on M32 s2 the two accounts differ by 0.25 (0.49 vs 0.74), as large as most
   between-model differences. With 2 accounts per new model, ḡ is noisy, and that limits the within-M16 test.
6. **Primary margin:** the 7-model ratio (0.67) is just under the 0.7 threshold, so a single different
   account could flip it to "not separable". The 8-model result depends on M8, a low-accuracy regime.
7. **Model accuracy as a third variable:** M16 models are 88–95% accurate and M32 models 99%. Accuracy,
   fragility and size are all correlated here.

## Failure modes

Same as T24. The Gaussian-noise fragility proxy assumes errors are independent across neurons.
