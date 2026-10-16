# Results: testing "Neural Networks Are Already Clean" with CPU-only resources

All hypotheses, thresholds and the decision rule were committed in `PREREGISTRATION.md` before data
collection (commit a83076a), and the AlgZoo addenda before any AlgZoo neuron was inspected.

## Study 1: Qwen3-0.6B MLP neurons (pre-registered H1-H3)

Setup: 1.2M tokens of pile-10k; 84 random neuron poles plus 28 random-direction controls, blinded and shuffled
together. Ten explainer agents saw training examples only. Ten separate judge agents saw only an
explanation and 12 held-out snippets, and never saw activations. 30 judge items paired an explanation with
the wrong unit's snippets. Code: `collect.py`, `build_items.py`, `analyze.py`; numbers in
`results/summary.json`.

| hypothesis | result | pre-registered bar | verdict |
|---|---|---|---|
| H1: real neuron poles rated A ("one main job") | **60.7%** (95% CI 50-71%) | >= 60% supports, < 40% against | supports (borderline; CI reaches into inconclusive) |
| H2: A-rate of neurons minus A-rate of random directions | **+25 pp** (61% vs 36%; CI +5 to +45) | >= +30 pp | **fail** |
| H3: judge AUROC, held-out top vs random tokens | **0.938** (CI 0.92-0.96) | >= 0.85 | pass |
| H3: judge AUROC, mid-range (20-50% of max) vs random | **0.799** (CI 0.76-0.83) | >= 0.70 | pass |
| H3: matched minus mismatched-explanation control (top) | 0.938 - 0.500 = **+0.44** | >= +0.15 | pass |

**Mechanical output of the pre-registered rule: "refuted at small scale"** (rule: H2 or H3 failing gives
"refuted"). **Corrected reading after independent review: H2 is underpowered, so the verdict is
inconclusive.** The +30 pp bar lies inside the 95% CI of the difference (+5 to +45 pp); with n = 28 random
directions the s.e. is about 11 pp, so pass/fail is close to a coin flip. A Fisher exact test does show
neurons are rated "one job" more often than random directions (OR 2.8, p = 0.028).

What failed, precisely:
- Neurons *are* rated cleaner than random directions: the difference's CI excludes zero. They miss the
  pre-set 30 pp margin.
- **36% of random directions were also rated "one main job."** Caveat found in review: the random
  directions were **partly made of neurons**. One neuron supplies a median 29% of a direction's top
  activation (5 neurons: 69%), and the directions rated A were the most single-neuron-dominated (top-1 share
  0.45 vs 0.36 for C and 0.26 for D). So this is partly "random directions inherit a sparse neuron's job",
  not only explainers' willingness to find a story. A cleaner control would project out the top neurons
  or use a dense rotation.
- Category breakdown for neurons: A 61%, B 10%, C 30%, D 0%. For random directions: A 36%, B 7%, C 46%, D 11%.
  The essay reports A ~90%, B ~7%, not-understood 2-4% for Qwen3-32B. Our 0.6B result is lower, which the
  essay's own small-model caveat predicts.

Evidence in favour of a privileged neuron basis (secondary, not part of the verdict):
- Explanations of neurons predict mid-range activations much better than explanations of random directions
  (AUROC 0.80 vs 0.59). Top activations: 0.94 vs 0.86. **Caveat:** the marked token of a held-out top snippet
  appears verbatim in the explanation for 179/336 neuron snippets (62/112 for random directions), so much of
  the top-bin AUROC is token lookup.
- **Null result previously omitted:** the only LLM-free "one job" proxy, the share of a unit's top-100
  examples sharing its most common token, shows **no difference**: 0.12 for neurons vs 0.11 for random
  directions (p = 0.60).
- Population statistics over all 896 sampled neurons vs 112 random directions (`population.py`,
  `results/population.json`): activation kurtosis median **99.7 vs 0.8** (p ~ 1e-110). The share of activation mass
  at >= 20% of max is median **6% vs 38%**: neurons are extremely sparse. This shows **sparsity, not "one
  job"**. It is close to automatic for an elementwise nonlinearity versus a projection averaging 3072
  coordinates.
- The sparsity also qualifies the essay's method: **94% of a typical neuron's positive activation mass sits
  below 20% of its max.** Explanations built from top examples describe the rare strong firings. The judges
  show they also partly predict the 20-50% band (AUROC 0.80), but nothing here tests the weak bulk below
  that.

## Study 2: AlgZoo, legibility vs understanding (pre-registered E4)

Each model got a blinded investigator agent with code access. It wrote a label and a formula f_i(x) per
neuron (no access to the weights in the formula, at most 5 constants per neuron). Scored on held-out data
(`algzoo/score.py`).

| model | params | model acc | neurons legible (A/B, R^2 >= 0.8) | labels rebuilt -> accuracy | gap recovered | verdict |
|---|---|---|---|---|---|---|
| M_4,3 (positive control) | 32 | 98.4% | 4/4 | 98.5% | 100% | legible: pass, mechanism: pass |
| **M_16,10 (Hilton's challenge)** | **432** | 95.4% | **14/16** | **54.1%** | **52%** | **legible: pass, mechanism: FAIL** |
| M_32,10 | 1,376 | 99.4% | stopped (draft in `algzoo/labels_32_10.py`, unscored) | | | |

The protocol works when understanding is achievable (positive control). On the 432-parameter model the
neurons are individually legible, but the labels together lose about half the model's skill. **Caveat:**
this is largely driven by the 5-constants-per-neuron limit. A 216-state lookup of per-neuron linear maps on
the same features rebuilds 98%, and replacing even one neuron by its label drops accuracy to 0.79-0.95
despite R^2 ~0.9, because the readout amplifies small errors. It shows "short labels lose skill", not "no
compact description exists". The size ladder has one data point beyond the control (M_32,10 was not scored).

Deeper first-principles analysis of M_16,10 (`m16/FINDINGS.md`):
- In the most common gate pattern, the recurrence has damped rotating modes; the slowest mode's phase
  encodes how long ago the max arrived (measured -42.7 deg/position vs 42.8 predicted). **Corrected:** the
  "harmonic series / Fourier memory" reading does not survive robustness checks. That pattern covers only
  10-16% of states, other common patterns give non-harmonic periods, and the damping differs across modes. 13
  of the 16 neurons were ablated, in groups, with mean-ablation (n2, n5 and n9 were not). The "n3/n13 are
  harmonics" attribution used a non-orthogonal decomposition and is weak.
- Errors grouped by what the model picks: blind region (2nd largest <= 0) 0.82%, picks the 3rd 2.14%, picks
  the max 0.52% (when x9 is below both), plus a ~0.27% remainder picking rank >= 4. (The 99.96% accuracy on
  rank-canonical inputs is close to true by construction: canonical inputs never fall in the blind region.)
- ARC's open challenge (estimating accuracy mechanistically as well as sampling) is **not met**. Under ARC's
  noisy-weights protocol (`m16/arc_eval.py`) my best estimator has RMSE 2.8 pp vs sampling's 0.3 pp at equal
  compute. Six hypotheses that failed are documented.

## What this says about the essay

1. **The headline metric is weaker than it looks.** Blinded agents call 36% of *random directions*
   "one main job". A "90% of neurons are clean" figure needs a random-direction baseline to mean anything.
   The companion paper should report one.
2. **Neurons are sparse and somewhat more interpretable than random directions, but the evidence is
   weaker than first written.** Kurtosis shows sparsity, not single purpose. The judgment-free one-job proxy
   shows no difference. Judge AUROC is inflated by token overlap. The random-direction control was partly
   neuron-dominated. The "privileged basis" claim is plausible but not established by this study.
3. **At 0.6B, 61% of neuron poles are rated single-job, not 90%.** That is consistent with the essay's own
   statement that smaller models are broader; it neither confirms nor refutes the 32B number.
4. **Short legible labels do not add up to the mechanism.** On the 432-param model, 14/16 neurons get
   one-line labels, yet labels limited to 5 constants each rebuild only 54% accuracy (richer per-state maps
   reach 98%). Some descriptions with R^2 ~0.9 still get the error mechanism wrong. The essay's "ambitious
   interpretability will be solved" does not follow from per-neuron legibility.

## Limits

- 0.6B, not 32B or 1T. CPU-only, 1.2M tokens (the essay used 100M). Agents read examples but did not run
  experiments on Qwen.
- One explainer and one judge per item (no test-retest reliability of labels); agents are the same model family
  as the essay's tools. CIs ignore clustering by explainer batch (batch A-rates range 2/8 to 7/8).
- The judges never saw the 5-20% band, where most activation mass lies.
- Corrections in this file come from an independent review (`INDEX.md`, R1).
- Agents were told not to read the answer keys but were not sandboxed from them.
- The pre-registered rule makes H2 decisive. A reader may reasonably weigh H3 and the population statistics
  more heavily; both are reported in full.
