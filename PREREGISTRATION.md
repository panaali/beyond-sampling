# Pre-registration: testing "Neural Networks Are Already Clean" at small scale

Written and committed **before** any activations were collected or any neuron was inspected.

## Target claim

Cammarata (Goodfire, 2026-10-15), *Neural Networks Are Already Clean*:
"In preliminary audits of Qwen3-32B, we estimate that roughly 90% of neuron poles have one main
job, often with small related offshoots; another ~7% have two related jobs; and only ~2-4% are not
yet understood." The essay also says smaller models have broader neurons (old GPT-2 XL re-audit: ~70%
"roughly one thing"), and that polysemanticity was mostly not superposition. The companion paper with
the methods is not released.

## Resources and what they can and cannot decide

4 CPU cores, 15 GB RAM, no GPU, no LLM API key; Claude subagents act as the "automated scientists".
So we **cannot** audit Qwen3-32B or Kimi K3. We test the same claim, with a protocol modelled on the essay's,
on **Qwen3-0.6B** (same family, same SwiGLU MLP neurons, 28 layers x 3072 neurons) and add controls
the essay does not report. A failure at 0.6B is weak evidence against the 32B number (the essay predicts
small models are worse); a pass at 0.6B, or a failure of the controls, is informative.

## Setup (fixed in advance)

- Neuron = MLP hidden unit, i.e. one coordinate of the input to `down_proj`: `silu(gate) * up`.
  A *pole* is its positive or its negative side (the essay's unit).
- Corpus: `NeelNanda/pile-10k`, 64-token windows, at most 4 windows per document, ~1.2M tokens.
  Positions 0-3 of each window are excluded from all statistics (attention-sink artifacts).
- Sample: seed 0. 32 random neurons per layer stored (896 neurons) for automated metrics.
  Agent study: 3 random neurons per layer (84), one random pole each, **plus 1 random-direction
  pseudo-neuron per layer (28)**, a unit Gaussian direction in the same 3072-d activation space, random pole.
  All 112 items are shuffled and blinded; agents are not told that controls exist.
- Explainer agent sees, per pole: 20 top-activating examples (even ranks of the top 40, distinct
  windows), 8 examples at 20-50% of max, 6 at 5-20% of max, each with the target token marked and its
  activation as % of max. It writes a one-sentence explanation and a category:
  A = one main job (possibly with related offshoots), B = two related jobs, C = several unrelated jobs,
  D = not understood.
- Judge agent (a different agent; it never sees activations) gets the explanation plus 12 held-out
  snippets in random order: 4 from the held-out top 40 (odd ranks), 4 at 20-50% of max (not shown to
  the explainer), 4 random positions. It rates 0-3 how strongly the marked token should fire.
  Control: 30 extra judge items pair an explanation with a *different* pole's snippets.

## Hypotheses and decision rule

Clean-neuron view (the essay) predicts:

- **H1 (headline replication)**: share of real neuron poles rated A.
  >= 60% supports; 40-60% is inconclusive; < 40% counts against (at 0.6B).
- **H2 (privileged basis)**: real neurons rated A at least 30 percentage points more often than
  random directions. If random directions are rated "one job" about as often, the rating method can't
  tell neurons from arbitrary directions, and an A-rate like "90%" is not evidence that neurons are clean.
- **H3 (explanations are predictive, not just plausible)**: judge AUROC, held-out top vs random >= 0.85,
  and mid-range (20-50%) vs random >= 0.70, for real neurons; and the matched explanation beats the
  mismatched control by >= 0.15 AUROC. Mid-range is where superposition would show up: if one-job
  explanations only cover the top activations, the essay's claim reduces to "top activations look clean".

Overall verdict: **supported at small scale** if H1, H2 and H3 all pass; **refuted at small scale** if
H2 fails or H3 fails (method can't separate neurons from random directions, or labels don't predict);
otherwise **mixed**. H1 alone failing is reported as consistent with the essay's own small-model caveat.

Descriptive (no threshold): share of each pole's total activation mass coming from tokens above 20% of
its max, for neurons vs random directions (how much of a neuron's output lives where explainers look).
