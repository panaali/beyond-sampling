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

## Addendum (2026-10-16, before any AlgZoo neuron was inspected): legibility vs. understanding

Prompted by Jacob Hilton (ARC), x.com/JacobHHilton/status/2110915993526821357: "have you interpreted my
432-parameter model yet?" ARC's bar for understanding is a mechanistic estimate of accuracy, not neuron
labels. The essay's claim is that neurons are *legible*. This tests whether, on a model small enough to
check exhaustively, legible neuron labels add up to the mechanism.

Model: AlgZoo `example_2nd_argmax()` = M_16,10, a 1-layer ReLU RNN, 16 hidden neurons, no biases,
432 parameters, input 10 i.i.d. N(0,1) numbers, output position of the 2nd largest. Measured accuracy 95.3%
(chance 10%).

- **E4a (the essay's method)**: an agent with code access investigates all 16 neurons of the final hidden state
  h_10 the way the essay's automated scientists do, and gives each a label and a category A-D (as above).
- **E4b (do the labels capture the mechanism?)**: for each neuron the agent also writes `f_i(x)`, a
  function of the input sequence implementing its label. It may use interpretable features of x (values,
  ranks, positions, running max, and so on) and at most 5 fitted scalar constants per neuron. It must not read
  W_hh or run the recurrence (audited by hand). Scored on 200k fresh sequences:
  (i) per-neuron R^2 of f_i vs the true h_10[i];
  (ii) **replacement accuracy**: argmax(W_oh @ [f_1..f_16](x)) vs the true 2nd argmax.
- Criteria: neurons count as *legible* if >= 12/16 are rated A or B and have R^2 >= 0.8. Labels
  *capture the mechanism* if replacement accuracy recovers >= 90% of the gap from chance to the model
  (>= 86.8%).
- Reading: legible but not capturing the mechanism supports Hilton (legible != understood); both passing
  supports the essay's "simple thing at scale works" at this scale. ARC's own criterion (a mechanistic
  accuracy estimate matching sampling) is out of scope and would not be met by either outcome.

### E4c (added 2026-10-16, before launching these investigators): a size ladder with a positive control

Same protocol and scorer (`algzoo/score.py --zoo h,n`), chance = 1/seq_len, criteria scaled: legible if
>= 75% of neurons are A/B with R^2 >= 0.8; mechanism captured if replacement accuracy recovers >= 90% of
the chance-to-model gap. Each model gets its own fresh investigator.

- **M_4,3** (`zoo_2nd_argmax(4, 3)`, 32 params, 98.4%): ARC reports understanding this one, so it is the
  **positive control**. If the protocol fails here, its failure on larger models is not informative.
- **M_16,10** (432 params, the challenge model): running since the addendum above.
- **M_32,10** (`zoo_2nd_argmax(32, 10)`, 1,376 params, 99.4%): the largest 2nd-argmax RNN.

Reading: the essay's thesis predicts gap recovery stays high as size grows. Hilton's view predicts it falls
off past the smallest models, even if per-neuron labels stay legible.
