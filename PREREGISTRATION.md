# Pre-registration: AlgZoo legibility versus understanding

This is an ARC-only edited extract of the historical protocol, with framing clarified during repository scope maintenance on 2026-10-20. The original numerical thresholds and procedures are retained; this edited wording is not represented as the original preregistered text. Original source bytes remain in Git history at `40f059fe09051164fdf04738b5b05e98c1271367` (same repository path).

## Addendum (2026-10-16, before any AlgZoo neuron was inspected): legibility vs. understanding

Prompted by Jacob Hilton (ARC), x.com/JacobHHilton/status/2110915993526821357: "have you interpreted my
432-parameter model yet?" ARC's bar for understanding is a mechanistic estimate of accuracy, not neuron
labels. This tests whether, on a model small enough to check exhaustively, legible neuron labels add up to the mechanism.

Model: AlgZoo `example_2nd_argmax()` = M_16,10, a 1-layer ReLU RNN, 16 hidden neurons, no biases,
432 parameters, input 10 i.i.d. N(0,1) numbers, output position of the 2nd largest. Measured accuracy 95.3%
(chance 10%).

- **E4a (neuron legibility)**: an agent with code access investigates all 16 neurons of the final hidden state
  h_10 and gives each a label and a category: A = one main job, B = two related jobs, C = several unrelated jobs, D = not understood.
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
  shows that this constrained description captures behaviour at this scale. ARC's own criterion (a mechanistic
  accuracy estimate matching sampling) is out of scope and would not be met by either outcome.

### E4c (added 2026-10-16, before launching these investigators): a size ladder with a positive control

Same protocol and scorer (`algzoo/score.py --zoo h,n`), chance = 1/seq_len, criteria scaled: legible if
>= 75% of neurons are A/B with R^2 >= 0.8; mechanism captured if replacement accuracy recovers >= 90% of
the chance-to-model gap. Each model gets its own fresh investigator.

- **M_4,3** (`zoo_2nd_argmax(4, 3)`, 32 params, 98.4%): ARC reports understanding this one, so it is the
  **positive control**. If the protocol fails here, its failure on larger models is not informative.
- **M_16,10** (432 params, the challenge model): running since the addendum above.
- **M_32,10** (`zoo_2nd_argmax(32, 10)`, 1,376 params, 99.4%): the largest 2nd-argmax RNN.

Reading: test whether gap recovery stays high as size grows or falls off past the smallest models, even if per-neuron labels stay legible.
