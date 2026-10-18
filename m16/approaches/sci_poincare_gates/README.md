# sci_poincare_gates: gate-switch symbols of the clock neurons (T3)

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [E27 sci_poincare](../sci_poincare/README.md)

## Question
E27 built a symbolic dynamics over **record events** (relative ranks + sign) and failed its ceiling test: the
reduced system "permutation + top-k values, low inputs frozen at typical values" is biased +0.07 to +0.75 pp,
always upward, because the low inputs are never forgotten. E27 also found that what record symbols cannot
predict sits in the gate switches of 1-3 rotating ("clock") neurons. Its untried variant: **use those
neurons' ReLU gate switches as the symbols.** If the switch itinerary carries the information about the low
inputs that matters, a reduced system "clock itinerary + top-k values" should have a ceiling bias near zero.

## Method (pre-registered 2026-10-18, before any scored run)

**Symbols.** For a set C of clock neurons, the letter at step t is the gate vector
g_t = [pre-activation of neuron j > 0 for j in C] (alphabet 2^|C|). The itinerary S(x) = (g_1..g_10) is
equivalent to the sequence of switch events (h_0 = 0 fixes the start). These are the crossings of the ReLU
hyperplanes, i.e. a Poincare section per clock neuron.

**Reduced system (built like E27's).** Input x; rank > k inputs are "low". As in E27 `ceiling2.py`, x_r = x with
the low inputs replaced by deterministic values given the k-th value c (V1: expected order statistics in their
actual order; V2: all at E[N(0,1) | below c]). The reduced network runs on x_r, **but the neurons in C use the
masks of the true itinerary S(x)**: h_j = 1[g_t,j] * pre_j (linear inside the symbol cylinder) instead of
ReLU(pre_j). All other neurons are ReLU as usual. Reduced correct = argmax == true 2nd-argmax of x. The reduced
system depends on x only through (S(x), top-k values and positions, order of the lows), so its accuracy is
exactly what a perfect integration over that symbolic + continuous state could return: the **ceiling**.

**Bias** = acc(reduced) - acc(true), paired on the same 600k N(0,1) inputs per model (rng seed 7, as E27).
The s.e. of a paired difference is sqrt(flip rate / N), reported per entry (~0.01-0.03 pp).

**Clock sets C** (rules fixed in `models.py`, identities printed by `python models.py`):
- `D3` (**primary**): top 3 neurons by E27's score of gate flips not predicted by (G_{t-1}, capped rank, sign),
  on an independent 200k sample (seed 5). Orig gives {3, 15, 11}, matching E27.
- `D5`: top 5 by the same score.
- `W3`: weight-only rule, top 3 by participation in complex eigenvectors of W_hh with |lambda| >= 0.5,
  weighted by |lambda|^10. (It disagrees with D3 on every model: orig {12, 11, 0}.)

**Controls (same samples):**
- `none`: E27's reduction, no forcing (reproduces E27 on its 10 models; the baseline).
- `R3`: 3 random non-D5 neurons per model (fixed rng seed 11) with forced true masks: are clock symbols special,
  or do any forced gates carry low-input information?
- `X3` (shuffled symbol): C = D3, but masks taken from S(x'), where x' keeps x's top-k values and positions and
  resamples the lows i.i.d. from N(0,1) truncated below c. Same linearisation, no information about the actual
  lows.
- `A16`: all 16 neurons forced to the true masks (the network becomes linear in the lows given S).

Grid: k in {3, 4, 5} x {V1, V2} x {none, D3, D5, W3, R3, X3, A16}.

**Models (18).** orig; zoo seeds 0-4; `noisy_models(6, s, seed=0)` for s = 0.0025 and 0.005. E27's ceiling table
had 10 models (its 4 noisy copies are the first 2 of each s here, same RNG stream); TASKS T3 asks for 18, so 8
noisy copies are added.

**Pass line (fixed in advance; comparable to E27 section 5).** A deductive estimate at cost 64k must have bias
well below the sampling s.e. sqrt(0.045/64000) = 0.084 pp. Primary configuration: **D3, k = 4, V1.**
- **PASS**: |bias| <= 0.08 pp on all 18 models, AND the primary bias is smaller than the X3 control's by more
  than 2 s.e. on at least 12 of 18 models (the gain is information, not linearisation).
- **PARTIAL**: the primary fails, but some other symbol configuration (D5/W3 at any k, V) has |bias| <= 0.08 pp on
  all 18. Flagged as post hoc (multiplicity: 30 non-primary configurations), needing a fresh-seed re-check.
- **FAIL** otherwise. Reported alongside: the mean |bias| per configuration vs E27's `none`.
A pass is a ceiling only (necessary, not sufficient): the number of distinct itineraries observed is reported,
because an estimator must also integrate over them.

**Alternative explanations to separate (rule 3):**
1. Any reduction in bias comes from forcing gates at all (linearising the clocks removes the Jensen effect),
   not from the symbol's information. Test: X3 vs D3 (same neurons, uninformative masks).
2. Any forced gates carry low-input information; the clocks are not special. Test: R3 vs D3 at equal |C|.
3. The reduction is limited by how C is chosen, not by the symbol idea. Test: D5, W3 and the A16 upper control.
   If even A16 is biased, the information left is in the continuous linear response to the lows, not in gates.

Files: `models.py` (models, clock rules), `ceiling_gates.py` (the grid), `results/` (outputs).

## Status
Pre-registered; scored run not started.

## Results
(pending)

## Compute accounting
Diagnostics only, no estimator. Forward passes: clock rules 18 x 200k; ceiling grid 18 x 600k x (1 + 42 reduced
variants + 3 resample passes for X3).

## Claims
(pending)

## Alternatives not ruled out
(pending; see the pre-registered list above)

## Failure modes
(pending) Known in advance: the ceiling is sampled (paired s.e. ~0.02 pp), not an exact integral; D3 is a
data-derived rule (it uses random inputs, as E27's diagnostics did).
