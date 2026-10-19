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
Done (2026-10-19). Verdict on the pre-registered rule: **FAIL. Negative (unverified)**, scoped below. No
estimator was built.

## Results
Run from this folder with the venv active: `python ceiling_gates.py` (all 18 models, ~1.5 min per model on 4
threads; per-model JSON in `results/`), then `python summarize.py` (all six k x V tables and the verdict:
`results/summary.md`). Paired s.e. of each entry: 0.011-0.065 pp.

**Primary configuration (D3, k = 4, V1), bias in pp:**

| model | none (E27) | D3 | D5 | W3 | R3 | X3 | A16 |
|---|---|---|---|---|---|---|---|
| orig | +0.104 | -0.011 | -0.258 | +0.072 | -0.047 | -0.273 | -0.589 |
| seed0 | +0.253 | -0.591 | -0.696 | +0.153 | +0.110 | -5.470 | -1.169 |
| seed1 | +0.598 | -0.606 | -0.915 | -0.017 | -0.987 | -5.763 | -0.956 |
| seed2 | +0.209 | -0.049 | -0.385 | -0.078 | +0.029 | -0.756 | -0.869 |
| seed3 | +0.204 | -0.493 | -0.611 | +0.109 | -0.160 | -12.857 | -0.781 |
| seed4 | +0.210 | -0.468 | -0.609 | +0.047 | -0.103 | -4.901 | -1.145 |
| s0.0025_0..5 | +0.086..+0.287 | -0.044..+0.167 | -0.288..-0.080 | +0.055..+0.255 | -0.063..+0.169 | -0.290..-0.072 | -0.613..-0.440 |
| s0.005_0..5 | +0.106..+0.539 | -0.018..+0.460 | -0.272..+0.288 | +0.073..+0.506 | -0.035..+0.409 | -3.337..+0.303 | -0.595..-0.113 |
| **mean abs (18)** | 0.259 | 0.218 | 0.300 | 0.174 | 0.172 | 1.959 | 0.578 |
| **max abs (18)** | 0.598 | 0.606 | 0.915 | 0.506 | 0.987 | 12.857 | 1.169 |

1. **Pass line not met.** D3 k4 V1 is within 0.08 pp on only 7 of 18 models (max |bias| 0.606 pp, seed1).
   Condition 2 holds, but only just: D3 beats the shuffled-symbol control X3 by > 2 s.e. on exactly 12/18.
   **No PARTIAL either:** none of the 30 other D5/W3/D3 configurations is within 0.08 pp on all 18 (the best
   max |bias| over the grid is 0.414 pp, D3 k5 V1). `none` reproduces E27's orig numbers (+0.10 / +0.09 pp at
   k = 4 / 5 V1), and on the 4 shared noisy copies it agrees within ~0.05 pp. E27 used the same seed 7, so the
   samples are E27's for those models.
2. **The symbols trade E27's positive bias for a model-dependent sign.** E27's reduction (`none`) is
   positive on 18/18 models (mean +0.259 pp). With the clock masks forced (D3) the sign is +10 / -8: about
   -0.5 pp on four of the five zoo seeds, -0.05 to +0.17 on the s = 0.0025 copies, -0.02 to +0.46 on
   s = 0.005. Forcing *all* gates (A16) is negative on 18/18 (mean -0.578). So forcing a gate itinerary adds a
   **negative** bias of its own, and D3's near-zero mean (-0.035) is mostly two opposite biases cancelling,
   not information closing the gap.
3. **Mechanism of the negative bias** [inferred]: a forced neuron is linear inside the itinerary's cylinder,
   but the frozen input x_r is generally *outside* that cylinder. A neuron forced "on" while its
   pre-activation is negative carries a negative value that ReLU would never produce, and the error
   compounds over steps. The reduction "symbol + frozen lows" is therefore not self-consistent. The values
   given to the lows must be conditioned on the cylinder, which this design (and E27's) does not do.
4. **The clocks are not special.** At equal |C| = 3, random non-clock neurons (R3) do as well as or better
   than the clock rule: mean |bias| 0.172 vs 0.218, max 0.987 vs 0.606. The weight-rule set W3 (never equal to
   D3) has mean 0.174. Adding the next two clocks (D5) makes the bias more negative (mean |bias| 0.300).
5. **The true masks do carry information.** X3 (same neurons, masks from resampled lows) is far worse, down to
   -12.9 pp on seed3, so the gate itinerary encodes low-input information that frozen values cannot
   reproduce. But the gain over a random gate set is nil (point 4).
6. **Symbol count.** D3 itineraries: 16.6k-64.3k distinct per 600k inputs (still undersampled, so the true
   support is larger). Even a passing ceiling would have needed an integral over tens of thousands of
   cylinders, each with a top-4 continuous part.
7. Other k / V: V2 (all lows at one value) combined with forcing is much worse (D3 k4 V2 mean |bias| 1.30,
   up to -9.8 pp), consistent with point 3. Collapsing all lows to one value puts x_r further from the
   cylinder. k = 5 V1 is the least bad (D3 mean 0.156, D5 0.134, still max > 0.4).

## Compute accounting
Diagnostics only, no estimator; none of it is charged to any estimate. Model forward passes (length-10
sequences): clock rules 18 x 200k = 3.6M; ceiling grid 18 x 600k x (1 true + 3 X3 resamples + 42 reduced)
= 497M. Total ~0.5G sequence passes, ~30 CPU-minutes per model-batch at 2 threads (~45 min wall).

## Claims
- [measured] The reduced system "clock gate itinerary (D3) + top-4 values, lows frozen (V1)" has ceiling bias
  -0.61 to +0.46 pp on 18 models, within 0.08 pp on 7/18: it fails the pre-registered pass line.
- [measured] No configuration in the pre-registered grid (D3/D5/W3 x k 3-5 x V1/V2) is within 0.08 pp on all 18.
- [measured] Forcing all 16 gates to the true itinerary with frozen lows biases accuracy downward on 18/18
  (k4 V1: -0.11 to -1.17 pp); E27's unforced reduction is upward on 18/18.
- [measured] At |C| = 3, random non-clock gate sets reduce |bias| as much as the clock set (mean 0.172 vs 0.218).
- [measured] True masks beat shuffled masks (X3) by > 2 s.e. on 12/18 models.
- [inferred] The negative bias comes from frozen inputs lying outside the forced cylinder (off-cylinder
  linear extrapolation). Not directly tested: no per-input cylinder-violation count was made.
- [inferred] Under this kind of reduction, clock gate switches are not a privileged symbol set for M_16,10
  and its retrains/copies.
- [speculative] A self-consistent version (lows at their conditional mean *given the cylinder*) would remove
  the negative part and leave E27's positive Jensen part, i.e. it would not pass either.

## Alternatives not ruled out (rule 3)
1. *Information vs linearisation* (pre-registered alternative 1): X3 is much worse than D3, so it is not
   pure linearisation. But X3 is a weak control: its masks are inconsistent with x_r in a different way, so
   its large negative values partly measure inconsistency rather than lack of information.
2. *Clock-specific vs any gates* (alternative 2): answered within the grid: R3 is as good. Untested: the
   best possible 3-neuron set.
3. *Choice of C vs the idea* (alternative 3): A16 (all gates) fails worst, so no choice of C fixes the design
   as built. That says the **design** (frozen lows + forced masks) is inconsistent. It does not say gate
   symbols are uninformative.
4. The FAIL could be an artefact of the off-cylinder mechanism (claim 3), not a limit of gate symbols.
   The cheapest separating test: lows at their empirical conditional mean given (S, top-k skeleton), or a
   projection of x_r onto the cylinder. Not run (post hoc).

## Exhaustion checklist (rule 5)
Tried: symbol sets D3, D5 (data rule), W3 (weight rule), R3, A16; masks true vs shuffled; low closures V1,
V2; k = 3, 4, 5; 18 models (orig, zoo seeds 0-4, 12 noisy copies at sigma 0.0025 / 0.005).
Not tried, and why:
- Cylinder-consistent low values (conditional on S): needs per-cylinder statistics over >= 20k cylinders.
  This is the main open variant (alternative 4).
- Stochastic reductions (carry the lows as noise, as E27 also left open).
- Soft symbols (sign of the pre-activation with a margin band), switch *times* only (with masks free), other
  section surfaces (e.g. phase of the rotating eigenmode instead of the ReLU hyperplanes).
- Larger sigma (0.05, the ARC protocol level) copies: E27's model set was used. Accuracy is lower there and
  the biases would likely be larger.
- Checkpoints other than the AlgZoo zoo seeds.
So this is "negative (unverified)" for the reduction **frozen lows + forced true clock masks**, not for gate
symbols in general.

## Failure modes
- The ceiling is sampled (600k paired inputs; s.e. 0.01-0.07 pp), not an exact integral. The verdict is
  far from the line (max |bias| 0.61 vs 0.08), so sampling error does not affect it.
- D3 is a data-derived rule (random diagnostic inputs, as in E27). W3 is weight-only and does no better.
- The forced-mask network is not the true network restricted to a cylinder whenever x_r leaves the
  cylinder. This is the main defect of the design (see claim 3), and an independent check should start there.
- Models: "18 models (as in E27)" was read as orig + 5 zoo seeds + 12 noisy copies. E27's ceiling table had 10.
