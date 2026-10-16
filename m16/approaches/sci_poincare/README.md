# sci_poincare: does M_16,10 reduce to a small symbolic system?

## Question
Are there coordinates in which the 10-step dynamics become a finite automaton over record events with a
low-dimensional continuous state? If so, P(correct) would be a computable sum over symbolic histories with
exactly integrable continuous parts, which would give a deductive accuracy estimate (INDEX Q4). This tests
whether such a reduction is exact, how much error it leaves, and whether that transfers across seeds.

## Method
Qualitative dynamics first, then ceiling tests. All scripts use random inputs, but **for diagnosis only**.
No estimator was built (see Results, test 5).

Natural coordinates, an exact model-free fact: x is equivalent to (relative-rank sequence r_t, order
statistics). By Renyi's record theorem the r_t are independent with r_t uniform on {1..t+1}. Record events
are therefore a product-measure symbolic stream. The candidate reduction is: symbols = relative ranks capped
at K (plus the sign of x_t, since with no biases the sign sets which neurons gate on), and continuous state =
the top-k values.

| file | test |
|---|---|
| `diag_resample.py` | keep the top-k values and positions, resample the rest below the k-th value from their exact conditional law, and measure how often correctness flips |
| `diag_memory.py` | forgetting: does it matter how long ago a low input arrived? |
| `diag_additive.py` | cut-HDMR additivity of the correctness margin in the low inputs (rank > 3), near 2-3 ties |
| `diag_automaton.py`, `diag_automaton2.py` | gate-pattern entropy given the symbol history, and a Markov automaton test H(G_t given G_{t-1}, r_t, sign x_t) per neuron |
| `ceiling_meanfield.py`, `ceiling2.py` | **ceiling test**: the exact accuracy of the reduced system (full permutation + top-k values, low inputs replaced by deterministic values given the k-th value), compared with the true accuracy. This is the best any perfect integration of the reduction could return |
| `diag_seeds.py` | the resample, additivity and automaton tests on the original and seeds 0-4 |

## Status
Done. Verdict: **negative.** No symbolic reduction of the kind proposed is accurate enough, and none was built
into an estimator.

## Results
Commands are run from this folder with the venv active, e.g. `python ceiling2.py 4 5`.

1. **No forgetting, no renewal structure** (`diag_memory.py`, original). Keep the top-3 and resample the
   low inputs: correctness flips 3.3% of the time (the error rate is 4.7%). Each low input adds about 0.4%
   whatever its age. Keeping the last k inputs lowers the flip rate only linearly (3.3% -> 2.1% at k = 7).
   The dynamics are a transient, not mixing. The integrator eigenvalue is at least 1 and the clocks have
   |lambda| ~ 0.95, so 10 steps contract by only ~0.6, and records are not resets. A Poincare return map or
   renewal decomposition has nothing to act on.
2. **Low-input effects are strongly non-additive** (`diag_additive.py`, `diag_seeds.py`). Cut-HDMR R^2 of the
   margin change near ties: orig 0.43-0.45; seeds 0-4: 0.30, 0.33, 0.31, 0.46, 0.31. Resample flip rate with
   the top-3 fixed: orig 3.3%, seeds 7.7 / 9.7 / 4.5 / 5.4 / 8.3% (errors 9.0 / 12.0 / 5.2 / 6.8 / 11.0%).
3. **The gate dynamics are not a finite automaton over record symbols** (`diag_automaton*.py`, 400k
   samples). With the history capped at K = 3, H(G_9 given history) = 0.77 bits (plug-in, biased low:
   55,720 histories). The Markov version H(G_t given G_{t-1}, r_t) stays at ~2 bits per step, and the modal
   next pattern has only 53-73% of the mass. Unpredictability is concentrated in a few neurons: on the
   original, gate flips not predicted by (G_{t-1}, rank symbol, sign) number 5.1 per sequence, of which n3
   accounts for 1.19 and n15 for 0.99, then n11, n5, n14. These are the rotating-mode neurons: their ReLU
   crossings happen at value-dependent phases. The concentration transfers across seeds, but the identity of
   the neurons does not (seed0: n8, n3; seed1: n12 at 1.9; seed2: n2; seeds 3-4 diffuse, 3.1-4.1 flips per
   sequence in total).
4. **Ceiling of the reduction "symbolic permutation + top-k values"** (`ceiling2.py`, 600k paired samples
   per model, s.e. of each difference ~0.02 pp). Bias of the reduced system's accuracy, in pp (V1 = low
   inputs at their expected order statistics in their actual order; V2 = all at the conditional mean; V3 = at
   the conditional median):

| model | V1 k=4 | V2 k=4 | V1 k=5 | V2 k=5 |
|---|---|---|---|---|
| orig | +0.10 | +0.15 | +0.09 | +0.14 |
| seed0 | +0.25 | +0.33 | +0.27 | +0.35 |
| seed1 | +0.60 | +0.75 | +0.43 | +0.54 |
| seed2 | +0.21 | +0.24 | +0.16 | +0.22 |
| seed3 | +0.20 | +0.08 | +0.14 | +0.20 |
| seed4 | +0.21 | +0.19 | +0.19 | +0.29 |
| s0.0025_0 / _1 | +0.17 / +0.29 | +0.22 / +0.32 | +0.16 / +0.25 | +0.19 / +0.24 |
| s0.005_0 / _1 | +0.35 / +0.54 | +0.45 / +0.25 | +0.26 / +0.44 | +0.32 / +0.20 |

   On the original, k = 2/3/4/6/7 (V1) give +2.99 / +0.25 / +0.09 / +0.08 / +0.07 pp: the bias plateaus
   instead of vanishing. **All 60 bias numbers are positive.** Freezing the low inputs at deterministic
   values always removes errors, because their fluctuations only widen the bands (a Jensen-type effect).
5. **Why no estimator was built.** To compete, a deductive estimate at cost C needs bias well below the
   sampling s.e., sqrt(0.045/C): 0.08 pp at 64k, 0.02 pp at 1M. The reduction alone already costs +0.1 to
   +0.6 pp before any integration error. Its symbolic part also has at least 5,040 position tuples at k = 4,
   each needing a 4-D integral. It cannot win at any budget, on any of the 10 models tested.

## Relation to H-E9 (bandwidth, catastrophic cancellation)
These results are consistent with H-E9. The tie margin is a small residual of large cancelling terms, so
replacing seven low inputs by typical values is a small change to the components and a large change to the
residual. The positive sign of the ceiling bias says more: the low inputs act as **noise injected into the
residual**. Averaging them out before the nonlinearity (the sign of the margin) always overstates accuracy.
Any reduction that summarises them by a deterministic state (mean field, modal gates, canonical points: E10,
E21 and this one) inherits a positive bias of 0.1-0.75 pp.

## Compute accounting
Diagnostics only, with no estimator. Forward passes: ceiling2 ~42M (10 models x 600k x 7); ceiling_meanfield
~12M; diag_seeds ~11M; automaton ~1.6M; others <1M. None are charged to any estimate.

## Claims
- [measured] Correctness depends on every low input, about equally whatever its age (no forgetting in 10 steps).
- [measured] Margin effects of the low inputs are non-additive (R^2 0.30-0.46, 6 models).
- [measured] Gate patterns are not determined by record and sign symbols (~2 bits/step left over); the
  uncertainty is concentrated in 1-3 rotating-mode neurons whose identity varies by seed.
- [measured] The reduction "permutation + top-k values (k = 4, 5)" has a ceiling bias of +0.07 to +0.75 pp,
  always positive, on 10 models.
- [inferred] No deductive estimator built on a reduction of this type can match sampling in MSE.
- [inferred] The positive bias is noise-averaging (Jensen) on a cancelling residual (H-E9).

## Alternatives not ruled out
- Stochastic rather than deterministic reductions: keep the low inputs as noise and propagate the exact
  distribution of the residual. H-E9 and E13 suggest the residual's law is hard to close, but this was not
  tested here.
- Other symbol choices (e.g. phase-crossing events of the rotating neurons as symbols) were not tried. The
  automaton test only says record and sign symbols are insufficient.
- The plug-in entropies are biased low for the large history classes. This makes the automaton look *better*
  than it is, so it does not weaken the negative result.
- Quantile-type values (V1) are not exact conditional means. V2 uses exact means and is no better.

## Failure modes
These are all diagnostics on samples. The ceiling numbers have ~0.02 pp s.e. Only 4 of the 12 noisy copies
were tested.
