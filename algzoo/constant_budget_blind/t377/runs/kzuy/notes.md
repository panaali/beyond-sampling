# Neuron labels for the second-largest-position RNN (h_10, 16 neurons)

## Method
All access went through `probe.hidden` / `probe.predict` on Gaussian or crafted inputs (seeds 1-41, 99, 123; never 20261016).
1. Basic statistics and correlations of each neuron with simple features (max, second, gaps, x_t, argmax indicators).
2. Linear regressions on x, sorted values and running maxima M_k = max(x0..x_k). Adding the running maxima raised R^2 a lot
   (e.g. 0.33 -> 0.98 for some neurons), so the state is mostly a memory of "the max so far" plus traces of individual inputs.
3. Checked that h is positively homogeneous (h(2x) = 2h(x), as expected with no biases), so the formulas use degree-1 features
   and no intercept.
4. Impulse inputs (one nonzero x_t) gave the response to a lone maximum at each position. It alternates with the parity of the
   max's position: the "max memory" sits in a different mix of neurons for even and odd argmax positions. A max at x9 is
   hardly stored at all, except through the x9-is-max detector.
5. Per-argmax-group regressions: inside each group h is close to linear, with gain G(argmax)*max (mostly set by parity, with
   special values for argmax 8 and 9) plus fixed, position-specific negative weights on the other inputs.
6. Greedy, ReLU-aware greedy and beam forward selection of at most 5 features from a dictionary of about 600 interpretable
   features: x_t, x_t when not the max, ReLU(+/-x_t), M_k, M_k - x_t, max - x_t, running second-largest S_k, capped values
   min(x_t, M_{t-1}), ReLU(x_t - M_{t-1}), pairwise max/min/ReLU-difference, top-two gap g = s0 - s1, and parity-of-argmax
   indicators times max or gap. Final constants were fitted by iteratively refitting least squares on the active set
   (max(y, pred) > 0) with a ReLU output. I also tried shared-weight position-set templates; they were not better.

## Per-neuron R^2 (fresh inputs, seed 123, N = 100000; f = labels.f)
| neuron | cat | R^2 | #const |
|---|---|---|---|
| 0 | A | 0.954 | 5 |
| 1 | B | 0.974 | 5 |
| 2 | A | 0.990 | 3 |
| 3 | B | 0.858 | 5 |
| 4 | B | 0.908 | 5 |
| 5 | A | 0.931 | 5 |
| 6 | A | 0.988 | 3 |
| 7 | A | 0.954 | 5 |
| 8 | A | 0.959 | 5 |
| 9 | A | 0.905 | 5 |
| 10 | A | 0.992 | 1 |
| 11 | A | 0.980 | 2 |
| 12 | B | 0.858 | 5 |
| 13 | A | 0.891 | 5 |
| 14 | A | 0.907 | 5 |
| 15 | C | 0.749 | 5 |

## Overall picture (inferred)
- **Max memory** (0, 3, 4, 5, 7, 8, 9, 13): these neurons are always on and carry the running maximum of x0..x7 or x0..x8 with
  a large gain. The gain depends on whether the max sits at an even or odd position, so the max is stored in an alternating
  (period-2) code. Each neuron also subtracts a few specific earlier inputs. 13 is extra sensitive to a new max at x8.
- **"How far is x_t below the max"** (2, 11 for x9; 6 for x7; 1 for x6): roughly c*(max - x_t), corrected by the top-two gap
  with a parity-dependent sign. When x_t is the second largest, max - x_t equals the gap, so with the gap correction these
  units can tell whether x_t is the runner-up.
- **Position-specific max detectors**: 10 (x9 is the max, value = x9 minus the second largest) and 14 (x6 is the max).
- **Early-position trackers**: 12 (x3 relative to x1 and to max(x0..x2)) and 15 (x2, x5, x6, x1 mixture); these are fit worst.
- The model's argmax-of-second accuracy on my sample was 0.88.

## Caveats
Formulas with 5 constants cannot capture the many small position-specific weights, nor the nonlinear handling of a max at
x8/x9 (in-group R^2 there was about 0.3-0.5). Neurons 3, 12 and 15 are only partly explained. Feature sets were chosen by
search (declared in STRUCT), so R^2 was measured on separate fresh inputs.

## Files opened
Only inside this directory: TASK.md (read), probe.py (only imported, never opened), and my own scripts and outputs
(feat.py, dic*.py, featlib.py, p*.py, p18helper.py, fit.py, final.py, gen.py, labtext.py, cands.json, *.out, *.json,
labels.py, notes.md).
