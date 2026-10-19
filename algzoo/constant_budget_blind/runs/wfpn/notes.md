# Notes: second-largest-position RNN, 8 hidden neurons

## Method
All access was through `probe.hidden` / `probe.predict`. Facts I measured first:
- The model gets the second-largest position right 42.6% of the time on Gaussian inputs.
- h_10 is positively homogeneous of degree 1: h(2x) = 2 h(x) exactly. That fits a network with no biases, so every formula is a ReLU of an intercept-free linear combination of degree-1 features.
- Zero inputs leave the state at 0. So feeding `[0,...,0, x0..x_{L-1}]` gives the hidden state after a prefix of length L. I used this to watch each neuron over time (`a7.py`, saved as Hpre.npy).
- Over time, neurons 0, 1, 2, 6 and 7 are active almost always (at least 98% from L=5 on). Their means grow roughly linearly with L, so they act as linear accumulators. Neurons 3, 4 and 5 are gates (active 64%, 96% and 83% of the time).
- One exploratory step went beyond pure input-output fitting (`a13.py`). I regressed h_10 on the length-9 prefix state plus x9, which recovers the one-step linear map. I used it only to understand the circuit, never in a formula, and the shipped formulas use only functions of x. It showed this chain:
  - n5 gates the current input.
  - n4 is mostly 1.49 x n5 delayed by one step.
  - n3 is strongly inhibited by the previous n4 (about -2.3).
  - n1 and n7 are inhibited by n3.
  - The accumulators are linear in the past values of n3, n4 and n5 plus the raw x values. A linear fit of each accumulator on the full prefix histories of n3, n4 and n5 plus x gives R^2 of 0.985 or more.

The formulas came from a greedy forward selection: up to 5 features, a ReLU fit refined by least squares on the active set, then a small gradient polish. The feature pool held:
- the x_k and the prefix maxima M_k;
- window sums of running-max gaps G_t = relu(max(x0..x_{t-1}) - x_t);
- window sums of top-two gaps g_t = relu(T_{t-1} - x_t), where T_k is the mean of the two largest of x0..xk;
- u_t = relu(x_t - T_{t-1});
- sorted values and sums of the top k;
- rank-based gaps.

Constants were fitted on seed 7 (100k inputs) and tested on seeds 8 and 11 (100k each).

## Per-neuron R^2 (fresh inputs, seed 11, 100k)
| neuron | cat | R^2 | gist |
|---|---|---|---|
| 0 | B | 0.750 | top-value level (T8) + shortfalls of x3..x5 |
| 1 | B | 0.746 | shortfalls of x4..x6 (esp. x6) + x4+x5 |
| 2 | B | 0.779 | broad shortfall sum x3..x9 + x7+x8 |
| 3 | B | 0.750 | "x7 not below earlier top-two mean" gate, with x3..x5 shortfalls |
| 4 | A | 0.943 | relu(T7 - x8) (+ small relu(T8 - x9)) |
| 5 | A | 0.963 | relu(1.08 T8 - 1.20 x9) |
| 6 | B | 0.707 | how far x4 is below the top / second values, + max(x0..x4) |
| 7 | B | 0.763 | shortfalls of x4..x6 (esp. x5) + x9 + top-value level |

## Overall picture
The network tracks, at each step, whether the new input x_t falls short of the "top level" T_{t-1}, the mean of the two largest values seen so far. That gate is n5_t ≈ relu(T_{t-1} - 1.2 x_t), and it held at every prefix length from 4 on with R^2 of 0.9 to 0.98.

Delayed copies (n4: lag 1) and an inhibited, oscillating copy (n3: roughly "x at lag 2 was not short") feed the five always-active accumulators. The accumulators sum these gate signals over time, each with its own position kernel: n0 peaks at x3..x5, n1 at x6, n6 at x4, n7 at x5..x6, and n2 is broad. On top of that they carry a general magnitude term. The result is a smeared positional code of where the inputs did *not* fall short, which the linear readout turns into a guess of the second-largest position. That guess is right 43% of the time.

The accumulators reach only R^2 of about 0.75 under the 5-constant limit. The missing variance comes from the full lagged history of the n3 gate, which oscillates with alternating signs over lags. Adding the true n3 history as a feature raises the accumulators to R^2 0.84–0.90, and adding n3's own lag structure would raise them further. Neither fits within 5 constants.

## Files opened
- TASK.md (read)
- My own scripts and data in this directory: a1.py–a19.py, feats.py, feats2.py, pool.py, featlib.py, fit.py, x.npy, h.npy, Hpre.npy, xpre.npy, o14*.txt, o15*.txt, o18a.txt, consts.txt, labels.py
- I imported probe.py but never opened or read it.
- I read one background-task output file the harness created for my own command. It is the stdout of a6.py, which lives in the harness tasks directory.
