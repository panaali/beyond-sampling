# Notes: h_10 neurons of the second-largest RNN (32 hidden)

## Method
1. **Probing only through `probe.hidden`.** Inputs were Gaussian (seeds 5, 7, 8, 11, 12, 13), with x9 sometimes clamped
   to ±20/±30/±40/±50/0. probe.predict matched the true second-largest position 99.45% of the time (seed 5, 100k inputs).
2. **Key structural fact (measured).** h_10[i] = ReLU(w_i·x9 + c_i(x0..x8)). Clamping x9 to large ± values gives the per-neuron
   slope w_i (constant across inputs) and the exact pre-activation c_i(prefix) wherever the neuron is active. Four neurons have
   large positive w (x9 "rises above" detectors), six have large negative w (x9 "is low" detectors), and most of the rest have
   |w| < 0.3. So the remaining work is to describe c_i as a function of the 9-number prefix.
3. **Feature search.** For each neuron: greedy, then beam (width 4) forward selection of up to 5 features. Each candidate was
   scored by an iterated "ReLU least-squares" fit (refit on the set {target>0 or prediction>0}). The library: M1, M2, M3 (top
   values of x0..x8), M1−M2, x9, x_j, ReLU(x8 − max(x0..x7)), ReLU(x_j − M2/M3), ReLU(M1/M2 − x_j), and position-set indicator
   features [argmax ∈ S]·{M1, M2, M1−M2}, [second ∈ S]·{M1, M2, M2−M3}, [S hits the top two]·{M1, M2}. The candidate position
   sets S came from a per-position coefficient table (a ReLU fit on [argmax=j]·M1 and similar terms), taking the top-k or
   bottom-k positions (k ≤ 4). Constants were refit on seed 11 (30k). For a few neurons I swapped in a simpler, more readable
   formula when it lost little R² (17, 21, 22, 27, 1). Every formula has the form ReLU(Σ c_k·feature_k) with ≤ 5 fitted constants
   and no intercept: the network has no biases, so h is positively homogeneous in x.

## Per-neuron R² (labels.f against probe.hidden, fresh seed 13, 50k Gaussian inputs)
0:0.964 1:0.992 2:0.890 3:0.814 4:0.678 5:0.927 6:0.968 7:1.000 8:0.563 9:0.771 10:0.863 11:0.908 12:1.000 13:0.964
14:0.792 15:0.906 16:0.997 17:0.995 18:0.954 19:0.893 20:0.773 21:0.985 22:0.905 23:0.985 24:0.775 25:0.934 26:0.876
27:0.946 28:0.853 29:0.946 30:0.799 31:0.979.  Mean 0.894, median 0.917, variance-weighted 0.948.

## Overall picture
- **Running order statistics of the prefix.** One neuron holds ≈8.1·M1 + 2.0·M2 (running max plus a little of the second).
  One holds ≈5.5·M2 (the running second-largest). Two hold the lead M1 − M2. One fires when every value is low (−M2).
- **Comparing the newest input with the running statistics.** These neurons all depend sharply on x9: ReLU(x9 − max) (new
  maximum), ReLU(x9 − M2) (x9 enters the top two), ReLU(M2 − x9) (x9 is not top-two), and variants. Some are gated off when x8
  was itself a new maximum, through a large negative ReLU(x8 − max(x0..x7)) term. That term is exactly the "new max" detector one
  step earlier, which shows how the recurrence reuses the same comparison circuit at every step. Six "x9 is low" neurons combine
  −x9 with M1/M2 and with a code for where the maximum is.
- **Position codes.** The output must name a position, so many neurons are position detectors: "x1/x3/x6/x7/x2 is among the
  top two (or is the max)", "the max is at x2 or x6", "the max is at x0, x4 or x6", "the max is at an odd position". There are
  also "x_j is NOT a top value" neurons (x0, x4, x5, x8). Together they form a distributed code for the positions of the largest
  and second-largest values, scaled by M1 or M2. The readout presumably combines this code with the x9 comparators: if x9 enters
  the top two, the answer is either 9 or the old argmax; otherwise it is the old second position (or the old argmax, when x9 is
  the new max).
- **Not well understood:** 4, 8, 20, 28 (D), and the mixed 3, 10, 14, 24, 30 (C). Their R² is 0.56–0.86. They look like
  linear mixtures of many position-specific prefix terms (neuron 8 is never zero, so it is a purely linear mix of h_9).

## Caveats
- The position sets were found by search (declared in STRUCT), and the constants were fitted on Gaussian inputs only.
- The labels quote rounded constants; labels.py holds the exact fitted values.
- The w·x9 split is exact by construction (ReLU of an affine function of x9). The descriptions of c_i are approximations.

## Files opened/read
- TASK.md (the instructions). No other pre-existing file was opened. probe.py was only imported, never read.
- My own files in this directory: a1.py–a11.py, feats.py, plib.py, flib.py, fit.py, relufit.py, core.py, search*.py,
  final.py, labtext.py, evalr2.py, spec*.json, fit*.json, the d*.npz data, g1/s1/b1/b2 logs, labels.py, notes.md.
