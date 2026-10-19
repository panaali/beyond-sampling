# Notes: neurons of the second-largest-position RNN (h_10, 32 units)

## Method
I only used `probe.hidden` and `probe.predict`, on Gaussian inputs (seeds 1-15, 21, 31, 41, 42, 61, 62, 77, 99, 501, 502, 777)
and on crafted "spike" sequences (one value raised above a constant background). I never opened probe.py.
Steps:
1. Per-neuron summaries: firing rate, mean activation by position of the overall max and second max, correlation with each x_t.
2. Spike responses at each position, which showed position-specific "clock" patterns (e.g. a neuron active only if the spike is at x2 or x6).
3. A check that the map is positively homogeneous: H(2x) = 2H(x) holds exactly. That fits a bias-free ReLU RNN, so I wrote every formula
   without intercepts, as ReLU of a linear combination of order statistics with position-dependent gates.
4. Finite-difference gradients dh/dx_t. They showed which inputs each neuron depends on. Many neurons ignore x9, and one ignores x8 too, so
   for those I switched to statistics of the prefix x0..x8 (or x0..x7). Most of the regressions improved after this switch.
5. Greedy forward selection (ReLU least squares) over a library: order statistics of the whole input and of prefixes, single x_t,
   values multiplied by indicators "argmax/second position in set S" (sets taken from per-position coefficient fits), and sums.
   Then I fitted the chosen hand formulas, each with at most 5 constants, by nonlinear least squares on 40k sequences (seed 501).
6. Evaluation on fresh sequences: seed 502 (40k) during selection, and a final check on seed 777 (50k) via `labels.f`.

## Per-neuron R^2 (fresh inputs, seed 777, 50k sequences)
0:0.960 1:0.989 2:0.881 3:0.845 4:0.634 5:0.891 6:0.949 7:0.999 8:0.786 9:0.749 10:0.859 11:0.878 12:0.999 13:0.970
14:0.796 15:0.892 16:0.996 17:0.994 18:0.930 19:0.873 20:0.779 21:0.943 22:0.900 23:0.978 24:0.846 25:0.933 26:0.842
27:0.887 28:0.889 29:0.955 30:0.803 31:0.974.  Mean 0.894, median 0.892.

## Overall picture
Notation: M = largest of x0..x8, S = second-largest of x0..x8.
- **Value registers that ignore x9:** 17 (about 8.1M+2.0S, the running max), 31 (about 5.5S, the second max), 1 and 5 (the margin M-S; 1 vanishes when x8 is the max).
  These hold the state after x8.
- **The last step (x9) compared with the registers:** 7 = 8.8·ReLU(x9-M) (x9 is the new max), 16 = 5.7·ReLU(x9-S) (x9 enters the top two),
  12 = 3.1·ReLU(S-x9) (x9 is not in the top two), 21 and 23 (graded versions). Neurons 0, 6, 13, 18 and 29 are ReLU(aM+bS-c·x9),
  "how far x9 falls below the earlier values". The weight on M depends on where M sits (a different set of positions for each neuron).
  So these neurons carry a position-of-max code together with the x9 comparison.
- **Position detectors:** 15 (x1), 26 (x2), 2 (x3), 19 (x6), 22 (x7) and 14 (x8). Each grows with its x_k and is boosted when x_k is the max or second max.
  Most subtract a small multiple of the sum of the other inputs.
- **Anti-detectors:** 20 (x0), 30 (x4), 24 (x5, which reads only x0..x7) and 25 (x8). Each carries the level of the top-two values but is zeroed when x_k is in the top two.
- **Position-of-max ("clock") codes:** 27 ({2,6}), 11 ({0,4,6}), 10 ({0,4,5}), 3 (early max), 8 (max early or at x5), 28 (max not at x8, weighted early)
  and 4 (second max at an even position, poorly understood). Per-position fits show irregular, parity-like position sets.
  This looks like the hidden state rotating after each new max, so the phase encodes when the max occurred.
- 9 is almost always silent; it fires only when most early inputs are negative.
Read-out logic, not verified (I had no access to the readout): the second-largest position follows from the position-of-max and position
codes, combined with whether x9 entered the top one or top two (7, 16, 12).

## Caveats
- Gates, position sets and feature choices were searched and are declared in STRUCT. The fitted reals are in CONSTANTS/FITTED.
- The weaker fits (4, 9, 8, 20, 14, 30) leave structure unexplained. Position-specific weights on the third and fourth ranks seem to matter, and I could not
  compress them into 5 constants.

## Files opened
In the working directory: TASK.md (read). I created and ran: a1.py, a2.py, feat.py, lib.py, a3.py, a4.py, a5.py, a6.py, gate.py, h9.py, det.py,
fx.py, run.py, cands.txt, cell.py, grad.py, hom.py, pat.py, cellL.py, b14.py, g9.py, tab.py, w4.py, final.txt, runf.py, final.json, mk.py,
labels.py, chk.py. I read the outputs g1.txt, g2.txt, gall.txt, G.npy and Gx.npy. I also read the stdout file of one of my own background runs, which the
harness put at /tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/tasks/bdc4yuya1.output (output of a5.py only).
probe.py was imported but never opened. No other files were read.
