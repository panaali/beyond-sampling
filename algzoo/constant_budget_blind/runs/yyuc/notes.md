# Notes: h_10 neurons of the second-largest RNN (16 neurons)

## Method
Black-box only (probe.hidden / probe.predict). Data: 100k Gaussian sequences (seed 1) for fitting, 50k fresh
sequences (seed 7) for all reported R^2. Model accuracy on seed 1: 94.9%.

1. Structural fact used throughout: h_10 = ReLU(L(h_9) + u*x9), so every neuron is ReLU(function of the
   prefix x0..x8 + one constant times x9), and (no biases) every neuron is positively homogeneous of degree 1.
   Every formula is therefore ReLU(linear combination of degree-1 features), no intercept.
2. Linear fits on sorted values and running maxima showed that most neurons depend on the largest values of
   the prefix with weights that depend strongly on *where* the maximum sits. Single-spike probes
   (x = e_p) gave a position-response profile per neuron; these fit a cosine in (9-p) with period about 3.25
   steps for almost every neuron (shared period). So the network stores the maximum in a rotating "clock":
   a position p has phase about 110.8 deg*(9-p), which orders the positions on a circle as
   2,5,8,1,4,7,0,3,6 (roughly the classes p mod 3, with drift). Each neuron reads the stored values with its
   own preferred phase window.
3. Formulas: greedy forward selection (up to 5 features, linear fit on active samples, then ReLU least
   squares) from a library: sorted prefix values P1>=P2>=P3 (of x0..x8), x_t, max(x0..x7),
   max / second-max over a position set S, and "P_r if its position is in S". Restricting S to arcs of
   the phase circle lost almost nothing compared with free subsets, supporting the clock picture.
   Discrete choices searched: the feature list and position sets (declared in STRUCT). Constants refit on
   all 100k samples and hard-coded.

## Per-neuron R^2 (fresh seed-7 data, 50k sequences)
| n | cat | R^2 | gist |
|---|---|---|---|
| 0 | A | 0.965 | ReLU(~13*prefix max (0 if at x8) - 7.6*x9), phase window x0,x2,x3,x6 |
| 1 | B | 0.878 | max(x0..x7) minus maxima over other phase arcs (top at x2/x5/x6) |
| 2 | D | 0.716 | second-largest of a set + x2; poorly understood |
| 3 | A | 0.956 | as 0, phase window x1,x2,x4,x5 |
| 4 | A | 0.947 | margin P1 - P2 of the prefix; zero if max at x8 |
| 5 | C | 0.859 | max over a phase arc with corrections for max/second at x5 |
| 6 | A | 0.966 | as 0, phase window x0,x1,x4,x7 |
| 7 | A | 0.963 | running max, weighted toward x0,x3,x6 |
| 8 | B | 0.902 | max over arc x0,x1,x3,x4,x7 minus runner-up, position bonuses |
| 9 | C | 0.825 | max over arc x0,x1,x4,x7 + x1 + position bonuses |
| 10 | A | 0.990 | ReLU(8.4*x9 - 4.7*P2 - 1.9*P3): x9 is in the top two |
| 11 | C | 0.867 | P1 gated on two position sets minus P2 |
| 12 | A | 0.935 | margin 21*P1 - 12.4*P2, reduced for big values at x2,x5 |
| 13 | D | 0.776 | max + second max over arc x0,x1,x3,x4,x7, partial |
| 14 | A | 0.932 | prefix max read through phase windows (strong at x1,x2,x4,x5,x8) |
| 15 | A | 0.9995 | 26.8*ReLU(x9 - 0.989*max(x0..x8)): x9 is the new maximum |
Mean R^2 = 0.905.

## Overall picture
- Last-step detectors: neuron 15 fires iff x9 is the overall maximum; neuron 10 iff x9 beats the prefix
  runner-up (x9 in top two). Neurons 0, 3, 6 are "earlier max minus x9" units (large negative x9 weight),
  each covering a different third of the phase circle, and are silenced when x8 is the prefix maximum.
- Value/margin units: 4 and 12 carry the gap between the largest and second-largest prefix values; 7 and 14
  carry the size of the prefix maximum.
- Position is encoded by the clock phase: the remaining neurons (1, 5, 8, 9, 11, 13, 2) read the maximum
  (and to a lesser extent the second maximum) through different phase windows, so the joint pattern
  across neurons tells the readout where the max and runner-up were. The second-largest position must be
  read from this mixture; the lower R^2 neurons (2, 13, 9) depend on finer position combinations (a
  lookup-table fit on (argmax, argsecond) positions with top-3 values also only reached about 0.82-0.86 for
  them), so their labels are approximate.

## Caveats
- Position sets come from greedy search and are partly fitting artefacts; the arc structure is the robust part.
- R^2 is measured on Gaussian inputs only.

## Files opened
- TASK.md (read). probe.py was only imported, never opened.
- My own scripts/data in this directory: explore1-9.py, fitlib.py, probe1.py, probe2.py, n2.py, cosfit.py,
  clock1.py, search1.py, greedy.py, greedy2.py, greedy3.py, featlib_src.py, fitall.py, mklabels.py,
  mkfresh.py, check.py, logs g1/g2/h1/h2/k1/k2.log, fit.json, *.npy, labels.py, notes.md.
