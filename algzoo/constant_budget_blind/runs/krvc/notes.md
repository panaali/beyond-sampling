# Notes: h_10 neurons of the second-largest-position RNN

## Method
Only `probe.hidden` / `probe.predict` were used (probe.py was imported, never opened). Data: 100k Gaussian
sequences (seed 1) for fitting/exploration, 20k sequences (seed 2) for a resampling-sensitivity scan (resample
one x_j, measure the change in each neuron), and a fresh 50k (seed 777) for the final R^2 below. Model accuracy
on seed-1 data vs the true second-largest position: 0.994.

Steps: (1) correlations and lookup-table R^2 against argmax/argsecond positions (of all ten and of x0..x8);
(2) the sensitivity scan, which splits neurons into those that depend on x9 (0,6,7,12,13,16,18,21,23,29, weakly
3,4,10) and those that are functions of x0..x8 only (all others); (3) a library of ~400 interpretable features
(order statistics M9,S9,T9 = largest/2nd/3rd of x0..x8; x9 versus them; for each x_j its margin over the
max/2nd/3rd of the other values; rank indicators; prefix maxima; position codes; parity of positions) and a
greedy forward search where each candidate set is scored by a fitted ReLU(linear) model (iterative least squares
on the active set). Each final formula is relu(c0 + sum of at most 4 weighted features) = at most 5 fitted
constants. Some neurons were seeded with a hand-chosen core feature (e.g. relu(S9-x9) gated off when x8 is the max
of x0..x8) before greedy completion. Feature choices are discrete searched choices and are listed in STRUCT.

## Per-neuron R^2 (fresh seed-777 data, 50k sequences)
0:0.951 1:0.986 2:0.892 3:0.683 4:0.639 5:0.901 6:0.943 7:0.999 8:0.723 9:0.761 10:0.752 11:0.798
12:0.999 13:0.923 14:0.871 15:0.908 16:0.996 17:0.995 18:0.925 19:0.892 20:0.765 21:0.983 22:0.911
23:0.979 24:0.792 25:0.898 26:0.842 27:0.927 28:0.840 29:0.950 30:0.782 31:0.977 — mean 0.881.
A least-squares linear probe trained on the true h_10 to reproduce the model's prediction gets 0.898 accuracy on
true h_10 and 0.785 when fed the formula h_10, so the formulas keep most but not all decision-relevant detail.

## Overall picture (inferred)
The state after x8 is a sorted memory of x0..x8: magnitude neurons (17: ~8*M9+2*S9; 31: ~5*S9; 1 and 5: gap
M9-S9) and position neurons that light up when a specific x_j is among the two largest of x0..x8 (x1:15, x3:2,
x6:19, x7:22, x8:14), or when x_j is NOT in the top two (anti-detectors: x0:20, x4:30, x5:24, x8:25), plus
multi-position codes for where the maximum is (27: {x2,x6}; 11: even positions; 10: {x4,x5}; 26: {x0,x2,x4};
28/8: early max) and a parity code for the second-largest (4). The final input x9 is then compared against the
stored order statistics, giving three clean case neurons: 7 = relu(x9-M9) (x9 is the max, so the answer is the
argmax of x0..x8), 16 = relu(x9-S9) (x9 in the top two, answer 9 or argmax of x0..x8), 12 = relu(S9-x9) (x9 not
in the top two, answer is the argsecond of x0..x8). Mixed neurons (0,6,13,18,29: relu(S9-x9) gated by "x8 is
not the max", plus max-position bonuses; 21,23: x9-in-top-two plus position terms) combine the case signal with
position information so the linear readout can pick the right slot.

Weak spots (R^2 < 0.8): 3, 4, 8, 9, 10, 11, 20, 24, 30. These look like distributed position codes whose exact
values probably depend on recency/decay effects not captured by order-statistic features.

## Files opened
TASK.md (read). Files I wrote and ran in this directory: a1.py-a13.py, feats.py, feats2.py, feats3.py, feats4.py,
lib.py, fitlib.py, fit_final.py, refit28.py, featcore.py, chk.py, labtext.py, gen.py, val.py, labels.py, notes.md,
and outputs g1-g6.txt, f1/f2.txt, fit_0.json, fit_16.json, x1.npy, H1.npy. probe.py was imported only. One
background-task output file of my own script (in the session tasks directory) was read.
