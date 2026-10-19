# Notes: what the 32 final hidden neurons do

## Method
I only used `probe.hidden` / `probe.predict` (black box) on Gaussian inputs (seeds 1, 2, 5, 11; never 20261016).
Steps: (1) correlate each neuron with x_j and with one-hot argmax / argsecond positions; (2) finite-difference
gradients of h_10 w.r.t. each input, grouped by position and by rank (largest, second, ...), which showed the
slope on x9 is a fixed constant per neuron and that most neurons depend on the *top-two values of x0..x8*
and on *which position* holds them; (3) ReLU-linear fits (iteratively re-weighted least squares on the
active set) over a catalog of interpretable features (x_j, prefix maxima, top-k of x0..x8, max/second of
"the others", rectified differences, sums, position indicators), with greedy forward selection limited to 4
features + bias (= 5 constants); (4) for "position-code" neurons, fitted a free offset per argmax position,
then replaced the 9 offsets by a single indicator of a searched position set (best split of the sorted
offsets); (5) tables of mean activation by (argmax, argsecond). The final formula family is
`[gate] * ReLU(c . features + b)` (or plain linear / single scale). Constants were fitted on 40k samples
(seed 1) and evaluated on fresh samples. All discrete choices (positions, sets, gates, feature choice) are
listed in `STRUCT` in labels.py.

## Per-neuron R^2 (fresh 50k Gaussian inputs, seed 5, using labels.f exactly)
0:0.809 1:0.941 2:0.917 3:0.954 4:0.760 5:0.973 6:0.999 7:0.876 8:0.938 9:0.998 10:0.813 11:0.591
12:0.995 13:0.799 14:0.874 15:0.933 16:0.972 17:0.577 18:0.959 19:0.981 20:0.877 21:0.530 22:0.999
23:0.980 24:0.912 25:0.819 26:0.997 27:0.846 28:0.753 29:0.761 30:0.750 31:0.997
(mean 0.871, median 0.915). A second fresh set (seed 11, 40k) gave the same numbers within 0.01.

## Overall picture (inferred)
The state after x9 has four kinds of units:
* **Value summaries of x0..x8**: weighted sums of the largest and second-largest (5, 12), the max-second gap
  (26), "x8 is a new record" (9).
* **Per-position top-two flags**: "x_j is among the two largest of x0..x8" (x0: 27, x1/x3: 25, x2: 0,
  x4: 4, x5: 13, x6: 7) and complementary "x_j is NOT in the top two" units that grow as x_j falls
  below the second-largest (x1: 29, x3: 28, x7: 24, x8: 3, 10); plus "x_j is the max" units (x4: 16, x7: 23)
  and a pair unit (x8 with x3/x5/x6/x7: 30).
* **Argmax-position code**: units of the form ReLU(a*max + b*second - ~1.9*x9 + offset), where the offset
  depends on a *set* of positions of the max of x0..x8 (1: {0,2,3}, 2/20: off when max at x8, 8: even
  positions, 15: {1,4}, 14: {0,1,2,3,5}). Together these look like a distributed binary-ish code of where
  the maximum is, so the readout can exclude the max from the top-two candidates.
* **Where x9 falls relative to the top two of x0..x8**: x9 is the new max (22), x9 exceeds the second
  (31, 18, 19), x9 below the second (6, and the -1.9*x9 family above).
The readout plausibly computes "positions in the top two" minus "position of the max", with x9 handled by
the dedicated x9-vs-top-two units. Neurons 11, 17, 21 remain poorly understood (D).

## Caveats
Position-flag neurons (0, 4, 13, 27, 28, 29, 25) depend on the order in which large values arrive in ways a
5-constant formula does not capture (R^2 0.75-0.85). The set choices for position codes were picked by
search on the same training data (declared in STRUCT).

## Files opened / written
Read: TASK.md (only file read outside my own outputs). Used probe.py only via import (not opened).
Written in this directory: feats.py, featlib.py, cat2.py, explore1-5.py, fd.py, rankgrad.py, cond.py,
greedy_run.py, greedy2.py, greedy3.py, tmpl.py, tmpl2-4.py, rank.py, posoff.py, setsearch.py, misc.py,
n21.py, sub.py, resid.py, tab.py, imp.py, imp2.py, imp3.py, mkspec.py, spec.json, final.py, fitted.json,
final_out.txt, genlabels.py, labels.py, check.py, notes.md, and data files x1/H1/x2/H2/G2 .npy and g*.txt
outputs.
