# Neuron labels for the second-largest-position RNN (16 hidden units)

## Method
Only `probe.hidden` / `probe.predict` were used (black box; probe.py internals never opened). I sampled 100k
N(0,1) sequences (seed 0) and checked the model is 95.3% accurate. Then:
1. Simple statistics: each neuron's correlation with each x_k, with the sorted values, and its mean grouped by the
   position of the largest/second-largest input.
2. Crafted inputs: a zero baseline with a spike of 3 at position p and a spike of 2 at position q (all 100 (p,q) pairs).
   This made x9-driven units (4: x9 is a new record; 1, 6, 7, 8: x9 is below the earlier maximum) and x8-driven units
   (5, 9) easy to see, and showed that the remaining units are tuned to *where* the maximum of x0..x8 sits.
3. Upper bounds ("ceilings") for formula families: ReLU of per-position lookup tables of the top values of x0..x8
   (and of the overall top-2 pair (p1,p2)).
4. Searches over interpretable families, every one ReLU(linear) with at most 5 fitted constants: (a) greedy selection
   from a library of contiguous window maxima max(xa..xb) and window second-largest values; (b) cosine tuning of the
   gap D = M8 - s8 by the position p8 of the prefix maximum (w, phi on a grid); (c) arbitrary position subsets.
   Constants were fitted by iterated least squares on the active set (ReLU-aware), on seed-0 data.
5. The final R^2 below is on fresh inputs (seed 424242, 100k sequences), as computed by `evalf.py` on `labels.f`.

## Per-neuron R^2 (fresh data)
| neuron | cat | R^2 | gist |
|---|---|---|---|
| 0 | A | 0.896 | top-2 gap of x0..x8, bump-weighted for max near x3 |
| 1 | A | 0.986 | earlier max (position-weighted) minus x9 |
| 2 | A | 0.963 | running max of x0..x8 |
| 3 | D | 0.544 | position-set code ({x0,x1,x6} up, {x5,x7} down) |
| 4 | A | 0.986 | x9 above earlier max |
| 5 | A | 0.935 | x8 above earlier runner-ups |
| 6 | A | 0.983 | max(x0..x7) minus x9 |
| 7 | A | 0.990 | overall max minus x9 |
| 8 | A | 0.983 | earlier max (early-weighted) minus x9 |
| 9 | A | 0.869 | x8 is a record among x2..x8 (weak unit) |
| 10 | A | 0.919 | max of x0..x8 above x0..x5 region (max late) |
| 11 | A | 0.875 | max of x0..x8 above x1..x6 region (max at edges) |
| 12 | A | 0.876 | max early (x0..x2) / late, not at x3..x6 |
| 13 | C | 0.682 | prefix max plus position-set bonuses |
| 14 | A | 0.894 | top-2 gap weighted for max near x3, plus max |
| 15 | A | 0.829 | max early (x0..x2) |
Ceilings with per-position lookups (many constants) were 0.94-0.996 for most units, 0.69 for neuron 3 and 0.78 for 13,
so 3 and 13 depend on something these families miss.

## Overall picture (inferred)
The network computes from three kinds of signal: (i) a memory of the running maximum of x0..x8 (neuron 2) and
"max minus x9" units (1, 6, 7, 8) that say how far the last input falls below what came before, plus a "x9 is a new
record" unit (4); (ii) units that check whether the most recent inputs (x8) are among the top values so far (5, 9);
(iii) a position code for the maximum of x0..x8: several units (0, 10, 11, 12, 14, 15, and partly 3, 13) are
ReLU(max of x0..x8 minus the max over some window), so each is silenced when the maximum lies inside its window. Together
they form a distributed, phase-like code of where the maximum (and, through the gap and runner-up terms, the
second-largest) sits. The readout can then pick the second-largest position: when x9 is a new record (4) the answer is
the old maximum's position; otherwise the code of the maximum/runner-up positions plus the x8/x9 comparisons decide.
Which units encode the second-largest position directly is not settled (3 and 13 look like they respond to the
runner-up's position but my formulas capture it poorly).

## Files opened
TASK.md (Read). probe.py was imported only, never opened. My own files in this directory: feats.py, lib.py, forms.py,
s1-s12.py, search.py, search2.py, wsearch.py, usearch.py, msearch.py, ssearch.py, ssearch2.py, sub3.py, form1.py,
fitall.py, evalf.py, labels.py, x0.npy/h0.npy (cached probe outputs), and their outputs srA/srB/wsA/wsB/usA/usB/msA/msB/
fA/s10/consts.txt.
