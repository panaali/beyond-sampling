# Neuron labels for the second-largest-position RNN (h_10, 32 neurons)

## Method
I accessed the model only through `probe.hidden` / `probe.predict`, on Gaussian inputs (seeds 5, 777, 31; never 20261016).
The model's argmax-of-readout accuracy on 100k inputs is 99.45 %. Because the RNN has no biases and uses ReLU, h_10 is
positively homogeneous of degree 1 in x. So every formula is written as ReLU(sum of c_k * feature_k) with no intercept, and every feature
is itself degree-1 homogeneous.

Steps: (1) correlations and linear fits against raw x, sorted values and argmax/second-argmax indicators;
(2) greedy forward selection (ReLU-linear fits on the active set) over a hand-built feature library;
(3) the key step: a "profile" fit h ~ ReLU(sum_j a_j*A*[pA=j] + sum_j b_j*B*[pB=j] + u*x9 + v*C), where A, B, C are the
largest, second- and third-largest of x0..x8 and pA, pB, pC are their positions. This alone gives R^2 > 0.9 for 22 of 32 neurons;
(4) the per-position profiles were compressed into position sets (declared discrete choices), and these were combined
with extra features (running maxima m_t, prefix second-maxima, ReLU(B - x_j), gaps below the running max, position ramps,
sum of inputs) by greedy search with swap refinement, using at most 5 coefficients;
(5) I made the final per-neuron choices by hand to keep them readable, refit on 40k inputs (seed 5) and tested on 60k fresh inputs (seed 777).
I also used zero-prefixed inputs once to look at intermediate states. They showed that the "position j" neurons code distance from the
end of the sequence, not absolute position. I did not use this in the deliverables.

## Per-neuron test R^2 (fresh inputs, seed 777, N=60000)
0:0.952 1:0.991 2:0.862 3:0.710 4:0.646 5:0.882 6:0.949 7:0.999 8:0.814 9:0.765 10:0.855 11:0.878 12:1.000 13:0.969
14:0.881 15:0.893 16:0.997 17:0.995 18:0.910 19:0.871 20:0.750 21:0.984 22:0.907 23:0.972 24:0.743 25:0.930 26:0.858
27:0.936 28:0.825 29:0.947 30:0.751 31:0.968   (mean about 0.88)

## Overall picture
h_10 is a code for three things: (i) the top two values of x0..x8 (A, B); (ii) where they sit; (iii) how x9 compares with
them. That is exactly what the task needs. If x9 > A the answer is pA, if B < x9 < A the answer is 9, otherwise it is pB.
- x9-comparison neurons: 7 = 8.8*ReLU(x9 - A) (x9 is the overall max); 16 = 5.6*ReLU(x9 - B) and 23, 21 (x9 is in the top two);
  12 = 3.1*ReLU(B - x9) (x9 not in the top two); 0, 6, 13, 18, 29 = ReLU(a*A + b*B - c*x9) with position-dependent boosts
  and a strong veto when x8 is the max.
- magnitude/gap neurons: 17 (8A + 2B), 31 (5.2B + 0.7A), 1 and 5 (A - B gap, with special handling of x8 being the max), 9 (negative second-maxima).
- position-of-max/second neurons: 2 (x3), 15 (x1), 19 (x6), 22 (x7), 26 (x2, also x0/x4), 27 (x2 or x6), 11 (x4, x0, x6), 10 (x4/x5),
  14 (x8 as second), plus "x_j not in top two" neurons 20 (x0), 30 (x4), 24 (x5), 25/28 (late positions, x8 veto).
  Several codes spread over sets of positions, some by parity (4: second-max at even positions; 21: max at odd positions).
  So positions look distributed across neurons, not one neuron per position.
- Poorly explained: 3, 4, 8 (and partly 20, 24, 30). They seem to mix decaying (recency-weighted) position information with
  sums of inputs.

## Caveats
Position sets and feature choices were searched on data (declared in STRUCT), so R^2 is slightly optimistic, though the
train and test R^2 agree to within 0.01. Labels give rounded coefficients; the exact ones are in labels.SPECS.

## Files opened
- TASK.md (read)
- probe.py was imported as a module but never opened or read.
- My own files in this directory: feat.py, lib.py, lib2.py, lib3.py, featlib.py, run_g*.py, prof.py, posfit.py, fam.py, auto.py,
  hand.py, probe24.py, s1.json, s2.json, final.json, fits.json, fam.json, auto_*.json, g*.txt / au_*.txt (outputs), d1.npz, d2.npz (cached probe outputs).
