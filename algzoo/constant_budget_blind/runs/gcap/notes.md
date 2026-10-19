# Neuron labels for the 8-neuron second-argmax RNN (blinded probe study)

## Method
I only called `probe.hidden` / `probe.predict` on inputs I generated (Gaussian seed 5 for fitting, 100k sequences;
seed 99 for held-out validation; plus crafted inputs). The model's top-1 accuracy on seed-5 data is 0.426.
Steps: (1) correlations of h_10 with x_t, sorted values, argmax/second indicators; (2) linear and ReLU-of-linear fits
on growing feature libraries (x_t, running max M_t, running second max S_t, "gap" features
ReLU(reference - x_t), pairwise ReLU(x_s - x_t), |x_t|, exponentially weighted running maxima); (3) crafted
probes: single impulses x_t = +-1 with zeros elsewhere, and setting x9 to 0/+-10 to read off the
dependence on the last input (h_10[i] = ReLU(g_i(x0..x8) + w_i x9) with w ~ [0, 0, -0.21, -0.17, -0.24, -1.21, +0.13, +1.37];
neurons 0 and 1 do not depend on x9 at all); (4) zero-left-padded inputs to read the hidden state after k<10 steps,
and lagged correlations between neurons; (5) greedy forward selection and exhaustive 4-term search of
ReLU(sum c_k feature_k) models with at most 5 constants, fitted by iterated active-set least squares (checked with
scipy least_squares, no gain). Formula = ReLU(linear combination of the declared features); no intercept (the
network has no biases, so h is positively homogeneous in x and so are all features used).

Ceiling check: a ReLU-of-linear model with ~150 features (x, M_t, S_t, gaps, all pairwise ReLU differences, |x_t|)
reaches only R^2 ~ 0.83-0.89 for neurons 0,1,2,3,6,7, so the 5-constant formulas (0.71-0.81) recover most of what this
feature family can express; the remainder is compounded ReLU gating that I could not compress.

## Per-neuron R^2 (fresh inputs, seed 99, 100k sequences)
| neuron | cat | constants | R^2 | formula (ReLU of) |
|---|---|---|---|---|
| 0 | B | 5 | 0.724 | 0.70 LEVEL - 0.97 x3 - 1.89 x4 - 2.04 x5 + 1.21 x8 |
| 1 | A | 5 | 0.813 | 1.65 G6 + 0.16 LEVEL - 1.87 R(x4-x6) + 1.52 g6 + 0.44 R(x4-x8) |
| 2 | C | 5 | 0.739 | 0.37 LEVEL - 0.83 x5 - 0.72 x3 + 0.49 g6 + 0.49 R(x8-x9) |
| 3 | B | 5 | 0.780 | 3.00 G5 + 1.34 x7 - 2.02 R(x3-x5) - 1.24 R(x4-x7) - 0.76 R(x1-x7) |
| 4 | A | 2 | 0.963 | 1.49 gap8 + 0.31 gap9 |
| 5 | A | 1 | 0.976 | 1.26 gap9 |
| 6 | B | 5 | 0.709 | 0.46 LEVEL - 0.28 x2 - 2.34 x4 + 0.41 x8 - 0.84 R(x2-x4) |
| 7 | B | 5 | 0.763 | 0.64 LEVEL - 2.41 x5 - 1.47 x6 + 1.29 x9 + 0.86 g4 |

R(z)=max(z,0); LEVEL = 10 * mean_{t=1..9} max(x0..xt); G_t = R(max(x0..x_{t-1}) - x_t);
g_t = R(2nd largest of x0..x_{t-1} - x_t); gap_t = R(mean of 3 largest of x0..x_{t-1} - x_t).

## Overall picture
Input enters mainly through two neurons with opposite signs (neuron 7 gets +1.37 x_t, neuron 5 gets -1.21 x_t; small
negative input weights on 2,3,4, small positive on 6). The recurrent part makes neuron 5's pre-activation a soft running
maximum (a decreasing weighted sum of the sorted earlier values, R^2 0.92 linear in sorted x0..x8), so neuron 5 reports
"how far the newest input falls below the top earlier values" - i.e. evidence that the newest position is NOT a top
(largest/second-largest) candidate. Lagged correlations show a delay line: neuron 4 at step 10 ~ neuron 5 at step 9
(corr 0.97), neuron 1 at step 10 ~ neuron 5 at step 7 / neuron 4 at step 8, neuron 6 at step 10 ~ neuron 1 at step 8
(corr 0.96), neuron 0 at step 10 ~ neuron 7 at step 9 (corr 0.98). Hence each neuron of h_10 carries "position t was
low relative to the running top values" for a specific lag (x9: n5, x8: n4, x6: n1, x5: n3/n7/n2, x4: n6/n0, x3: n2),
superimposed on a positive magnitude signal (LEVEL) that is amplified over time (impulse responses grow roughly
1.3-1.5x per step), so early large values dominate the magnitude. The readout presumably compares these
position-specific "was-low" signals against the shared magnitude to pick a position. Coverage of positions 0-2 is weak,
consistent with the model's low accuracy (0.43). [inferred]

## Files opened
- TASK.md (read). probe.py was only imported, never opened.
- My own files in this directory: lib.py, s1.py-s19.py, greedy*.py, exh*.py, fitfinal.py, refine.py, validate.py,
  labels_template.py, labels_core.py, extra.txt, labels.py, their outputs (g2*.txt, g3*.txt, g4*.txt, e*.txt),
  and data arrays x.npy, H.npy, h0/hn/hp.npy, g.npy, hs.npy.
