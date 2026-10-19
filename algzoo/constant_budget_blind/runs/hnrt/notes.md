# Neuron labels for h_10 of the second-argmax RNN: notes

## Method
I could reach the model only through `probe.hidden` / `probe.predict`, on Gaussian inputs (seeds 1-13, 21; never 20261016).
1. I checked that `hidden(2x) == 2*hidden(x)` exactly. With no biases the network is positively homogeneous of degree 1, so every
   feature I used is degree-1 homogeneous (values, order statistics, maxima over position sets, ReLU of differences).
2. I took finite-difference gradients (step 1e-2) and tabulated the median dh_i/dx_j by (rank of x_j, position j). Inputs ranked
   4th or lower have almost no effect on any neuron. Everything is driven by the top 2-3 values, their positions, and x9.
3. I tabulated the mean of h_i by (argmax position, second-max position) and fit flexible "ReLU of a linear combination" models
   (rank-by-position one-hots, prefix maxima, suffix maxima). These gave upper bounds of about 0.8-0.99 for each family.
4. I ran a greedy forward selection of at most 5 features inside a single ReLU, h ≈ max(0, Σ c_k φ_k), with constants fit by
   iterated least squares on the active set. The dictionary held x_j; overall and x0..x8 order statistics; running max, second
   and third max of x0..x_t; suffix maxima; mod-3 class maxima; record increments; rank-at-position gates; and the maximum and
   second maximum over every subset of positions (about 2000 features). The position sets in each formula are discrete choices
   made by this search. They are declared in STRUCT.
5. I refit the final constants on 80k samples (seed 11) and scored them on fresh data. All constants are hard-coded in labels.py.

## Per-neuron R^2 (labels.f vs probe.hidden, 100k fresh inputs, seed 21)
| neuron | cat | R^2 | constants |
|---|---|---|---|
| 0 | A | 0.967 | 4 |
| 1 | C | 0.875 | 5 |
| 2 | D | 0.798 | 5 |
| 3 | A | 0.963 | 4 |
| 4 | A | 0.951 | 5 |
| 5 | A | 0.894 | 5 |
| 6 | A | 0.965 | 4 |
| 7 | A | 0.972 | 5 |
| 8 | A | 0.935 | 5 |
| 9 | B | 0.841 | 5 |
| 10 | A | 0.990 | 3 |
| 11 | C | 0.876 | 5 |
| 12 | A | 0.950 | 5 |
| 13 | B | 0.858 | 5 |
| 14 | A | 0.951 | 5 |
| 15 | A | 0.999 | 1 |
(The model's accuracy on Gaussian inputs is 0.949.)

## Overall picture (inferred)
- **Where the maximum is, coded by a period-3 phase.** Many neurons hold a copy of the maximum of x0..x8 whose weight depends
  on where that maximum sits. The preferred position sets follow position mod 3, distorted for the most recent steps:
  {1,2,4,5,7,8} (neuron 14), {0,2,3,6,7,8} (5), {0,1,3,4,7} (8, 13), {0,1,4,7} (9), {2,5,6,8} (11), {0,3,6} (7).
  So the hidden state seems to store the argmax as a rotating phase pattern, with the max's magnitude as the amplitude.
- **Margin neurons (4, 12)** carry the largest of x0..x8 minus the second and third largest, i.e. how far the leader stands out.
- **Comparisons with the last input:**
  - 15 = 27·relu(x9 − max(x0..x8)), active only when x9 is the overall maximum.
  - 10 ≈ relu(8.3·x9 − 4.6·second − 2.0·third of x0..x8), active when x9 is roughly in the top two.
  - 0, 3 and 6 ≈ relu(phase-weighted max of x0..x8 + small·second max − 8·x9), silent when x8 was the max of x0..x8. Together
    they say "x9 is not the leader", plus a 3-way phase code of where the leader is.
- The readout presumably combines the phase codes of the leader and the runner-up with these "did x9 or x8 take over"
  signals to find the second-largest position. I did not verify this (I had no readout access beyond `predict`).
- Neurons 1, 2, 9, 11 and 13 mix several subset-maxima and second-maxima. Their single-ReLU 5-feature formulas plateau at
  R^2 ≈ 0.80-0.88, so they probably involve nested ReLUs that this formula class cannot express.

## Caveats
- The greedy subset-maximum features for the C, D and some B neurons are a fitted description, not a verified mechanism.
- I made no causal tests beyond the conditional tables and gradients.

## Files opened
- TASK.md, probe.py (only imported, not read)
- Files I wrote in this directory: util.py, dict.py, dict2.py, featlib.py, fit.py, gen.py, greedy.py, greedy2.py, greedy3.py,
  greedy4.py, addone.py, evalr2.py, s1.py to s11.py, grad.py, c1.json, c2.json, their outputs (g*.txt, *_out.json, grad.npz,
  x200.npy, H200.npy), labels.py, notes.md
- The background-task output file for the first greedy run
