# Notes: neurons of the second-argmax RNN (32 hidden units)

## Method
I used only `probe.hidden` / `probe.predict`, on Gaussian inputs (seeds 1, 2, 5, 7) and on uniform and 3x-scaled
Gaussian inputs as robustness checks. I never read probe.py's internals or any weights. The model's accuracy on seed 1
was 99.4%.
1. Screening: for each neuron, correlations with x_j, with one-hot argmax and argsecond positions, and with sorted
   values; tables of the mean activation in each (argmax, argsecond) cell (`tab.py`); per-cell linear fits.
2. Time structure: padding the input on the left with zeros gives earlier hidden states, because the network has no
   biases and the zero state stays zero. This showed that the "position" neurons are tuned to a *lag* (the time
   since the max was seen). For example, one neuron fires when the max was 2 steps ago, at every sequence length.
   I used this only for interpretation; the formulas use h_10 alone.
3. Fitting: greedy forward selection (at most 5 terms) over a bank of interpretable, degree-1-homogeneous features:
   x_j, running maxima M_k = max(x0..xk), running second and third largest, record gaps ReLU(x_k − M_{k−1}), order
   statistics, x_p gated by rank, ReLU(x_p − third-largest), and position sets for gates. The selection used ReLU
   regression (repeated least squares on the active set). Position sets were searched greedily (`psearch` in
   `lib.py`). The final constants are least-squares / ReLU-regression fits on seed 1 (100k sequences), written into
   `labels.py` by `fitall.py`.
4. Evaluation: R² on fresh seed-5 Gaussian inputs (50k sequences), shown below. It was checked again on
   uniform(−2,2) inputs and on 3×N(0,1) inputs (`check2.py`), with similar numbers. The network is
   positively homogeneous (no biases), so every formula is built to be homogeneous too.

## Per-neuron R² (fresh Gaussian test, seed 5)
| n | cat | R² | n | cat | R² | n | cat | R² | n | cat | R² |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | A | 0.962 | 8 | C | 0.779 | 16 | A | 0.997 | 24 | A | 0.700 |
| 1 | A | 0.978 | 9 | A | 0.756 | 17 | A | 0.995 | 25 | A | 0.861 |
| 2 | A | 0.878 | 10 | A | 0.721 | 18 | A | 0.916 | 26 | B | 0.810 |
| 3 | D | 0.597 | 11 | A | 0.808 | 19 | A | 0.870 | 27 | B | 0.894 |
| 4 | D | 0.401 | 12 | A | 0.999 | 20 | A | 0.730 | 28 | A | 0.852 |
| 5 | A | 0.899 | 13 | A | 0.973 | 21 | B | 0.979 | 29 | A | 0.957 |
| 6 | A | 0.955 | 14 | A | 0.761 | 22 | A | 0.904 | 30 | A | 0.739 |
| 7 | A | 0.999 | 15 | A | 0.889 | 23 | A | 0.958 | 31 | A | 0.973 |

## Overall picture (interpretation, [inferred])
The neurons fall into four groups:
- **Value registers for x0..x8:** 17 ≈ 8.1·max + 2.0·second (of x0..x8); 31 ≈ 5·second(x0..x8) + …; 1 = gap
  between the top two of x0..x8 (zero if x8 is the max); 5 = gap + record bonus.
- **Last-step comparators for x9:** these are exact or near-exact ReLU comparisons.
  - 7 = 8.8·ReLU(x9 − max(x0..8)) (x9 is the new max).
  - 16 = 5.7·ReLU(x9 − third) (x9 is in the top two).
  - 12 = 3.1·ReLU(second(x0..8) − x9) (x9 is not in the top two).
  - 21 and 23 are variants of "x9 is in the top two".
  - 0, 13, 29, 6 and 18 are variants of "x9 is not in the top two", reweighted by where the max sits.
  - 9 fires only on mostly negative inputs.
- **Lag / position code:** a family of neurons signals that x_p is the max or one of the top two: 22 (x7),
  19 (x6), 2 (x3), 15 (x1), 14 (x8), 26 (x2, also x0 and x4), 27 (x2 or x6), 11 (x4, x0, x6), 10 (x4 or x5), 8 (x5).
  An anti-family is silent when x_p is in the top two: 24 (x5), 30 (x4), 20 (x0), 25 (x8), 28 (x8 is the max).
  Over time these behave as a clock of "steps since the max or second was seen", which the readout can decode into
  the position of the second largest. Several units also show parity (even/odd position) modulation: 4 and 21.
- **Not understood:** 3 and 4. They are low-variance and only partly fit.

## Caveats
- The greedy feature selection and the position-set search are discrete choices (declared in STRUCT). Some 5-term
  formulas (3, 8, 26, 28) are descriptive fits, not mechanisms.
- I did not test the readout, and I did not ablate any neuron.

## Files opened or created
- Opened (read): `TASK.md`.
- Created and used (all in this directory):
  - scripts: `feat.py`, `lib.py`, `s1.py`–`s30.py`, `tab.py`, `fitall.py`, `check2.py`, `genlabels.py`
  - data and outputs: `x1.npy`, `h1.npy`, `xt.npy`, `ht.npy`, `g1–g3.txt`, `G1–G3.txt`, `p_*.txt`, `q_*.txt`,
    `ps_a.txt`, `ps_b.txt`, `H1–H3.txt`, `final_r2.txt`
  - deliverables: `labels.py`, `notes.md`
- `probe.py` was imported only; I never opened it.
