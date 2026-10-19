# Neuron labels for the 32-unit second-largest RNN: notes

## Method
I only used `probe.hidden` / `probe.predict` on Gaussian or hand-built inputs (seeds 1-22 and 424242; never 20261016).
The model is about 99.1% accurate on fresh Gaussian data.
1. **Survey.** For every neuron I looked at correlations with x_t and with the sorted values, and at its mean given the argmax and second-argmax positions.
2. **Splitting off x9 exactly.** h_10 = ReLU(w*x9 + g(x0..x8)), so setting x9 to plus or minus 20 and 25 gives the input weight w and the exact pre-activation g(x0..x8) for each neuron. Twelve neurons have a large |w|. Neurons 7, 9, 3, 17, 18, 20, 22, 28 and 30 have w between -3.7 and -6. Neurons 8 and 13 have w of about +6.9 and +6.0, and neuron 19 has w of about +0.95. All the others have |w| < 0.2.
3. **Fitting g.** I fitted g by least squares on interpretable features: the largest (ps1) and second-largest (ps2) of x0..x8, and their gap counted only when the argmax lies in a position set T. T was searched greedily. With w fixed, h = ReLU(w*x9 + fit) reproduces these neurons with R^2 between 0.975 and 0.999.
4. **The other neurons.** I ran a greedy forward feature search (at most 4 features plus an intercept), using both a linear and a ReLU-of-linear fit, over a library of about 180 features. The library holds rank and record features: x_t's lead over the other prefix values, margins over the second- and third-largest, running maxima, indicators for the argmax / second-argmax position, and "x_t leads the values x_0..x_k". I also used structured families: a "step-t" analogue of the x9 family for x8, and a gap times position-set family with one or two searched sets. I kept whichever won on a fresh test set.
5. **Probing the hard neurons.** For neurons 0, 12, 29 and 31 I measured finite-difference derivatives with respect to each x_t, split by the rank of x_t. I also varied one input at a time on crafted bases.

All constants are hard-coded in `labels.py`, at most 5 per neuron. The discrete choices I searched for are listed in `STRUCT`.

## Per-neuron R^2 (`labels.f` vs `probe.hidden`, 100k fresh N(0,1) sequences, seed 424242)
| neuron | R^2 | neuron | R^2 | neuron | R^2 | neuron | R^2 |
|---|---|---|---|---|---|---|---|
| 0 | 0.770 | 8 | 0.999 | 16 | dead (var 0.003) | 24 | 0.917 |
| 1 | 0.899 | 9 | 0.994 | 17 | 0.986 | 25 | dead (var 0.003) |
| 2 | 0.912 | 10 | 0.783 | 18 | 0.992 | 26 | 0.908 |
| 3 | 0.981 | 11 | 0.926 | 19 | 0.991 | 27 | 0.998 |
| 4 | 0.901 | 12 | 0.835 | 20 | 0.994 | 28 | 0.975 |
| 5 | 0.814 | 13 | 0.990 | 21 | 0.943 | 29 | 0.807 |
| 6 | 0.961 | 14 | 0.991 | 22 | 0.982 | 30 | 0.980 |
| 7 | 0.997 | 15 | 0.874 | 23 | 0.907 | 31 | 0.698 |

Neurons 16 and 25 are predicted as 0. Their R^2 is about 0 because their variance is negligible (they are almost never active).

## Overall picture
- **"How far x9 sits below a reference"** (3, 7, 9, 17, 18, 20, 22, 28, 30). Each is about k*max(0, r - x9). The reference r is the maximum of x0..x8 when that maximum sits in a neuron-specific position set, and the runner-up otherwise. Neuron 7 uses max(x0..x7).
  - The position sets work like a distributed code for where the prefix maximum is. They also give two thresholds, ps1 and ps2, so the readout can tell whether x9 is the largest, the second-largest, or lower. When x9 falls between ps2 and ps1, x9 is the answer.
- **"x9 is in the top two"** (8 and 13). Neuron 8 is x9 - max(x0..x8). Neuron 13 is x9 - (the second-largest of x0..x8). Neuron 19 tracks the overall size of the top values.
- **Step-8 analogues** (14 and 27). Neuron 27 fires when x8 is a new running maximum. Neuron 14 measures how far x8 is below the earlier top values.
- **Position-specific memories** (21, 1, 11, 2, 23; also 26, 29, 0, 15). Each fires when x1, x2, x3, x4, x6 or x7 (or an early x0/x1) is the largest of x0..x8, or among the top two, and scales with the margin.
- **Where the maximum sits, weighted by the gap** (4, 6, 12, 15). These record the gap between the largest and second-largest of x0..x8, weighted by where the maximum is: late for 6, middle for 4, early for 15, positions 2-7 except 5 for 12.
- **A common shape.** Almost everything is a function of ranks and of the positions of the top two values. The typical computation is a running "max so far / runner-up so far" kept with position tags, plus a final comparison of x9 against those two references.

Weak fits: 31 (0.70), 0 (0.77), 10 (0.78), 29 (0.81) and 5 (0.81). They are small or distributed, and they depend on order and recency in ways my families do not fully capture.

## Files opened
- TASK.md
- Files I wrote: explore1.py, ex2-ex11.py, feats.py, g1-g16.py, evalr2.py, labels.py, notes.md
- Saved arrays and JSON fit outputs in this directory
- Background-task output files under /tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/tasks/ (these are outputs of my own scripts)

I did not open probe.py. I used it only through import.
