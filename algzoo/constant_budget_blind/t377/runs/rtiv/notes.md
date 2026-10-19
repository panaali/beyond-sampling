# Neuron labels for the 16-unit second-largest RNN: notes

## Method
I used only `probe.hidden` / `probe.predict` on Gaussian inputs (seeds 1, 2, 3, 7, 11) and one crafted-input test.
Basic facts first: accuracy is 0.91. h_10 is exactly positively homogeneous (h(2x) = 2h(x)), which fits a ReLU net with no biases,
so all formulas are degree-1 homogeneous with no intercept. Ten units are always positive ("dense"). Six are often zero ("gap" units 3, 5, 6, 8, 12, 14).

Steps: (1) correlate with x_t and condition on the argmax / second-argmax position; (2) linear regressions on dictionaries
(x_t, running maxima pm_t = max(x0..x_t), sorted values, record increments up_t = ReLU(x_t - pm_{t-1}), gaps
dn_t = ReLU(pm_{t-1} - x_t)); (3) linear fits per argmax group. These showed that the self-weight of the max depends on
the max's position, with a strong even/odd alternation; (4) greedy search over compact 5-constant families; (5) a nonlinear
ReLU fit (scipy least_squares) for the gap units. Constants were fitted on seed 2 (100k) and tested on seed 3 (100k). Final R^2 is on seed 11 (200k).

Crafted test that could have failed (craft.py): I set the max = 3 at position k and the second-largest = 2 or 0 elsewhere, with the other values near -1.
The gap units (3, 6, 8) are much larger when k is even, and unit 12 when k is odd. In the weak parity, units 3 and 8 drop sharply when the
second-largest drops from 2 to 0, so they then track the second-largest value and not the max. This confirms the parity/blend term.

## Per-neuron R^2 (fresh seed 11, 200k Gaussian sequences)
| n | cat | consts | R^2 | form |
|---|---|---|---|---|
| 0 | A | 5 | 0.861 | max(x0..x8), S, even-max boost, penalty on x0,x4,x5,x7 |
| 1 | A | 5 | 0.958 | delayed running max + S - small sum penalty |
| 2 | A | 5 | 0.901 | delayed max - penalty x0+x7+x8 |
| 3 | A | 5 | 0.920 | ReLU(record - x9), record = M (even) / ~S (odd) |
| 4 | A | 5 | 0.884 | delayed max + S - penalty x1,x5,x7,x8 |
| 5 | A | 2 | 0.947 | ReLU(8.7 x9 - 7.7 max(x0..x8)) |
| 6 | A | 5 | 0.986 | ReLU(~13 (record - x9)) |
| 7 | A | 5 | 0.842 | delayed max + S - penalty on all but x0,x2,x9 |
| 8 | A | 5 | 0.927 | ReLU(record - x7) |
| 9 | A | 5 | 0.850 | max + S, argmax-position adjust, +x6 |
| 10 | A | 5 | 0.854 | max + S - penalty x3,x7,x8 |
| 11 | A | 5 | 0.886 | delayed max - penalty on x1,x2,x3,x4,x6,x9 |
| 12 | A | 5 | 0.932 | ReLU(~12 (record - x9)), opposite parity |
| 13 | A | 5 | 0.865 | max + S, argmax-position boost, penalty on all but x1,x3,x9 |
| 14 | B | 4 | 0.772 | ReLU(6.6 max(x0..x6) + 3.9[max odd]M - 6.6 x4 - 2.6 x9) |
| 15 | A | 5 | 0.881 | delayed max, boost if max at x5/x9, penalty x4,x5,x8 |

The formulas are in labels.py (SPEC holds the form, the discrete choices and the constants). Discrete choices (penalty sets P, argmax sets Q,
target positions, parity) were found by greedy search on seed-1 data and are declared in STRUCT.

## Overall picture (inferred)
- **Delayed max code.** All ten dense units carry a large multiple (about 25-45x) of the running maximum. The max reaches them with a lag:
  a record at x8 counts partly and a record at x9 hardly at all. They also carry some of the second-largest value S.
  Each dense unit then subtracts its own position-weighted sum of the inputs, with a large weight on 1-4 chosen positions and a
  small weight on the rest. So each dense unit is "max minus how big the values at my positions are". A linear readout can
  compare these across units to find which position holds the second-largest value. A full linear model in (x_t, pm_t) reaches R^2 of about 0.93-0.97
  for these units. My 5-constant versions lose about 0.05-0.1, mostly to position-specific self-weights of the max.
- **Gap / record units.** Units 3, 6 and 12 measure ReLU(record - x9), unit 8 measures ReLU(record - x7) and unit 14 roughly ReLU(record - x4).
  Unit 5 fires only when x9 is a new record. These units handle the late positions, where the max code above has not caught up.
- **Parity oscillation.** The stored "record" alternates with the parity of the max's position. In one parity it equals the max M. In the
  other it is a blend of M and the second-largest value S (for units 3 and 8 it is almost pure S). This suggests a recurrent mode with a negative eigenvalue
  that swaps max and second-max information between unit groups at each step. Units 3, 6 and 8 share one phase and unit 12 has the opposite phase.

## Alternatives not ruled out / caveats
The dense-unit labels are compressions. The penalty sets were found greedily and a different 5-constant basis could fit about as well.
The parity mechanism is inferred from input-output behaviour only. Unit 14 is the least understood (R^2 0.77).

## Files opened
- TASK.md (read)
- probe.py was used via import only, never opened
- My own files in this directory: a1.py ... a17.py, feats.py, gap.py, n14.py, n14b.py, craft.py, resid.py, fit.py, check.py,
  labels.py, d1.npz (my own cached probe outputs), notes.md
