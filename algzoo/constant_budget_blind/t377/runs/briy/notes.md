# Neuron labels for the second-argmax RNN (16 hidden units)

## Method
I only called `probe.hidden` / `probe.predict` on inputs I generated (seeds 1, 3, 5, 7, 99, 2027). The model's argmax accuracy on N(0,1) inputs was 0.881.
1. Correlated each h_10 unit with a broad feature set: inputs, sorted values, running maxima M_k = max(x0..xk), the
   prefix second-largest S_k, argmax/second-argmax positions. I also tabulated mean activation against (argmax, second-argmax).
2. Ran linear and ReLU least-squares fits on growing dictionaries: running-max rise/fall terms ReLU(x_k - M_{k-1}) and
   ReLU(M_{k-1} - x_k), min(x_k, M_{k-1}), parity-split maxima M_k*[argmax of x0..xk odd], and
   odd-position running-max increments. Features were chosen by greedy forward selection plus swap refinement, with at most 5 features.
   For the "x_t compared with the earlier max" units I searched a fixed template over (k, t):
   ReLU(a*M_k + b*S_k + [argmax odd]*(c*M_k + d*S_k) - e*x_t).
3. Intermediate states: the network has no biases, so leading zeros leave the state at 0. Feeding
   [0,...,0, x0..x_{t-1}] therefore gives the state after t steps. I used this to inspect how a unit evolves
   over time, for example to see that one unit fires exactly when the running max is 4 steps old.
4. Every constant was refit with an iterative ReLU least-squares fit on 200k sequences. All constants are hard-coded in `labels.py`.
   Every formula is ReLU(linear combination of at most 5 features), and each feature is positively homogeneous in x,
   which matches the bias-free network.

## Per-neuron R^2 (fresh inputs, seed 2027, 100k sequences)
| unit | cat | R^2 | unit | cat | R^2 |
|---|---|---|---|---|---|
| 0 | A | 0.968 | 8 | A | 0.931 |
| 1 | A | 0.986 | 9 | A | 0.921 |
| 2 | A | 0.980 | 10 | A | 0.993 |
| 3 | C | 0.854 | 11 | A | 0.986 |
| 4 | A | 0.922 | 12 | B | 0.820 |
| 5 | A | 0.917 | 13 | A | 0.904 |
| 6 | A | 0.996 | 14 | A | 0.882 |
| 7 | A | 0.943 | 15 | C | 0.746 |

## Overall picture (inferred)
- **Running-max memory.** Units 0, 3, 4, 5, 7, 8, 9 and 13 are always active. Each is approximately a positive
  multiple of the running max of x0..x7 or x0..x8, minus a fading trace of recent inputs. Many carry a term that
  depends on whether the max sits at an odd or even position. Regressing on running maxima gives alternating-sign
  coefficients, which points to oscillating (negative-eigenvalue) recurrent dynamics: the max is stored with a sign
  pattern that flips each step.
- **Comparing the last inputs with the earlier max and second max.** For x9, units 10, 2 and 11 split the cases:
  - 10 = ReLU(x9 - max(x0..x8)), active when x9 is the new max.
  - 2 and 11 are about c*ReLU(R - x9). Depending on the parity of the earlier argmax, R is close to the max or the
    second-largest of x0..x8. The two units use opposite parity choices, so together they cover both "x9 below the max"
    and "x9 below the second-largest".
  - Taken together, these let the readout decide whether x9 lies between S8 and M8, which is exactly when x9 is the
    second-largest.
  - Units 6 (for x7) and 1 (for x6) do the same job for earlier positions.
- **Age-of-max detectors.** Unit 14 fires only when the overall max is at x6. Its intermediate states show it marks a
  max that is 4 steps old. Unit 13 boosts a new max at x8. Unit 10 marks a new max at x9.
- **Not fully understood.** Units 12 and 15, and partly 3, have position-specific linear traces (12: x3; 15: x2 vs x5;
  3: x1, x2, x4). The intermediate states suggest a delay-line or rotation mechanism that carries specific inputs
  forward. My 5-constant formulas capture only 75-85% of their variance.

## Caveats
- Several labels describe the parity-dependent reference as "roughly the max" or "roughly the second-largest". The
  fitted coefficients are blends, for example 1.5*M - 0.5*S.
- The discrete feature choices came from search over about 150 candidate features, so R^2 is slightly optimistic. The
  fresh-data test above guards against overfitting the constants, not the feature choice.

## Files opened
- TASK.md (read), probe.py (only imported, never read).
- Files I created: feat.py, dic.py, dic3.py, dic4.py, dic5.py, forms.py, p1-p19.py, fit.py, fit2.py, labels.py, notes.md.
- Data files: x1.npy, h1.npy, x7.npy, Ht.npy, fit_out.json, sel.json.
