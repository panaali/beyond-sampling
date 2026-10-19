# Neuron labels for the 16-neuron second-largest RNN: notes

## Method
The only access to the model was `probe.hidden` and `probe.predict`, called on Gaussian inputs drawn with seeds 1, 2, 5, 6, 7, 8 and 11. I never opened `probe.py`. The probe's predictions match the true argmax-of-second on 95.3% of N(0,1) inputs.

1. **Screening** (e1, e2, e3, e4): correlations, mean of each neuron grouped by the position of the max and of the second-largest, and linear fits of each neuron inside each "position of the max" group. The useful features were order statistics: M, S and T, the largest, second-largest and third-largest of x0..x8 (or of x0..x9), the position of the max, and x9 taken separately. Within a group, most neurons are almost exactly linear in (M, S, x9) once a ReLU is applied. The coefficients change smoothly with the position of the max, and many of them look like a cosine of position with a period of about 8 to 9 steps, which would come from a rotating "time since the max" representation.
2. **Further screens** (e5 to e10): "max minus x_j" kernels, rank-by-position one-hot tables, ReLU(x_j) features, running maxima and running second maxima, and an exhaustive search over 4-feature subsets for neurons 5 and 9.
3. **Parametric fitting** with at most 5 real constants per neuron:
   - `fit.py` and `run2.py` fit the cosine templates with scipy least_squares.
   - `code.py` fits templates in which each position gets an integer code u ∈ {-2,...,2}. The codes are found by coordinate descent, and the inner step is an active-set least-squares fit. These integer codes are the declared discrete choices.
   - Every R² below is measured on fresh inputs (seed 11, 100k sequences; `evalr.py`).

## Templates (M, S, T, am, a2 are computed over x0..x8 unless noted)
- **F+ (neurons 1, 6, 7, 8):** `b*ReLU(S + w(am)*(M-S) - x9)` with `w(am) = c0 + A*cos(ω*am + φ)`. The neuron fires when the last input falls below a "running top value", which is a mix of the max and the runner-up. The mix depends on where the max sits.
- **F- (neuron 4):** `b*ReLU(x9 - S - w(am)*(M-S))`. It fires when x9 is the new maximum.
- **F2 (neuron 2):** `b*(S + w(am)*(M-S))`. This is a pure memory of the top value, and x9 has no effect on it.
- **Q2 margin neurons (0, 10, 11, 12, 14):** `ReLU((M-S)*(a1+b1*u1[am]) + (S-T)*(a2+b2*u2[a2]) + g*T)`. The gap between the largest and the runner-up is gated by where the max sits. The u1 code is close to a cosine bump, and each neuron goes silent at a different position: 7, 3, 4, 5 and 7 respectively.
- **R2 position-code neurons (3, 13, 15):** `ReLU(v1*(a1+b1*u1[p1]) + a2*(v2+v3) + b2*(v2*u2[p2]+v3*u2[p3]) + e)`. Here v1..v3 and p1..p3 are the top three values of x0..x9 and their positions.
- **Neuron 5:** `ReLU(12.9*x8 + 1.14*M7 - 7.99*S7 - 2.56*mean(x0..x7) - 1.45)`.
- **Neuron 9:** `ReLU(17.1*x8 - 15.6*M - 0.735*T7 - 0.226*x9 - 0.18)`. This is effectively an indicator that x8 is the max of x0..x8, multiplied by about 1.5*x8.

## Per-neuron R² (fresh inputs, seed 11, N=100000)
| neuron | cat | template | R² |
|---|---|---|---|
| 0 | A | Q2 | 0.912 |
| 1 | A | F+ | 0.987 |
| 2 | A | F2 | 0.986 |
| 3 | C | R2 | 0.665 |
| 4 | A | F- | 0.997 |
| 5 | A | x8 detector | 0.921 |
| 6 | A | F+ | 0.985 |
| 7 | A | F+ | 0.998 |
| 8 | A | F+ | 0.984 |
| 9 | A | x8-is-max detector | 0.919 |
| 10 | A | Q2 | 0.859 |
| 11 | A | Q2 | 0.806 |
| 12 | A | Q2 | 0.840 |
| 13 | C | R2 | 0.757 |
| 14 | A | Q2 | 0.874 |
| 15 | B | R2 | 0.801 |

## Overall picture (inferred)
The final state holds three kinds of information:
- **x9 against the running top values.** Neurons 1, 6, 7 and 8 measure how far x9 falls below the running top value, and neuron 4 measures how far x9 rises above the max. Together they let the readout decide whether x9 is the max, the second-largest, or neither.
- **The top value of the prefix, and where the max sits.** Neuron 2 holds the top value. Neurons 0, 10, 11, 12 and 14 hold the max-versus-runner-up margin, and each is silent for a different position of the max. Their position profiles look like phase-shifted cosines, so as a group they give a population code for the position of the max.
- **The position of the second-largest value.** Neurons 3, 13 and 15 carry higher-frequency codes for where the large values sit. Neurons 5 and 9 report whether x8 is the max or the runner-up.

The second-largest position can then be read out by combining "where the max is" with "where the runner-up is". The margin terms tell the readout whether the max and the runner-up are close enough to matter.

The position dependence comes in a family of near-cosine profiles with a period of about 8.6 positions. That suggests a rotating recurrent mode, but I did not verify this mechanism.

## Weaknesses
- Neurons 3 and 13 are only partly explained: R² 0.66 and 0.76.
- The Q2 and R2 position codes are integer lookup tables found by search, so they describe the neurons rather than derive them from a mechanism.
- The F templates miss slightly when the max sits at x8.

## Files opened or created (all inside the working directory)
- Opened: TASK.md.
- My own scripts and outputs: feat.py, e1.py to e10.py, fit.py, run.py, run2.py, code.py, evalr.py, out_q.txt, cq8.txt, cqg.txt, cr2a.txt, cr2b.txt, cq2a.txt, cq2b.txt, x2.npy, h2.npy, labels.py and this notes.md.
- Background-task output files were also read: they are the harness's copies of my own script outputs.
- probe.py was imported but never opened.
