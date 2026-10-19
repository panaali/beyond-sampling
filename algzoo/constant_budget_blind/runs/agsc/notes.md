# Neuron labels for the 8-neuron second-largest RNN

## Method
I only used `probe.hidden` / `probe.predict` on Gaussian and crafted inputs (seeds 1-35, 777; never 20261016).
1. Basic statistics, correlations with features, and impulse responses (one non-zero input).
2. Hidden-state trajectories: since the RNN has no biases, an all-zero prefix keeps the state at 0, so
   `hidden([0]*(10-t) + x[:t])` gives the state after reading x0..x_{t-1}. I used this only to look at how
   each neuron changes over time. I did not fit the recurrence or estimate weights.
3. Upper bounds: per-(argmax, argsecond) tables and linear models with per-position weights reach only
   R^2 ~0.73-0.84 for six of the neurons. Their residual variance comes from nonlinear interactions over time.
4. Feature search: greedy and beam forward selection over a library (values, sorted values, positions,
   running maxima, "drop" features, positive mass), with ReLU refits by nonlinear least squares. The key
   feature that came out was the **drop below the earlier maximum**: DR_t = relu(max(x0..x_{t-1}) - x_t). A
   variant, DA_t, uses the mean of the two largest earlier values as the reference.
   Ablation (p26.py): replacing each drop feature by raw -x_t or relu(-x_t) lowers R^2 a lot (for example
   neuron 0: 0.71 -> 0.54/0.44; neuron 7: 0.76 -> 0.63/0.60). So the running-max reference matters.
5. Final constants were fitted on 100k sequences (seed 31) and tested on fresh data.

## Per-neuron R^2 (fresh inputs, seed 777, 200k sequences)
| neuron | cat | R^2 | constants |
|---|---|---|---|
| 0 | A | 0.722 | 5 |
| 1 | A | 0.773 | 5 |
| 2 | A | 0.770 | 5 |
| 3 | B | 0.760 | 5 |
| 4 | A | 0.954 | 4 |
| 5 | A | 0.982 | 4 |
| 6 | A | 0.708 | 5 |
| 7 | A | 0.764 | 5 |

Neurons 4 and 5 are clean. The other six are described only to about 70-77% of their variance. Their "A"
means one dominant kind of job; it does not mean the formula is exact. Linear models with about 20-320
parameters reach only ~0.75-0.84 on those six.

## Overall picture
The network (it predicts the second-largest position for only ~43% of inputs) builds a **delay line of
"x_t was not a new top value" signals**. Each neuron is a ReLU of one or a few drop terms at its own
positions:
- x9: neuron 5 (relu(top two of x0..x8 - x9))
- x8: neuron 4
- x7: neuron 3 (with the opposite sign: a large x7 raises it)
- x6: neuron 1, and partly neuron 7
- x5: neurons 7, 0 and 2
- x4: neurons 6 and 0
- x3: neurons 0, 2 and 3
- x2: neuron 6, weakly

Each neuron also adds a magnitude term (top1+top2, the max of x0..x8, or the positive mass of x0..x2).
Neurons 0, 1, 2, 6 and 7 grow roughly linearly with time and grow more after a large early value. A large
value at a neuron's "drop" position pushes it down a lot, often close to resetting it. So the population
records which positions held values below the running top values. A linear readout can then decide which
position holds the second-largest value. The position-specific behaviour has to come from the recurrent
dynamics, because the input weights are the same at every time step. I did not model that mechanism.

## Discrete choices searched
Drop positions, the reference used (running max vs mean of the two largest earlier values), the magnitude
feature and its window, and the three position classes for neuron 2. Details are in STRUCT in labels.py.

## Files opened/created
I opened TASK.md. I imported probe.py through `import probe` but never opened it. I created: feat.py,
fit.py, lib.py, lib2.py, traj.py, p1.py-p26.py, fitfinal.py, writec.py, labels.py, notes.md, and
H6.npy/x6.npy (trajectory data). No files outside this directory were read.
