# Neuron labels for the 16-neuron second-largest RNN

## Method
I only used `probe.hidden` / `probe.predict` on Gaussian inputs (seeds 1, 7, 11, 12, 99) and a few crafted inputs:
single impulses at ±e_j, and pairs of impulses (1 at position i, 0.5 or 0.9 at position j).
The model is 94.9% accurate on seed-1 data.
Because the network has no biases, h_10 is positively homogeneous and piecewise linear in x. So I first fitted
linear models of each neuron inside cells defined by the order of the inputs (argmax, arg-second). This showed:
- the useful frame is the **prefix x0..x8** (M, S = its largest and second-largest values; P, Q = their positions)
  plus a direct linear term in x9 (h_10 = ReLU(state after x8 + u*x9));
- most neurons are about g(P)*M + h(Q)*S, where the gain g depends on where the max sits, with a clear
  **period-3 pattern** over positions (e.g. high at x0,x3,x6). That points to a rotation-like recurrence (complex
  eigenvalues near e^{±2πi/3}) acting as a position code for the max;
- some neurons also carry "memory" of particular early inputs (x1, x0/x1/x3, x2) or are pushed down by particular
  inputs (x7, x8, x2/x5). Rank-by-position regressions showed these inputs count at any rank, and they behave like
  max(x_j,0) (the negative part is lost to an earlier ReLU).

Final formulas are all ReLU(linear combination) with at most 5 fitted constants. Discrete choices: a 3-way grouping of
argmax positions into gain classes (1-D clustering of the unconstrained per-position gains, then greedy local search),
and for some neurons a 2-way grouping of the second-max position, or a small memory set A of positions (chosen
among 1–3 candidates suggested by the rank-by-position weights). Constants were fitted by nonlinear least squares on
40k training sequences (seed 11) and checked on fresh data (seed 12 during the search; seed 99 for the final numbers).

## Per-neuron R^2 (fresh data, seed 99, N=50,000)
| neuron | cat | consts | R^2 | gist |
|---|---|---|---|---|
| 0 | A | 5 | 0.969 | x9 below earlier max (gain set {0,2,3,6} high) |
| 1 | B | 5 | 0.840 | max position code (x2/x5 high) minus S and x0,x4,x7 |
| 2 | B | 3 | 0.765 | second-largest of x0..x4 plus memory of x2 |
| 3 | A | 5 | 0.963 | x9 below earlier max (gain set {1,2,4,5} high) |
| 4 | A | 5 | 0.950 | margin of max over runner-up |
| 5 | A | 5 | 0.876 | max position code ({0,3,6,8} high) |
| 6 | A | 5 | 0.965 | x9 below earlier max (gain set {0,1,4,7} high) |
| 7 | A | 5 | 0.963 | max position code ({0,3,6} high) plus S |
| 8 | B | 5 | 0.855 | max position code ({4,7} high) minus x2,x5 |
| 9 | B | 5 | 0.820 | memory of x1 plus max position code ({0,4,7}) |
| 10 | A | 3 | 0.984 | x9 is in the top two (ReLU(8.4 x9 − 6.3 S + 0.2 M)) |
| 11 | B | 5 | 0.845 | max position code ({2,5,8} high) minus S and x7 |
| 12 | A | 5 | 0.948 | margin of max over runner-up (≈20M − 12.7S) |
| 13 | B | 5 | 0.820 | memory of x0,x1,x3 plus max position code ({4,7}) |
| 14 | A | 5 | 0.926 | max position code ({1,2,5,8} high) plus x8 |
| 15 | A | 1 | 0.999 | 27·ReLU(x9 − M): x9 is the new max |

Mean R^2 = 0.906. Unconstrained upper bounds within this feature family (separate gains for every position of
max/second/third) were about 0.80–0.86 for neurons 2, 9 and 13, so their remaining error comes from interactions I did
not capture (pair-impulse tables show order-dependent effects between early positions).

## Overall picture (inferred)
- **Max tracking with a position code.** Most neurons hold the running maximum M, scaled by a gain that depends on
  where M occurred. The gains repeat with period about 3, with drift, so different neurons pick out different phases
  ({0,3,6}, {1,4,7}, {2,5,8}, ...). Together these codes let a linear readout recover the position of the max.
- **Second-max information** comes in as ±S terms. Neurons 4 and 12 mostly encode the margin M − 0.65S. Others add S
  with a gain that depends on its position (5), or carry memory of specific early inputs (2, 9, 13).
- **Last-step comparisons.** Because x9 has no later steps to be encoded, dedicated neurons compare it directly:
  15 = x9 is the new max, 10 = x9 beats the earlier runner-up, i.e. it is in the top two, 0/3/6 = x9 is below the
  earlier max. 0/3/6 have a strong −8·x9 input and phase-coded gains on M, and are silent when the earlier max is x8.
- I did not measure the readout weights, so how the readout combines these is not verified.

## Files opened
- TASK.md (read), probe.py (only imported; its source was never opened).
- My own files in this directory: feat.py, p1.py–p21.py, fitlib.py, final_fit.py, gen.py, labeltext.json, check.py,
  resA.json, resB.json, resB.log, x1.npy, h1.npy, labels.py, notes.md, plus the background-task output file
  for my own final_fit run.
