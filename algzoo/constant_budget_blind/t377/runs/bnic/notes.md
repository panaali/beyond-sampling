# Notes: what the 16 final hidden units of the second-largest-position RNN compute

## Method
I only used `probe.hidden` and `probe.predict`, on Gaussian inputs (seeds 1, 2, 5, 7) plus a few hand-made inputs:
single impulses, an all-zero input, and inputs padded with leading zeros (the network has no biases, so leading
zeros leave the state at 0 and give the state after a shorter prefix). Steps:
1. Basic statistics: how often each unit is active, its correlation with each x_j, and its mean as a function of the
   position of the max and of the second-largest.
2. Linear fits on large feature libraries (x_j, running maxima M_k, sorted values, gaps). Inputs plus running maxima
   already explain 0.93-0.97 of the variance of every unit. Impulse tests show a period-2 (even/odd position) signature
   of where the max was.
3. A better basis: gaps T_j = max - x_j, max*[argmax=j], margin*[argmax=j] with margin = max - second-largest,
   ReLU(x9 - max(x0..x8)) and ReLU(second-largest(x0..x8) - x9). A full fit on this basis gives R^2 0.92-0.99.
4. Compression to at most 5 constants. Position sets came from sorting the full-fit coefficients and cutting them
   into contiguous groups. Then a greedy-plus-swap search picked 5 features from atomic and set-sum features, using
   Gram matrices so the search was fast. Every formula is linear with no intercept, because the network is
   positively homogeneous (h(cx) = c h(x) for c > 0). I fitted on 100k samples (seed 1) and tested on 100k fresh
   samples (seed 7).

## Per-unit R^2 on fresh inputs (seed 7, 100k sequences)
| unit | cat | R^2 | constants |
|---|---|---|---|
| 0 | B | 0.924 | 5 |
| 1 | A | 0.971 | 5 |
| 2 | B | 0.932 | 5 |
| 3 | A | 0.965 | 3 |
| 4 | B | 0.953 | 5 |
| 5 | A | 0.945 | 2 |
| 6 | A | 0.991 | 3 |
| 7 | B | 0.938 | 5 |
| 8 | A | 0.964 | 3 |
| 9 | B | 0.920 | 5 |
| 10 | B | 0.903 | 5 |
| 11 | B | 0.923 | 5 |
| 12 | A | 0.986 | 3 |
| 13 | B | 0.889 | 5 |
| 14 | B | 0.887 | 5 |
| 15 | B | 0.942 | 5 |

## Overall picture
The model's own predictions match the true second-largest position on 91% of the inputs.
- **Running max.** One unit (1) is essentially a memory of the running maximum.
- **Gaps below the max.** Most units hold a position-weighted sum of gaps, max - x_j: how far each position's value
  sits below the final max. Each unit weights a different set of positions. A position with a small gap is a strong
  candidate for second place. Because the readout is linear, it can compare these position-specific sums and find
  which position is closest to the max. A unit's value grows when its preferred positions sit far below the max.
- **Where the max is.** Almost every unit also carries a term proportional to the margin (max minus second-largest),
  whose sign and size depend on where the max sits. This often follows an even/odd pattern, like an oscillating
  memory of when the max arrived. It lets the readout rule out the max's own position, and also tells it how close
  the runner-up is.
- **A late max.** A large negative correction appears when the max is x9 or x8. The network has had no time to fold
  a late record into its gap code.
- **Units gated on one input.** Three units (3, 6, 12) measure ReLU(max - x9), with parity- or position-dependent
  margin corrections. Unit 8 measures how far x7 is below the max, and unit 14 how far x4 is below the max. Unit 5
  fires only when x9 is a new record, in proportion to how much it beats max(x0..x8).

## Caveats
- I searched the position sets per unit, so they are discrete fitted choices. They are declared in STRUCT in labels.py.
- The dense units (0, 2, 4, 7, 9, 10, 11, 13, 15) are really 20-parameter position-weighted combinations. Their
  5-constant summaries lose 3-10% of the variance, and their labels describe only the dominant weights.

## Files opened
TASK.md, probe.py (called through import only; I did not read its internals), and my own files in this directory:
s1.py-s24.py, feats.py, fitall.py, labels_template.py, labels_fit.py, meta_part.py, labels.py, the saved
x*.npy / h*.npy / H14.npy arrays, s13.out, s19.out, s23.out, s19.json, s23.json, and the background-task output
files for the s9/s13 runs.
