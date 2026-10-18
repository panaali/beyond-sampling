# random_control — T232: trained M_16,10 vs random weights of the same architecture (diagnostic)

↑ [TASKS](../../../TASKS.md) · [width16_probe](../width16_probe/README.md) · [SCALING_OBSTRUCTIONS](../../SCALING_OBSTRUCTIONS.md)

**All claims unverified.** Diagnostic only; no estimator.

## Question
How far below "generic" is the trained M_16,10? Does training create structure that cuts the growth of exact prefix
regions well below what random weights of the same architecture give?

## Pre-registration (written 2026-10-18, before any model below was run)
**Models** (H = 16, no biases, h0 = 0, readout 10 x 16):
- (a) the target `alg_zoo.example_2nd_argmax()`;
- (b) `frob0..4`: i.i.d. N(0,1) entries, each matrix (W_ih, W_hh, W_out) rescaled to the trained matrix's Frobenius norm
  (rng seeds 100..104);
- (c) `row0..4`: i.i.d. N(0,1) rows rescaled so each row of W_hh, W_ih (a scalar per neuron: random sign, trained
  magnitude) and W_out has the trained row's norm (seeds 200..204);
- (d) `perm`: the trained W_hh with entries permuted at random within each row (seed 300); `flip`: the trained W_hh with
  each entry's sign flipped with probability 1/2 (seed 301). W_ih and W_out are trained.

**Measurements**, per model: exact prefix region counts at k = 2, 3, 4 (k = 5 if a model's k = 4 run takes < ~1 min,
so k = 5 stays under ~10 min), using `width16_probe/prefix.py` (estT geometry, the T192 start). Per-step growth.
Mass coverage N_delta at k = 4 for delta = 1e-2, 1e-4, 1e-6 (exact leaf masses summed per gate word). 2nd-argmax
accuracy at T = 10 from 1e6 samples (diagnostic). A theoretical upper bound on the prefix region count at each k (below).

**Upper bound.** At step t (inputs x_0..x_t, dimension t+1) each existing region is cut by at most H = 16 central
hyperplanes, giving at most min(2^H, 2 sum_{i=0}^{t} C(H-1, i)) pieces (Zaslavsky/Cover count for a central arrangement
in R^{t+1}). Step 1 gives 2 regions (sign x_0). So N_k <= 2 prod_{t=1}^{k-1} min(2^16, 2 sum_{i<=t} C(15, i)) =
64 / 15,488 / 1.78e7 / 6.9e10 for k = 2 / 3 / 4 / 5. This holds for any weights.

**Decision (descriptive, pre-registered).** Let g_trained(k) be the target's growth (x23.3 at k = 3, x26.6 at k = 4) and
g_rand(k) the median over the 10 random draws.
- **Training cuts growth strongly** if g_rand(4) >= 2 x g_trained(4) (random >= ~x53/step) and N_1e-4 at k = 4 is >= 3x
  higher for random than for trained.
- **No training effect** if g_rand(4) is within ±30% of g_trained(4).
- Otherwise **moderate**. The (d) controls say whether any effect comes from W_hh's row structure (perm) or its signs (flip).
**Alternative explanations to keep in mind:** a difference may come from scale (an effective gain, the fraction of
active neurons) rather than "structure". Row matching (c) controls part of that. Random networks may also be mostly dead
or saturated, giving FEWER regions; that would mean training increases complexity.

## Method
`rc.py model k [mass]` builds the model as pre-registered, runs `width16_probe/prefix.py` (unchanged; it reproduces T192
and the independent T55 exact-arithmetic recount at k <= 4), and at k = 4 sums exact leaf masses (estT.mass, d = 4) per gate
word. `drive.sh` runs all 13 models sequentially (10-min timeout each). `summary.json` holds the table below. k = 5 was
**not run**: k = 4 geometry takes 26-36 CPU-s per model, so k = 5 would be ~15 min each, over the pre-registered ~10 min.

## Results (unverified)
| model | regions k = 2 / 3 / 4 | growth k3, k4 | N_delta k = 4 (1e-2 / 1e-4 / 1e-6) | inverse-Simpson | acc (1e6, diag.) | active frac |
|---|---|---|---|---|---|---|
| target | 34 / 792 / 21,032 | x23.3, x26.6 | 1,737 / 7,021 / 13,164 | 112 | 0.9529 | 0.77 |
| frob0 | 34 / 838 / 24,550 | x24.6, x29.3 | 8,616 / 18,642 / 22,742 | 540 | 0.1004 | 0.52 |
| frob1 | 34 / 828 / 23,988 | x24.4, x29.0 | 10,350 / 19,558 / 22,743 | 783 | 0.1030 | 0.36 |
| frob2 | 34 / 832 / 23,900 | x24.5, x28.7 | 9,939 / 19,079 / 22,400 | 811 | 0.1140 | 0.39 |
| frob3 | 34 / 818 / 22,666 | x24.1, x27.7 | 8,945 / 18,034 / 21,281 | 514 | 0.1170 | 0.42 |
| frob4 | 34 / 810 / 23,394 | x23.8, x28.9 | 7,680 / 16,135 / 20,022 | 337 | 0.1188 | 0.56 |
| row0 | 34 / 874 / 25,778 | x25.7, x29.5 | 4,930 / 10,486 / 15,354 | 398 | 0.0841 | 0.32 |
| row1 | 34 / 762 / 19,354 | x22.4, x25.4 | 3,191 / 6,832 / 10,473 | 580 | 0.0971 | 0.52 |
| row2 | 34 / 808 / 23,294 | x23.8, x28.8 | 5,224 / 10,008 / 14,403 | 852 | 0.0997 | 0.47 |
| row3 | 34 / 788 / 21,932 | x23.2, x27.8 | 4,611 / 9,335 / 13,602 | 451 | 0.0695 | 0.48 |
| row4 | 34 / 886 / 26,074 | x26.1, x29.4 | 5,239 / 10,711 / 15,439 | 545 | 0.0999 | 0.70 |
| perm0 | 34 / 804 / 21,972 | x23.6, x27.3 | 4,269 / 9,421 / 14,023 | 233 | 0.1345 | 0.68 |
| flip0 | 34 / 770 / 20,418 | x22.6, x26.5 | 3,029 / 6,420 / 10,013 | 381 | 0.1006 | 0.47 |

Upper bound (any weights): 64 / 15,488 / 1.78e7 at k = 2 / 3 / 4 (pre-registered). At k = 2 every model has exactly 34 regions, the
maximum for the two half-planes {±x_0 > 0} each cut by 16 lines (2 x 17). So the k = 2 bound above is loose by 2x. Mass sums
are within 5.4e-11 of 1 at k = 4. The target's k = 4 coverage reproduces T222 exactly.

**Verdict under the pre-registered rule: no training effect on region growth.** Median random growth is x24.2 (k = 3)
and x28.9 (k = 4), against the trained x23.3 and x26.6, within ±30% (it is the same law). Random draws have 19,354-26,074 regions
at k = 4, against the target's 21,032. Mass coverage differs only *moderately*: the median random N_1e-4 is 13,423, 1.9x the
target's 7,021 (the "strong" line was 3x). The difference is larger at coarse delta (N_1e-2: 3,029-10,350 vs 1,737, i.e.
1.7-6x; inverse-Simpson effective count 233-852 vs 112) and shrinks toward delta = 1e-6 (10,013-22,743 vs 13,164, i.e. 0.8-1.7x).
The structure-destroying controls (perm, flip) sit between the target and the random draws on coverage, and are
generic on counts.

**Reading.** [measured] The trained M_16,10 has a generic number of linear regions for its architecture: training does not
reduce the exponential growth of the region count (x23-30/step is what random weights give too). [measured] What training
does is put more mass on fewer regions at coarse resolution: the heaviest few thousand regions hold 99% of the mass (vs
~5-10k for random). At the precision a competitive estimator needs (delta <= 1e-4) the advantage is <= 2x. [inferred] Region
enumeration therefore cannot exploit "training made it simple". Any exploitable structure must be in *what the regions
compute* (the mechanism), not in how many there are.

## Claims (unverified)
- [measured] Random Frobenius-matched and row-matched draws give region growth x22-26 (k = 3) and x25-30 (k = 4), the same as
  the trained target (x23.3, x26.6).
- [measured] Trained mass concentration at k = 4: N_1e-2 1.7-6x and inverse-Simpson 2-8x smaller than random; N_1e-4 ~1.9x;
  N_1e-6 0.8-1.7x.
- [measured] Accuracy (diagnostic): trained 0.953, random 0.07-0.12 (chance 0.1), perm 0.13, flip 0.10.

## Alternatives not ruled out
- k <= 4 only; growth could diverge between trained and random at larger k (the trained step factor rises, x23 -> x27 -> x30).
- Gain/sparsity: random nets have fewer active units (0.32-0.70 vs 0.77 at T = 10), which affects masses more than counts. Not
  controlled beyond row matching.
- Other random ensembles (orthogonal W_hh, matched spectrum) were not tried.

## Compute accounting
13 models x (k = 2, 3, 4 geometry + k = 4 masses + 1e6-sample accuracy): ~31 CPU-min total, single thread. No estimator cost.
