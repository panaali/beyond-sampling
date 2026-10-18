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

## Method / Results / Claims
(filled after the runs)
