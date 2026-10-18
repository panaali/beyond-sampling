# T279: linear-relaxation (CROWN-style) certification tree on the sphere

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [E19 branch_bound](../branch_bound/README.md)

## Question
The model has no biases, so correctness depends only on the direction of x. Partition the sphere into simplicial
cones, each inside one truth cone (constant label), and bound every cone's logit margins with a sample-free linear
relaxation of the RNN. Split until each cone is certified correct or certified wrong. Estimate = certified-correct
mass + ½ uncertified mass, with the deductive guarantee |bias| <= ½ uncertified mass. How many cones, and how many
FLOPs, does that need? Does it scale towards T=10?

## Pre-registration (2026-10-18, before any run)
**Root cones.**
- Truth region (i, j) = {x_i > x_j > x_k for all other k}: the label is j.
- It equals R·1 + cone(e_i, −e_k for k ≠ i, j). Splitting the line into its + and − halves gives 2·T(T−1) simplicial root
  cones: 24 at T=4, 180 at T=10 (the same cells as E19's branch_bound).
- Generators are normalized to unit length. The sign of a margin on a cone equals its sign on the simplex of its
  generators (homogeneity), so all bounds are over that simplex. A linear function's minimum is its minimum vertex value.
- Exact cone masses come from `cone_mass4.mass4_rays` (T109, abs. error < 1e-7) for d = 4. For d = 5 an exact
  orthant method is still needed (to be found or written; d <= 6 is supported per the DRI).

**Bounds.**
- **CROWN:** backward linear relaxation. Intermediate pre-activation bounds come from back-substitution to the input
  simplex at every step; ReLU upper = chord, lower = 0 or identity (adaptive, by the larger side).
- **IBP:** interval propagation from the box hull of the simplex. This is the E19 method family, run on the same tree
  as the comparison.
- Certified correct: lower(m_k) > 0 for every rival k, with m_k = logit_label − logit_k. Certified wrong: some rival
  k has upper(m_k) < 0.

**Splitting.** Best-first by uncertified mass: bisect the longest edge (largest angle between generators). The new
ray r_a + r_b (normalized) is shared by the two children.

**Cost (T196 FLOPs, multiply-add = 2).** Bound propagation is counted per cone from the implemented operations (FLOP
counter in code), plus mass4 ops × 2 (κ = 2, T196). A forward pass at (16,4) is 2,304 FLOPs, and the LB3 FPE scales
the same way.

**Test model:** zoo (16,4) seed 0 (`2nd_argmax_16_4_2^30_0`, ARC public checkpoint). Truth by 4e7 samples, used for
scoring only.

**Go line / stop rule (Fable, from the DRI):**
- At T=4, **STOP** if less than 90% of the "gap >= 0.1" mass is certified (correct or wrong) with fewer than 1e4
  cones. "Gap >= 0.1" means min(M1 − M2, M2 − M3) >= 0.1 in N(0,1) units. The certified fraction of that mass is
  measured by sampling 1e6 points and locating their leaves; this diagnostic sampling is not part of the estimator.
- Otherwise go to (16,5) and extrapolate cones and FLOPs against T.

**T=10 lenient-bar arithmetic, stated now.** The lenient bar under the Directive 8 amendment needs uncertified mass
<= ~0.5 pp at <= 64k FPE ≈ 3.8e8 FLOPs.
- A CROWN bound at T=10 with intermediate back-substitution costs about Σ_t 2·(2H)·t·2H² ≈ 2–4e6 FLOPs per cone with
  a 10-vertex simplex. So the budget pays for only ~100–200 cones, fewer than the 180 roots.
- The roots contain the whole near-tie decision surface. H-M7 counts >= 1e6 linear regions, and about 5% of the mass
  is error near ties.
- Prediction: T=10 is ruled out by more than 3 orders of magnitude unless the T=4 cone count is tiny. T=4 must show
  how the cone count grows with the uncertified mass for the extrapolation.

**Alternatives written first.**
1. Cones that touch the network's decision boundary can never be certified, so uncertified mass falls only as
   fast as the boundary layer thins. Expected: uncertified mass ~ c·n^(−1/(d−1)) per boundary, i.e. slowly.
2. CROWN may be no better than IBP on this recurrent net, because of the error amplification seen in E19 (5e5x).
   Test: certified mass vs cones, CROWN vs IBP on the same tree.
3. Most uncertified mass may sit in the blind region (all inputs nearly negative), where the margins are tiny.

## Status
Pre-registered.
