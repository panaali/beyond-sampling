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

## Method (as run)
- `tree.py`:
  - roots, CROWN (full back-substitution), `crown_fwd` (shared forward symbolic intermediate bounds, back-substitution
    only for the final margins), IBP;
  - exact masses via T109 `mass4_rays`; children partition the parent (mass_B = parent − mass_A);
  - best-first splitting of the longest edge; leaf location for the diagnostic.
- `run_t4.py`: certified mass vs cones with the gap >= 0.1 diagnostic (1M sampled points; not part of the estimator).
- `cost.py`: matched cone counts for crown / crown_fwd / ibp, with a FLOP breakdown.
- `dev2_t10.py`: the add-on at (16,10).

The model is zoo (16,4) seed 0. Its truth is 0.99787 ± 7e-6 (4e7 samples); the gap >= 0.1 mass is 0.783.

## Status
**Done. STOP rule fired at T=4; (16,5) not run.** All claims unverified.

## Results
### T=4, zoo (16,4) seed 0 (`results/cost.json`, `logs/t4_crown_to_1e5.log`)
| cones | CROWN uncertified | CROWN FLOPs | crown_fwd uncertified | crown_fwd FLOPs | IBP uncertified | gap >= 0.1 mass certified (CROWN) |
|---|---|---|---|---|---|---|
| 1e3 | 1.000 | 2.8e8 | 1.000 | 9.3e7 | 1.000 | 0.000 |
| 3e3 | 0.973 | 8.5e8 | 0.974 | 2.8e8 | 1.000 | 0.034 |
| 1e4 | **0.811** | 2.8e9 | 0.816 | 9.4e8 | 1.000 | **0.238** |
| 3e4 | 0.589 | 8.5e9 | 0.596 | 2.8e9 | 1.000 | 0.512 |
| 1e5 | 0.354 | 2.8e10 | 0.360 | 9.4e9 | 1.000 | 0.782 |

- **Stop rule:** at 1e4 cones only 23.8% of the gap >= 0.1 mass is certified (needed 90%), so STOP. Even 78% needs
  1e5 cones.
- **Certified wrong mass is 0 throughout.** The network is 99.8% accurate, and its error regions are thinner than
  any leaf.
- **Estimate at 1e5 cones:** certified-correct + ½ uncertified = 0.823, so the bias is −17.5 pp, with deductive bound
  ±17.7 pp, after 2.8e10 FLOPs. That is 1.2e7 forward passes at 2,304 FLOPs, where sampling would reach an s.e. of
  1.3e-5.
- **Soundness:** 0 violations on 200k sampled points (no certified-correct point wrong, no certified-wrong point
  right).
- **Rate:** uncertified mass ~ n^-0.36 (1e4 → 1e5). Reaching 0.5 pp would need ~1e10 cones at T=4, about 5e14 FLOPs
  with crown_fwd.
- **Why it is slow (alternative 1 confirmed):**
  - Every root generator lies on a tie face (e_i: x_j = x_k; −e_k: x_i = x_j; ±1: all equal). A cone that keeps a tie
    vertex can't be certified when the network is wrong at that tie.
  - Interior cones need angular size ~0.03–0.1 rad before CROWN decides them: 49% of interior cones of radius 0.1 and
    85% of radius 0.03 are decided.

**Cost breakdown (DRI / Astra question).**
- Full CROWN: 1.42e5 FLOPs per cone at T=4, of which 99.7% is bound propagation (back-substitution) and 0.3% cone
  masses.
- Shared forward bounds (crown_fwd): 4.7e4 FLOPs per cone (3.0x cheaper) and almost the same certified mass
  (0.360 vs 0.354 uncertified at 1e5). At T=4 they cut the cost about 3x at no loss.
- IBP (E19's family): 8.8e3 FLOPs per cone, but it certifies **nothing** at any cone count up to 1e5 (alternative
  2: CROWN >> IBP, but CROWN is not enough).

### T=10 (add-on, DRI; independent check of T289), DEV2 fused_s0, 180 root sectors (`results/dev2_fused_s0_t10.json`)
| bound | median width/margin [p10, p90] | sectors decided | total cost |
|---|---|---|---|
| IBP | 1.2e5 [6.7e4, 5.6e5] | 0/180 | 632 FPE |
| crown_fwd | 2.1e5 [2.2e4, 7.7e5] | 0/180 | 9,764 FPE |
| CROWN (full) | 972 [244, 3,294] | 0/180 | 32,247 FPE |

- Width = hi − lo of each rival margin bound over the sector; margin = |m_k| at the sector's centroid direction.
- All bounds are sound at the centroid.
- At T=10 the shared forward bounds are no tighter than IBP; only full back-substitution helps, and it is still
  1e3x too loose.
- This agrees with T289 in kind (forward ~1e6, CROWN 3–7e3, with T289's own code and definition).

### T=10 lenient-bar arithmetic (Directive 8 amendment)
The bar is uncertified mass <= ~0.5 pp at <= 64k FPE ≈ 3.8e8 FLOPs.
- Full CROWN on just the 180 root sectors already costs 32k FPE, half the budget, and decides none of them.
- At T=4, getting even 64% of the mass certified took 1e5 cones and 2.8e10 FLOPs. T=10 has 2.5x the dimension, and
  the bounds are 1e3x looser on whole sectors.
- **Ruled out by many orders of magnitude.**

## Compute accounting
FLOPs are counted in code: matmuls at 2 per multiply-add, elementwise ops at 1; mass4 ops × 2 (κ = 2, T196).
CPU about 1.5 h in total (the T=4 tree to 1e5 cones for three bounds, the diagnostic location, the aborted 3e5 run
and the T=10 add-on). One heavy process at a time, `ulimit -v 2000000`.

## Claims
- C1 [measured] At (16,4) zoo0, a sound CROWN certification tree certifies 19% of the mass with 1e4 cones and 65%
  with 1e5 (2.8e10 FLOPs). The pre-registered stop rule fires (gap >= 0.1 certified 23.8% < 90% at 1e4).
- C2 [measured] Uncertified mass falls as ~n^-0.36. Certified-wrong mass stays 0, so the estimate's deductive bound
  is ½ the uncertified mass (17.7 pp at 1e5 cones).
- C3 [measured] Back-substitution is 99.7% of the cost. Shared forward bounds cut it 3x at T=4 with no loss, but at
  T=10 they are as loose as IBP (median width/margin 2e5 vs 1e5; CROWN 1e3).
- C4 [measured] IBP (E19 family) certifies nothing at T=4 up to 1e5 cones.
- C5 [inferred] Linear-relaxation trees are closed for M_16,10 under the Directive 8 budgets. The tie-face geometry
  forces leaves to shrink towards every truth boundary, and relaxation looseness grows with T.

## Alternatives not ruled out
- Tighter relaxations (optimised α / α-CROWN, LP/MILP per cone, splitting on ReLU gates instead of input space)
  could cut the cone count. But the tie-face vertices cap any input-space tree (alternative 1), and gate-splitting
  meets H-M7's >= 1e6 regions.
- Better split heuristics (along the margin gradient rather than the longest edge) were not tried.
- (16,5) was not run, per the stop rule.

## Failure modes
- Leaf location for the diagnostic is a Python loop over internal nodes; it stalled at 3e5 leaves (run aborted;
  the CROWN data to 1e5 are kept).
- The stable interface used by T289 is `tree.py` load/roots/crown/crown_fwd/ibp/Flops/Tree. `return_bounds=True`
  was added without breaking the old calls.
