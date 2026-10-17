# skeleton_quadrature (T103): deterministic quadrature of the top-3 skeleton integral

↑ [Approaches index](../README.md) · [INDEX](../../../INDEX.md) · [TASKS](../../../TASKS.md) · predecessor: [review/stoch_margin (R5)](../../../review/stoch_margin/README.md)

## Question
The accuracy is p = E_S[q(S)]. The skeleton S is the top-3 positions (720 ordered triples) together with their
values v1 > v2 > v3, and q(S) = P(correct | S) integrates out the 7 low inputs.

The question has two parts:
1. Can a deterministic rule over S reach a **>20x MSE gain over Monte Carlo on this integral** (at equal node count)
   on DEV models? The rule may share structure across the 720 position triples: symmetries, position classes, one
   shared value rule, or slicing in the tie direction.
2. If so, does that become a win over plain sampling at 16k-64k FPE, with the cost of q(S) counted?

## Pre-registered alternatives and cheapest tests (AGENTS rule 3; written before building, commit 809c9f0)
1. **q is smooth in the values once the low inputs are integrated, so a few Gauss nodes per triple suffice.**
   Against: thin or heavy-tailed error layers (H-E12). *Test D1:* profiles of q along the tie direction, and the
   error-mass tails.
2. **Positions enter through a few features, so one value rule serves many triples.** Against: each triple is its
   own function. *Test D2:* q(triple, v) for all 720 triples at fixed v.
3. **Even a perfect skeleton rule cannot win much end to end,** because the inner 7-D integral keeps its own error.
   *Test D0:* measure the inner rule's gain g_in. The end-to-end ceiling is g_in / (1 - leverage).
4. **Any gain comes only from stratifying the 720 triples.** *Test D3:* compare stratified MC against the QMC rules.

## Method
All of this is diagnosis with random inputs, on two DEV models, `dev_s0.005_0` and `dev_s0.005_1`. The truths are
from 20M plain samples each (`truth.py`): 0.909559 +- 0.000064 and 0.869853 +- 0.000075.

**Cube coordinates.** U1 = Phi(v1)^10, U2 = (Phi(v2)/Phi(v1))^9, U3 = (Phi(v3)/Phi(v2))^8. Under the true law these
are independent uniforms, and the triple is uniform and independent of them (checked: means 0.500, correlations
< 0.003). In these coordinates p = (1/720) sum_t int_[0,1]^3 q(t, U) dU. The 2-3 tie is the face U3 = 1 and the 1-2
tie is U2 = 1.

**How each rule is scored.** Every rule node computes q from M = 128-1024 iid inner draws. That is an unbiased
stand-in for an exact q, and its noise sum w^2 q(1-q)/(M-1) is subtracted. A rule's MSE is either its variance over
R randomisations (scrambles, shifts, assignments) or its MSE against the truth; both are reported. The MC baseline
is Var_S(q)/N, with Var_S(q) taken from D1. As a check, the MC rule itself scores 1.27 and 1.08 at R = 24.

| file | what |
|---|---|
| `rv.py`, `d1_shape_lib.py` | helpers (copied from review/stoch_margin) and the cube-coordinate maps |
| `d0_inner.py` | D0: scrambled Sobol vs iid for the inner 7-D integral, per skeleton |
| `d1_shape.py` | D1: variance shares and error-mass tails over 20k skeletons (M = 512); 60 lines in U3 with Gauss-Legendre (GL) checks |
| `d2_positions.py` | D2: q for all 720 triples at 24 value points (M = 256); feature shares; SVD |
| `d3_rules.py` | D3: randomised rules mc / strat / sob4 / shared3 / pertri3 / tieGL at N = 720 k |
| `d3b_product.py` | D3b: fully deterministic per-triple product-GL rules, the same for all triples |
| `truth.py`, `gains.py`, `run_diag.sh` | truths, the summary tables, and the exact sequence of commands |

## Results
Reproduce with `bash m16/approaches/skeleton_quadrature/run_diag.sh`, then print the D3 tables with
`python m16/approaches/skeleton_quadrature/gains.py MODEL`. The D3 line for dev_s0.005_1 inside `run_diag.sh` was
stopped after its truth was computed; the R = 24 run replaces it.

**D0. The inner rule** (`d0_inner.py MODEL 300 12`): gain of scrambled Sobol over iid for q(S), per skeleton.

| model | n = 16 | n = 64 | n = 256 |
|---|---|---|---|
| dev_s0.005_0 | 1.97 | 3.10 | 3.13 |
| dev_s0.005_1 | 1.64 | 2.48 | 3.31 |

Time-reversed dimension order gives the same within noise. [measured]

**D1. Where the variance and the error mass live** (`d1_shape.py MODEL 20000 512 60 1024`)

| quantity | dev_s0.005_0 | dev_s0.005_1 |
|---|---|---|
| Var_S(q); leverage Var_S(q)/p(1-p) | 0.0544; 0.66 | 0.0813; 0.72 |
| main-effect share: triple (720 classes) | 0.13 | 0.24 |
| shares of (p2,p3) / time order / p1 | 0.04 / 0.02 / 0.02 | 0.10 / 0.06 / 0.04 |
| shares of U1 / U2 / U3 / d23 | 0.02 / 0.10 / 0.17 / 0.21 | 0.01 / 0.11 / 0.13 / 0.18 |
| error mass with d23 > 0.05 / 0.2 / 0.5 | 45% / 22% / 7% | 37% / 25% / 9% |
| error mass carried by the top 5% / 10% of skeletons | 51% / 80% | 39% / 71% |
| along U3 lines: per-line abs error of GL2 / 4 / 8 / 16 | 27% / 8.5% / 2.7% / 1.6% | 15% / 5.7% / 1.8% / 1.0% |

- [measured] Main effects add up to only about 45% of Var_S(q); the rest is interaction.
- [measured] Along the tie direction alone, q is smooth: an 8-node Gauss rule integrates a U3 line to about 2-3%.
  Pre-registered alternative 1 holds in 1-D.
- [measured] About 20% of the error mass sits in a thin layer of the other tie: 20-24% of it at d12 < d23, and
  20% at d12 < 0.05.

**D2. Sharing across triples** (`d2_positions.py MODEL 24 256`), at fixed value points:
- [measured] The across-triple variance at a fixed v (law half) is 0.049 / 0.079, i.e. **90-97% of Var_S(q)**.
  So the triple-averaged function m(U) explains only 3-10% of the variance. **One value rule (or control variate)
  shared by all triples cannot help.**
- [measured] Share of the across-triple variance explained by position features, as dev_s0.005_0 / dev_s0.005_1:

  | feature | classes | share |
  |---|---|---|
  | p1p2 | 90 | 0.54 / 0.54 |
  | p2p3 | 90 | 0.37 / 0.45 |
  | time order | 6 | 0.21 / 0.25 |
  | order x p2p3 | 216 | 0.56 / 0.73 |
  | last position, or (first, last) | 8, 36 | < 0.06 |

  No small set of position classes captures it.
- [measured] SVD of the centred 720 x 24 matrix: rank 1 / 3 / 8 hold 64-70% / 84-86% / 96% of the energy (the
  noise is 0.2-0.35%). The structure is partly low-rank, but a rank-r surrogate needs r coefficients per triple.

**D3. Rules against MC on the skeleton integral.** Gain = (Var_S(q)/N) / MSE.

| rule (N = 2880, k = 4 per triple, R = 24) | dev_s0.005_0 var-gain / truth-gain | dev_s0.005_1 |
|---|---|---|
| mc (control) | 1.27 / 1.11 | 1.27 / 1.08 |
| strat (every triple, iid U) | 1.42 / 1.33 | 0.83 / 0.87 |
| sob4 (4-D Sobol, triple = floor(720 u0)) | **3.79 / 3.51** | **3.95 / 4.13** |
| shared3 (one N-point 3-D Sobol set spread over the triples) | 1.49 / 1.49 | 2.21 / 2.28 |
| pertri3 (each triple its own 4-point Sobol set) | 3.66 / 3.63 | 3.18 / 3.31 |

With R = 8 on dev_s0.005_0 (`d3_rules.py dev_s0.005_0 8 128 2,4,8`), noisy by about x0.3-2:
- sob4 gives 2.7-3.8 at k = 2, 4 and 8;
- shared3 gives 1.1-5.8;
- pertri3 gives 1.7-2.0 at k = 4 and 8 (15-17 at k = 2, an outlier given the k = 4 and 8 values);
- tieGL rules (U3 by Gauss, deterministic) are **biased** by +0.26 to +1.55 pp, a gain of 0.07-0.9.

**D3b. Deterministic product-GL rules** (`d3b_product.py MODEL 128`), the same rule in every triple.
They are biased high by **+2.0 to +7.6 pp** at every size from 1440 to 23,040 nodes. Examples:

| rule | N | dev_s0.005_0 | dev_s0.005_1 |
|---|---|---|---|
| 1x1x2 | 1440 | +6.5 | +7.6 |
| 2x4x4 | 23k | +2.0 | +2.1 |
| 1x4x8 | 23k | +2.1 | +2.4 |

The error mass sits in thin layers at the U2 = 1 and U3 = 1 faces, between the outermost Gauss nodes and the face,
and in 3-D a product rule with few nodes misses them. [measured] (The convergence is monotone but slow; the rule
was not checked against a fine product grid.)

**Oracle Neyman allocation over triples.** It gains at most about 1.4x on the within-triple part (from the D1
per-triple variances, 28 skeletons per triple, so if anything an overestimate). [measured, rough]

**End to end (pre-registered test 3).** Take a skeleton rule with gain G at N nodes and n inner points per node
(Sobol, gain g_in(n)), at budget B = nN. Then:

MSE / sampling = leverage x n / G + (1 - leverage) / g_in(n)

using the measured leverage (0.66 / 0.72) and the g_in values from D0 [inferred from the measured parts]:

| G | best n | end-to-end gain over sampling, dev_s0.005_0 / dev_s0.005_1 |
|---|---|---|
| 4 (best measured) | 1 | 1.9 / 2.2 |
| 20 (the target) | 1 | 2.7 / 3.1 |
| infinite | 256 | 9 / 12 (needs G infinite at N = B/256 = 64-250 nodes, below 720) |

With one low draw per node (n = 1), the rule is simply a 10-D QMC rule over the inputs: the territory of E24-E26
(1.3-3x on the leaderboard).

## Exhaustion checklist
| # | variant | result | status / reason |
|---|---|---|---|
| 1 | stratify the 720 triples, iid values | 0.8-1.4x | the between-triple main effect is only 13-24% |
| 2 | 4-D Sobol with the triple as a coordinate (sob4) | **3.5-4.1x** | best found |
| 3 | per-triple 3-D Sobol (pertri3) | 1.7-3.6x | k = 4 points per triple is too few for 3-D QMC |
| 4 | one 3-D Sobol set shared across triples (shared3) | 1.1-5.8x (R = 8), 1.5-2.3x (R = 24) | D2: 90-97% of the variance is triple x value interaction |
| 5 | shared value function as a control variate | at most 1.03-1.11x (D2 bound) | same reason as 4 |
| 6 | position classes (p2p3, order, order x p2p3, last ...) | best class leaves 27-46% of the across-triple variance | at most 2-4x on that part, and still per-class value rules |
| 7 | tie-direction Gauss in U3 + Sobol in (U1,U2) (tieGL) | biased +0.3 to +1.5 pp | Gauss in U3 is fine on a line (D1) but too coarse with k <= 8 per triple; the 1-2 layer is also missed |
| 8 | deterministic product Gauss rules, 10 sizes up to 23k nodes | biased +2 to +7.6 pp | thin face layers |
| 9 | Neyman allocation over triples (oracle) | <= 1.4x | the within-triple variance is fairly even |
| 10 | inner rule: Sobol vs iid | 1.6-3.3x | caps the end-to-end gain (see the table above) |
| 11 | low-rank separable surrogate q ~ sum_r a_r(t) b_r(U) | not built | the rank-3 residual is 14-16% (<= 7x on that part); rank 8 (96%) needs >= 8 fitted coefficients per triple, i.e. k >= 8 nodes per triple from noisy q |
| 12 | boundary-layer transforms (double-exponential towards U2, U3 -> 1) | not built | could remove the product-rule bias, but a 3-D product rule needs k >> 8 per triple; the N = B/n budget allows k ~ 1-20 |
| 13 | exact tie-segment walks (integrate d23 exactly given everything else) | done elsewhere | E25 sci_tao: 2.5x at T = 10; E15 tube: 1.3-2.9x |
| 14 | rank-1 lattices (any N) instead of truncated Sobol at N = 720 k | not built | Sobol at non-power-of-2 N loses its balance; a lattice may add a little (<= 2x guess [speculative]), far from 20x |
| 15 | prefix sharing of forward passes across triples | not built | a cost saving (<= 2x) that does not change G |
| 16 | permutation symmetry (relabelling positions) | not applicable | the RNN is order-dependent; D2 shows no reducing symmetry (no small feature class) |

## Status
**Done. Negative.** No skeleton rule came near 20x over MC on the skeleton integral. The best are 3.5-4.1x (sob4)
and 3.2-3.6x (pertri3); every deterministic Gauss-type rule is biased by 0.3-7.6 pp. Even a 20x rule would give at
most about 2.7-3.1x over sampling end to end, because the inner integral holds 28-34% of the variance and Sobol
gains only 1.6-3.3x on it. That is the same class as the existing quadratures.

## Claims
- [measured] Across-triple variance at fixed values is 90-97% of Var_S(q), so a value rule shared by all triples
  cannot help.
- [measured] No small position feature explains the triple dependence (best: 216 classes, 56-73%).
- [measured] The best skeleton rules give 3.5-4.1x over MC at N = 2880 (two DEV models, R = 24).
- [measured] Deterministic Gauss rules in cube coordinates are biased by +0.3 to +7.6 pp up to 23k nodes. q is smooth
  along a single tie line, but the error mass sits in thin face layers in 3-D.
- [inferred] The end-to-end gain is capped at about 2-3x unless G is infinite and the inner rule is perfect.
  T103's ">20x" criterion is out of reach (measured best: about 4x) and would not be enough on its own.

## Alternatives not ruled out
- Checklist rows 11, 12 and 14: low-rank surrogates, boundary-layer transforms and lattices. A combination of a
  layer-adapted transform with lattices *might* reach about 10x [speculative]; that still caps the end-to-end gain
  near 2.5x.
- Only 2 DEV models (sigma 0.005). The gains are noisy at R = 24 (about +-30%).
- q at the nodes uses iid inner draws (unbiased, noise subtracted). A deterministic inner rule is accounted for only
  through g_in.
- The MC means at R = 24 sit +0.17 / +0.22 pp from the truths (about 2 s.e.). The other rules and the 20k-skeleton
  D1 mean (-0.15 +- 0.16 pp) show no bias, so this is probably chance, but it was not chased.

## Compute accounting
Diagnostics only, none charged to any estimate. Forward passes: D0 about 6e6, D1 about 2.5e7, D2 about 9e6, D3 about
1.5e8, D3b about 2.4e7, truths 4e7. Single core on a heavily shared machine (load 10-15), about 3 h wall time.
