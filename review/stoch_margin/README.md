# review/stoch_margin (T79): independent check of the negative E31

↑ [INDEX](../../INDEX.md) · [TASKS](../../TASKS.md) · reviewed folder: [m16/approaches/stoch_margin](../../m16/approaches/stoch_margin/README.md)

Reviewer: DRI sub-agent (Claude session 011hwZqHr8rxiHqJ7yTBUF1z), 2026-10-17. This folder only; the reviewed
folder is not edited. No post-January-2026 ARC/AlgZoo source was read (nothing to log in `lit/web_log.md`).

## Question
E31 claims that for M_16,10, carrying the low inputs as a distribution given a top-k skeleton cannot beat sampling.
Its evidence: the best top-3 reduction is biased by -0.13 to -0.45 pp; faithful reductions need top-5; and once the
skeleton is that large, integrating the low inputs out cuts the variance by at most ~1.2x. The leverage ceilings
are 1.6x / 1.3x / 1.2x for top-3 / 4 / 5. This review asks:
1. Do E31's numbers hold up when the code is written again from scratch, on different DEV models?
2. What variants exist? This is the exhaustion checklist.
3. Do the most promising untried variants work?
4. Verdict.

## Method
All of this is diagnosis with random inputs, on DEV models only: `dev_s0.005_0`, which E31 also used, plus
`dev_s0.0025_1` and `dev_s0.005_1`, which E31 did not use. No test model was used and no estimator was scored.

The code was written from scratch and does not import `stoch_margin/sm.py`. It differs from E31's in these ways:
- the Jacobian is computed analytically, by a backward product of gate matrices, instead of by finite differences;
- the truncated normals are sampled with `scipy.stats.truncnorm`;
- the Gaussian orthant probability uses 16k draws;
- the leverage uses a pair-covariance estimator;
- the seeds are different.

| file | what |
|---|---|
| `rv.py` | helpers: forward pass, analytic Jacobian, skeleton and conditional sampling |
| `fidelity.py` | re-check of E31 diag_cond/diag_condk. Reductions R0 / lin / add / gauss, plus new oracle laws slin, slin_r, lin_r (below) and the leverage. |
| `leverage.py` | Var(E[f\|S]) / Var(f) by the pair trick, Cov(f(S,U1), f(S,U2)), for 11 skeleton definitions, including record times |
| `anova.py` | variant A: ANOVA (conditional-expectation) additive surrogate of the logits (oracle components) |
| `closure.py` | variant B: Gaussian and Gaussian-mixture closures in hidden-state (pre-activation) coordinates, once or sequentially |
| `slin_nodes.py` | variant C: statistical linearisation plus Gaussian residual, with moments from a FIXED deterministic node set of n points (not an oracle) |
| `table.py` | prints all the tables below from the JSON files |
| `run_all.sh` | runs everything (single core, about 2 h in total) |

Definitions of the new reductions. All are evaluated on the same M = 256 paired low draws as the model, as in E31.
- **slin**: OLS of the 10 logits on [1, x_low] over the draws (statistical linearisation).
- **slin_r**: slin plus an independent Gaussian residual with the true residual covariance. Its law is "a linear
  map of independent truncated normals plus Gaussian noise".
- **lin_r**: the same as slin_r, but around E31's Jacobian lin.

"bias" = mean over skeletons of (q_red - q), in pp of accuracy, +- s.e. "abs" = mean |q_red - q|. The bar is
about 0.1 pp: sampling's RMSE is 0.17 pp at 16k FPE and 0.08 pp at 64k FPE.

## Results
Run everything with `bash review/stoch_margin/run_all.sh [fid|lev|anova]`, then print the tables with
`python review/stoch_margin/table.py`. The JSON files in this folder hold the full outputs.

### 1. Re-check of E31's fidelity numbers (agree)
`python review/stoch_margin/fidelity.py MODEL K NS 256` (NS = 6000 for K = 3, 4; NS = 4000 for K = 5)

| model | K | 1/leverage (CMC gain ceiling) | R0 | lin | add (cut-HDMR) | gauss (true moments) |
|---|---|---|---|---|---|---|
| dev_s0.005_0 | 3 | 1.53 | +0.79 +- 0.17 | -0.08 +- 0.11 (abs 2.5) | -3.56 +- 0.11 | -1.99 +- 0.03 * |
| dev_s0.0025_1 | 3 | 1.43 | +0.12 +- 0.15 | -0.35 +- 0.12 (abs 2.5) | -3.78 +- 0.11 | -1.99 +- 0.03 * |
| dev_s0.005_1 | 3 | 1.38 | +0.13 +- 0.18 | **-0.57** +- 0.12 (abs 3.1) | -3.56 +- 0.11 | -1.75 +- 0.03 * |
| dev_s0.005_0 | 4 | 1.38 | +0.56 +- 0.13 | +0.05 +- 0.06 (abs 1.0) | -0.45 +- 0.04 | -0.98 +- 0.02 |
| dev_s0.0025_1 | 4 | 1.23 | +0.10 +- 0.10 | -0.16 +- 0.06 (abs 0.9) | -0.45 +- 0.04 | -0.92 +- 0.02 |
| dev_s0.005_1 | 4 | 1.24 | +0.04 +- 0.14 | -0.16 +- 0.07 (abs 1.2) | -0.37 +- 0.04 | -0.88 +- 0.02 |
| dev_s0.005_0 | 5 | 1.23 | +0.48 +- 0.13 | -0.03 +- 0.04 (abs 0.4) | -0.06 +- 0.02 | -0.42 +- 0.02 |
| dev_s0.0025_1 | 5 | 1.16 | +0.31 +- 0.10 | -0.02 +- 0.05 (abs 0.5) | -0.06 +- 0.01 | -0.40 +- 0.02 |

\* These use one common set of 16k Gaussian draws for all skeletons, so their errors are correlated across
skeletons. A rerun with fresh draws per skeleton (`FRESH_GZ=1 python review/stoch_margin/fidelity.py dev_s0.005_0 3
2000 256 7`) gives **-2.11 +- 0.06** for `dev_s0.005_0`, matching E31's -2.14. In that rerun lin is -0.49 +- 0.21.

On the same model, E31 had R0 +0.84, lin -0.13, add -3.45, gauss -2.14 and gain 1.54. Agreement is within about
1 s.e. for every reduction except gauss (explained by the footnote). On the two new models, lin at K = 3 ranges from
-0.08 to -0.57 pp, a slightly wider range than E31's -0.13 to -0.45; it is noisy (s.e. 0.11-0.21). [measured]

### 2. Leverage ceiling (agrees, with a second estimator)
`python review/stoch_margin/leverage.py MODEL 400000`. Values are p(1-p)/Var(E[f|S]), the most that a method could
gain over plain sampling if it computed E[f|S] exactly at zero cost and sampled S. The s.e. of the leverage is
about 0.007.

| model | top1 | top2 | top3 | top4 | top5 | pos2 | pos3 | pos4 | pos3+v1,v2 | records (max) | records (max, 2nd) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev_s0.005_0 | 22.5 | 3.25 | **1.54** | **1.34** | **1.25** | 15.1 | 11.2 | 10.5 | 2.85 | 43 | 14.1 |
| dev_s0.0025_1 | 25.2 | 3.48 | 1.43 | 1.24 | 1.16 | 23.9 | 15.0 | 14.9 | 3.02 | 80 | 19.3 |
| dev_s0.005_1 | 22.7 | 2.77 | 1.38 | 1.23 | 1.16 | 7.2 | 5.6 | 5.6 | 2.33 | 27 | 6.5 |

- On `dev_s0.005_0` the pair estimator reproduces E31's 1.54 / 1.34 / 1.25 exactly. It also agrees with the
  256-draw estimator in `fidelity.py` (1.53 / 1.38 / 1.23). [measured]
- Coarser skeletons have higher ceilings. Examples: positions only (5.6-24x), record times (6.5-80x), and the
  "tie-direction slicing" skeleton pos3 + (v1, v2), which leaves v3 random (2.3-3.0x).
- In all of those coarser skeletons, the 2-3 tie itself (v2 - v3) lies inside the integrated part. Between the
  top-3 positions-only skeleton (leverage 0.07-0.18) and top-3 with values (0.65-0.73), the values account for
  75-90% of the skeleton variance. [measured]

### 3. Variant A: ANOVA additive surrogate (oracle components)
`python review/stoch_margin/anova.py MODEL K NS 256 256 24`

g_j(x) = E[L | x_j = x, S] is computed on 24 conditional-quantile nodes, each by 256 inner draws. The surrogate is
L_A = E[L] + sum_j (g_j - E[L]). Its law is a sum of independent terms, so it could be convolved exactly once the
g_j are known.

| model | K | NS | anova | anova, inner noise extrapolated to infinity | anova + Gaussian residual | lin on the same skeletons |
|---|---|---|---|---|---|---|
| dev_s0.005_0 | 3 | 800 | -0.21 +- 0.07 (abs 0.76) | -0.20 | -0.62 +- 0.07 | -0.31 +- 0.28 |
| dev_s0.0025_1 | 3 | 800 | -0.23 +- 0.08 (abs 0.76) | -0.22 | -0.53 +- 0.07 | -0.87 +- 0.36 |
| dev_s0.005_1 | 3 | 800 | -0.11 +- 0.08 (abs 0.88) | -0.11 | -0.37 +- 0.07 | -0.49 +- 0.32 |
| dev_s0.005_0 | 4 | 600 | +0.05 +- 0.03 (abs 0.25) | +0.05 | -0.01 +- 0.03 | +0.01 +- 0.14 |
| dev_s0.0025_1 | 4 | 600 | +0.02 +- 0.03 (abs 0.21) | +0.03 | -0.04 +- 0.03 | -0.07 +- 0.15 |

- ANOVA additivity is much better than cut-HDMR additivity: -0.1 to -0.2 pp against -3.6 pp at K = 3. Its
  per-skeleton error (0.8 pp) is a third of lin's.
- Even as an oracle it stays biased by -0.11 to -0.23 pp at K = 3, which is above the bar on all three models.
  [measured]
- Its components are 6-D conditional expectations. Here they cost 7 x 24 x 256 = 43k forward passes per skeleton;
  no deterministic route to them at a cost of about 1 FPE is known. [inferred]

### 4. Variant B: closures in hidden-state coordinates (non-Gaussian in h)
`python review/stoch_margin/closure.py MODEL 600 1024`

All use exact moments from 1024 particles. The first two columns replace the pre-activation z_t by a Gaussian at a
single time t, then run the exact network. The last two columns re-close at every step.

| model | once at t = 0 | once at t = 1..9 | seq Gaussian | seq mixture by sign of 2 / 4 most ambiguous neurons | seq mixture by full sign pattern |
|---|---|---|---|---|---|
| dev_s0.005_0 | -1.8 | -3.6 to -4.6 | **-49.7** | -41.9 / -31.9 | -34.4 |
| dev_s0.0025_1 | -1.8 | -3.4 to -4.9 | -50.2 | -41.0 / -30.6 | -34.5 |
| dev_s0.005_1 | -1.8 | -3.0 to -4.7 | -45.6 | -38.8 / -28.1 | -31.3 |

(pp, s.e. 0.1-1.3.)
- A closure that is the identity reproduces q exactly (-0.02 +- 0.06), so the pipeline is unbiased.
- A single Gaussian state closure at any time costs 2-5 pp, which is worse than E31's Gaussian on the margins.
- Sequential assumed-density filtering is catastrophic, including the mixture versions with up to 2^16 components.
  The hidden state of this network is far from Gaussian (it holds the running max/second-max behind ReLU
  thresholds). [measured]

### 5. Variant C: statistical linearisation plus Gaussian residual (the only faithful K = 3 law found)
Oracle (`fidelity.py`, slin_r column, NS = 6000): **-0.08 +- 0.02** (dev_s0.005_0), **-0.08 +- 0.02**
(dev_s0.0025_1), **+0.00 +- 0.03** (dev_s0.005_1), abs 0.7-0.8 pp. At K = 4 it is -0.02 / -0.01 / +0.01.

By contrast:
- slin without the residual is **+0.55 to +0.71**, the H-E10 upward bias.
- lin_r, the residual around the mean-point Jacobian, is -0.35 to -0.47.
- The pure Gaussian with the same mean and covariance is -2 pp.

So the shape that the bounded, skewed truncated-normal inputs pass through a linear map is what matters. [measured]

Without the oracle, using a fixed deterministic node set of n points (`python review/stoch_margin/slin_nodes.py MODEL
K 3000 256`):

| model | K | n = 16 | n = 32 | n = 64 | n = 128 | oracle (256 random draws) |
|---|---|---|---|---|---|---|
| dev_s0.005_0 | 3 | -3.69 | -0.27 | -0.17 | -0.11 +- 0.03 | -0.07 +- 0.03 |
| dev_s0.0025_1 | 3 | -3.94 | -0.32 | -0.23 | -0.15 +- 0.03 | -0.11 +- 0.03 |
| dev_s0.005_1 | 3 | -3.77 | -0.30 | -0.17 | -0.09 +- 0.04 | -0.01 +- 0.04 |
| dev_s0.005_0 | 4 | -0.31 | -0.10 | -0.09 | -0.06 +- 0.02 | -0.06 +- 0.02 |
| dev_s0.0025_1 | 4 | -0.25 | -0.07 | -0.07 | -0.05 +- 0.02 | -0.04 +- 0.02 |
| dev_s0.005_1 | 4 | -0.20 | -0.04 | -0.02 | -0.01 +- 0.02 | -0.00 +- 0.02 |

- At K = 3 the bias falls below about 0.15 pp only at n >= 128, i.e. 128 FPE per skeleton.
- Sampling the skeleton and paying 128 FPE per skeleton gives efficiency <= 1.5 / 128, about 0.01 of plain
  sampling.
- A deterministic skeleton quadrature on 16k-64k FPE would have 125-500 skeleton nodes for 720 position triples,
  each with a step-like 3-D value integral. [inferred]

## Exhaustion checklist
| # | variant | tried by | result (bias at K = 3 unless stated) | why it is closed / not pursued |
|---|---|---|---|---|
| 1 | conditional-mean point (R0) | E31 + T79 | -0.03 to +0.84, abs 3-5 | per-skeleton 3-5 pp; the sign depends on the model |
| 2 | Jacobian linearisation at the mean (lin) | E31 + T79 | -0.08 to -0.57 (s.e. 0.1) | above the bar on 4 of 6 runs; abs 2.5-3 pp |
| 3 | lin at the conditional median | E31 | -0.48 to -0.72 | worse than 2 |
| 4 | cut-HDMR additive | E31 + T79 | -3.5 to -3.8 | the gates interact |
| 5 | Gaussian with the true margin moments | E31 + T79 | -1.75 to -2.2 | ceiling for any 2-moment closure in margin space |
| 6 | anchor mixtures of linearisations | E31 | -2 to -13 | off-centre linearisations extrapolate badly |
| 7 | oracle gate-pattern mixture, top 1-50 patterns exact | E31 | -0.12 to -0.36 | about 150 distinct patterns per 256 draws |
| 8 | larger skeleton K = 4 / 5 with lin or add | E31 + T79 | K = 4: -0.16 to +0.05; K = 5: -0.06 to -0.02 | faithful at K = 5, but the ceiling is 1.16-1.25x |
| 9 | lin + Gaussian residual (oracle) | **T79** | -0.35 to -0.47 | the Jacobian at the mean is the wrong slope |
| 10 | statistical linearisation (OLS) without residual | **T79** | +0.55 to +0.71 | Jensen / H-E10 |
| 11 | **statistical linearisation + Gaussian residual (oracle)** | **T79** | **-0.08 to 0.00** | faithful; see 12 for the cost |
| 12 | 11 with moments from n deterministic nodes | **T79** | n = 32: -0.3; n = 128: -0.09 to -0.15 | costs >= 64-128 FPE per skeleton against a gain ceiling of <= 1.5x |
| 13 | ANOVA (conditional-expectation) additive, oracle | **T79** | -0.11 to -0.23 | biased even as an oracle; the components are 6-D integrals |
| 14 | ANOVA + Gaussian residual, oracle | **T79** | -0.37 to -0.62 | worse than 13 |
| 15 | single Gaussian closure of the hidden pre-activation at time t (exact moments) | **T79** | -1.8 to -4.9 | the state is non-Gaussian |
| 16 | sequential Gaussian ADF on pre-activations | **T79** | about -50 | the error compounds |
| 17 | sequential Gaussian-mixture ADF (split on 2, 4 or 16 neuron signs) | **T79** | -28 to -42 | the mixture does not fix it |
| 18 | skeleton = positions only / record times / top-1-2 (leverage) | **T79** | ceilings 2.8-80x | E[f\|S] then contains the 2-3 tie itself, i.e. the whole hard band integral (H-E12); a low-dimensional reduction does not exist, and 1-D v3 slicing of it is variant 19 |
| 19 | exact tie-direction slicing + linear law for the rest (pos3 + v1, v2) | not built | inherits 2/11 per v3 node; ceiling 2.3-3.0x | cost: (number of v3 nodes) x (4.9 FPE for lin, or >= 64 FPE for slin_r) per skeleton exceeds the ceiling; lin bias as in 2 [inferred] |
| 20 | hybrid: x4 exact by 1-D quadrature + lin for the other 6 | not built | fidelity = lin at K = 4 (-0.16 to +0.05), ceiling of K = 3 (1.4-1.5x) | about 8 x 4.9 FPE per skeleton, about 40x the ceiling [inferred] |
| 21 | second-order ANOVA (pair terms) | not built | - | 21 2-D terms, each a 5-D conditional integral; costs more than 13, which already exceeds the budget per skeleton |
| 22 | K = 2 skeleton | not built (E31 crashed) | - | the 2-3 tie is then in the noise; the ceiling is 2.8-3.5x (measured), but the fidelity must be worse than K = 3, where lin and ANOVA already fail |
| 23 | eigenmode / phase coordinates for the closure | not built | - | 15-17 show the obstruction is the non-Gaussianity of the state, which a linear change of coordinates does not remove [inferred]; not tested |
| 24 | moment propagation done analytically (rectified-Gaussian moments) instead of by particles | not built | - | it can only be as good as 16, which used exact particle moments |
| 25 | deterministic skeleton quadrature with sharing across the 720 position triples (q depends on positions through a few features) | not built | - | the one open structural route; needs a > ~20x gain over MC on the skeleton integral at about 128 FPE per node, or > 3x at about 5 FPE with lin's bias |

## Verdict
**Dead (verified)** as a way to beat sampling on M_16,10 at 4k-64k FPE, with one correction to E31's wording.

- [measured] E31's numbers reproduce with independent code on one shared and two new DEV models:
  - lin's K = 3 bias is -0.08 to -0.57 pp;
  - cut-HDMR's is -3.6 to -3.8 pp;
  - Gaussian with the true moments is -2.0 to -2.1 pp;
  - the conditional-MC ceilings are 1.38-1.54 / 1.23-1.38 / 1.16-1.25 for K = 3 / 4 / 5 (E31: 1.54-1.60 / 1.32-1.34
    / 1.20-1.25).
- [measured] **Correction:** "faithful reductions need top-5" is false as stated.
  - Statistical linearisation plus a Gaussian residual is faithful at K = 3: -0.08 to 0.00 pp as an oracle.
  - ANOVA additivity comes within -0.11 to -0.23 pp as an oracle.
  - The obstruction is not fidelity but cost against leverage: the faithful K = 3 law needs moments that cost
    >= 64-128 FPE per skeleton to compute without an oracle (n = 64: -0.17 to -0.23 pp; n = 128: -0.09 to -0.15 pp).
  - Integrating the low inputs out is worth at most 1.4-1.5x.
- [measured] Hidden-state closures (Gaussian, mixture, once or sequential) are far worse, at -1.8 to -50 pp.
- [inferred] The fidelity-leverage trade-off therefore holds in a sharper form. Every faithful law found costs far
  more per skeleton than the <= 1.5x it can return. The coarser skeletons with higher ceilings put the 2-3 tie, the
  part that is actually hard, back into the integrated part.

## Alternatives not ruled out
- **Variant 25:** a skeleton quadrature that shares structure across position triples, so that q is learned from
  few nodes per triple. Not tested. It is the only route under which the slin_r law (faithful) could pay. It would
  need a > ~20x gain over MC on the skeleton integral.
- **A cheaper deterministic route to the statistical-linearisation moments.** Examples: Stein-type average
  Jacobians from a few tangent passes, or moments by sequential exact propagation of a non-Gaussian family.
  The particle-moment ADF (variants 15-17) suggests that propagated moments are poor, but a method that computes only
  E[L], Cov(L, x_low) and the residual covariance, without closing the state, was not tried with n < 32 cleverly
  chosen nodes (Gauss-type rules in 7-D). Sobol with n = 16 fails (-3.7 pp) because 8 regression parameters are
  fitted on 16 points.
- **Coverage.** Only 3 DEV models, at sigma 0.0025 and 0.005. The orig model and the larger-noise DEV copies were
  not checked here; E31 covered orig.
- **Statistical caveats.**
  - The oracle laws use in-sample moments from the same 256 draws on which they are scored. With 8 parameters per
    logit the optimism is about 3% of the residual variance; slin_r's -0.08 could hide a small bias of either sign.
  - The reductions' "exact law" is approximated by the 256 paired draws, as in E31.
- **Cost model.** "Efficiency <= gain ceiling / FPE per skeleton" assumes plain MC over skeletons. A deterministic
  skeleton quadrature could do better (E26's LTQ found 2-3.4x on the full integrand for orig and sigma 0.0025, but
  not on the seeds). That route is variant 25.

## Compute accounting
Diagnostics only, none charged to any estimate. About 6e7 forward passes in total (fidelity about 4e7, ANOVA about
6e7 short rows, leverage 1.6e7, closures 3e6). About 2 h of CPU on one core. Data (npz) are in
`data/review_stoch_margin/` (git-ignored).
