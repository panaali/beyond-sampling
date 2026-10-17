# cone_enum: enumerate error cones, integrate their Gaussian mass

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [STATUS](../../../STATUS.md)

## Question
Can M_16,10's accuracy be computed deductively as 1 - (sum of Gaussian masses of the "error cones")?
Each error cone is a polyhedral cone on which the network is linear and wrong. They would be found by walking
across facets, not by sampling model inputs. Can this match plain sampling at equal compute?

## Method
- `cones.py`: for a point x, `cone_of` builds the cone with key (full 10x16 gate pattern, predicted class k,
  position of max i, position of 2nd j). It is {A x >= 0} with 160 gate rows, 9 logit rows and 9 order rows.
  It is an error cone iff k != j. `cheb` is a Chebyshev-centre LP (HiGHS). `cone_mass_ahr` is the
  model-free mass integrator: adaptive Gaussian hit-and-run inside the cone (exact 1-D truncated-normal
  moves) learns the cone's shape. Then 1-D preintegration along the samples' mean direction gives
  Phi(-L(y)), and scrambled-Sobol importance sampling over the 9-dim complement uses a Gaussian fitted to the chain.
  Earlier integrators failed and are kept: `cone_mass` (plain preintegration RQMC) and `cone_mass_is`
  (cross-entropy IS). See Failure modes.
- `walk.py`: deterministic seeds. For all 720 (p1,p2,p3) triples, canonical order-statistic inputs are pushed
  across the 2nd/3rd and 1st/2nd ties by bisection, the same probes as `arc_eval.est_probe`. Then a
  best-first walk (GeoCert-style) runs over error cones. Facets are found from the binding rows of the IS points.
  The walk steps 1e-7 across each facet, a forward pass checks for an error, and new error cones are queued
  with priority = mass flux through the facet. `est_cone` / `estimator.py` is the arc_eval/leaderboard
  interface. `DEDUCTIVE = True`: no random model inputs. The integrator's random numbers only sample
  inside an already-built polyhedron.
- Diagnostics (random inputs, **not** in the estimator): `diag_count.py` (number of error cones),
  `diag_tree.py` (step-by-step gate-path tree), `test_mass.py`/`dbg_mass.py`/`dbg2.py` (integrator vs
  brute-force MC), `check_cover.py` (MC check of the walk's enumerated mass).

## Status
**Done. Negative result.** The number of error cones explodes (>= 4e5), and the per-cone integration is expensive (about 630
forward-equivalents per cone) and biased low on thin cones. At 64k forward-equivalents, RMSE is 67-91x worse than sampling.

## Results
(Run with `source .venv/bin/activate`, from `m16/approaches/cone_enum/`.)

1. **Cone counts** (`python diag_count.py`, `python diag_tree.py`; 1M samples, diagnosis only)
   - The 46,678 error inputs fall in **40,887 distinct error cones**. 82% of error inputs sit in a cone seen
     once. Chao1 lower bound on the number of error cones: **>= 444,000**.
   - Even an oracle ordering is not enough. The 10k most frequent error cones hold 34% of the error mass;
     the top 1000 hold 11%.
   - Gate-path tree (prefix patterns): 2 / 34 / 587 / 4.6k / 22k / 70k / 163k / 287k / 416k / 535k distinct
     prefixes at t = 1..10. The Chao1 bound at t=10 is >= 2.4M. Tree search with LP pruning (the
     Hahnloser/Curto pointer) therefore has >= 10^6 leaves, with one LP per node.
2. **Integrator accuracy** (`python dbg2.py`): on a cone of mass 2.0e-5 (MC, 40 hits), adaptive
   hit-and-run plus fitted IS gives 1.4e-5 / 0.9e-5 / 1.7e-5 at 550 / 680 / 1660 forward-equivalents.
   An oracle proposal fitted to the true inside points gives 1.95e-5 +- 0.07e-5, so the IS stage is sound and the
   chain's mixing in needle-shaped cones is the problem. Plain preintegration RQMC and CE-IS were off by
   10^1 to 10^30 (`test_mass.py` history).
3. **Walk coverage on the original model** (`python check_cover.py 2000`, about 3 min):
   2000 error cones cost **1.27M forward-equivalents**. Their true (MC, 4M samples) mass is **0.19 pp of
   the 4.68 pp error (4.1%)**. The integrator reports 0.091 pp (median ratio integrator/MC = 0.34,
   IQR 0.22-0.53 on the 80 cones with >= 20 MC hits). Sampling at 1.27M passes has s.d. 0.019 pp.
4. **ARC protocol** (`python eval_noisy.py 64000`; truths with `true_acc(n=4M)`; budget 64k forward-equivalents):
   see table below. At this budget the walk reaches about 85 error cones, so the estimate is about 0.9997 for
   every model.

| model set | truths | estimate (all) | bias on original | RMSE (6 copies) | sampling RMSE at equal compute | ratio |
|---|---|---|---|---|---|---|
| original | 0.95318 | 0.99983 | **+4.67 pp** | - | 0.083 pp (s.d.) | 56x |
| sigma 0.0025 | 0.915-0.946 | 0.9993-0.9999 | - | **6.42 pp** | 0.096 pp | 67x |
| sigma 0.005 | 0.823-0.931 | 0.9985-0.9999 | - | **10.79 pp** | 0.119 pp | 91x |

Raw log: `eval_noisy_64000.log`, `eval_noisy_64000.json`. Under the leaderboard, `estimator.py` now caps seeding at
half the budget. At budgets <= 21k it finds almost no cones, so the estimate is about 1.0.

## Compute accounting
Everything model-dependent is counted in FLOPs and divided by one forward pass (5,760 FLOPs = 10x(16x16+16)x2+10x16x2):
seed bisection passes (about 10.7k passes for all 720 triples x 2 ties), cone construction (10 matrix
products, about 9 passes), the LP (about 2(m+d)(d+1) x iterations x 4 FLOPs, about 10-30 passes), hit-and-run
(4 K m d per step, about 300-500 passes), IS (about 140 passes), and one forward pass per facet crossing.
The average is about **630 forward-equivalents per error cone** (1.27M / 2000). No model-free precomputation is reused.
Input generation for sampling is not charged to the baseline.

## Claims
- [measured] The original model has >= 40,887 error cones (full-pattern keys), and 82% of the error mass is
  in cones hit once in 1M samples. Chao1 >= 444k.
- [measured] The best possible ordering (top cones by MC frequency) covers only 34% of the error with 10k cones.
- [measured] The walk covers 4.1% of the error mass with 2000 cones at 1.27M forward-equivalents.
  Sampling at that compute has s.d. 0.019 pp; this estimator's bias is -4.5 pp.
- [measured] Under ARC's protocol at 64k forward-equivalents, RMSE is about the full error rate (several pp)
  against about 0.1 pp for sampling.
- [inferred] Even with a free, exact integrator and perfect cone ordering, full-pattern enumeration needs
  >= 4e5 cones x (cone build + 1 LP, about 20 passes) ~ 1e7 forward-equivalents for full coverage. At that
  compute sampling's s.d. is about 0.007 pp. The approach cannot beat sampling on this model.
- [inferred] The hit-and-run integrator is biased low (median x0.34) because the chains do not reach the far
  ends of needle-shaped cones within about 100 steps.

## Alternatives not ruled out
- *Coarser regions might be few.* Keys using only the last step's gates (plus pred, i, j) give 4.5k distinct
  error keys with 3.6% singleton mass (`diag_count.py`). But such regions are unions of cones, not
  convex, so they cannot be integrated as cones. Treating earlier steps analytically (moment propagation)
  is a different approach (see `../propagation`). Not tested here.
- *A better integrator* (Botev minimax tilting, LIN-ESS + HDR from Gessner et al. 2020) would fix the x0.34
  bias but not the count. Not tested. The count alone gives the >= 1e7 cost floor above.
- *The walker's ordering is poor.* At 100 cones it found 0.017 pp, against 0.15 pp for the oracle top-100.
  Better seeds (e.g. vertices of tie hyperplanes) would help by about 10x at most. The oracle bound (34% at 10k cones) still holds.
- *Chao1 is only a lower bound*, so the true cone count may be larger. This makes the conclusion stronger.

## Failure modes
- Cones are needles: the Chebyshev ball radius is about 0.01-0.02 rad, while inside points spread to |y|/s ~ 1.
  Plain preintegration RQMC gives about 0. Cross-entropy IS collapses on weight degeneracy (estimates 1e-30).
  Hit-and-run mixes slowly.
- Fragmentation: about 2.4M gate patterns, most of mass under 1e-6, and the error lies along tie bands that
  cross many patterns. A deterministic budget runs out long before coverage.
- Uncovered mass cannot be bounded without sampling. The estimator is always biased toward accuracy 1.
- On noisy copies the error grows (6-12%), and the bias grows with it.
