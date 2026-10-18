# T323: candidate dossier for the R1 continuation estimator (T317c)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

**Status: draft for the DRI (2026-10-18). Not reviewed and not confirmed. Author: T241 worker, who also wrote the
independent re-implementations T312 (R1W V1) and T320 (T317c).** Every number below cites its source. Numbers marked
[inferred] were not measured directly.

"Roughly match" is our phrase, not ARC's: the talk says "match" (§1).

## Question
Does the project's best deductive estimator of the accuracy of an AlgZoo RNN meet ARC's stated bar, and how far
short of it does it fall?

## 1. The problem and ARC's bar
**Model.** M_16,10 is AlgZoo's 1-layer ReLU RNN with hidden size H=16, sequence length T=10 and no biases. It is
trained to output the position of the second-largest of 10 i.i.d. N(0,1) inputs. Its accuracy is p ≈ 0.953. The
estimators are scored on *retrains*: models from the same recipe with other seeds.

**ARC's bar.** Source: [`lit/web_arc/hilton_talk_2026-09_notes.md`](../../../lit/web_arc/hilton_talk_2026-09_notes.md).
The quotes are approximate, taken from an auto-generated transcript.
- **The goal.** "produce a mechanistic estimate of the accuracy ... match random sampling in terms of the like amount
  of computation relative to like mean squared error" [33:12]. Hilton calls this "as yet completely open" [33:00].
- **Which models.** Models come from "taking a random seed or maybe adding noise to the weights" [33:22–33:31]. MSE
  is averaged over "randomness over the weights" [35:29]. No noise level, copy count or seed list is given.
- **What counts as mechanistic.** "deductive rather than inductive" [24:57]; "just like no sampling" [25:02]; "not
  allowed to just like sample some inputs ... and assume ... other things will behave similarly" [25:05]. Each step
  should be a "valid deductive step" [28:50]. Fitting a regression on samples is "a little bit cheating" [17:52].
  Declared approximations are allowed ("roughly zero. I'm ignoring some noise" [25:22]).
- **Compute.** Only "amount of computation" is named. There is no FLOP convention and no budget.

**The project's score.** Directive 8, amendment 2, in TASKS.md:
F = mean_m(e_m²) · max(C, 1k FPE) / mean_m(p_m(1−p_m)).
- e_m is the estimate error on model m.
- C is the estimator's cost in forward-pass equivalents (FPE = FLOPs/5,948, the cost of one sampled forward pass).

Plain sampling with C forward passes has F ≈ 1 by construction. **Strict bar: F ≤ 1.** **Lenient bar: F ≤ 10.** A
*candidate* must pass the lenient bar on the point estimate and have upper CI ≤ 10 on the sealed panel HP2. Two
caveats:
- **"Roughly match" does not appear in the talk.** The talk says "match". The literal reading supports the strict
  bar. The lenient factor of 10 is a project convention, taken from the 2.4–10x range of ARC's own M_4,3 estimators
  (E32). It is not ARC's threshold.
- The 1k-FPE cost floor is also a project convention. A principled floor would be about 80 FPE.

## 2. The method, step by step
Code: `m16/approaches/r1_cont/` on branch `claude/t218-tie-tail-labels` (owner T218). The entry point is
`estimator_c.est(model)`, which calls `cont_c.est(model, G=20)` and returns (estimate, FLOPs as executed).
**D** = deductive (exact, or a rigorous computation up to float64 rounding). **A** = declared approximation. No step
draws random model inputs.

1. **Order the inputs [D].** With no biases, correctness depends only on the input values. Let v1 > v2 > … > v10 be the
   sorted inputs. The joint density of the top 3 is 720·φ(v1)φ(v2)φ(v3)·Φ(v3)^7 (exact order statistics).
2. **R0/R1W closure [A].**
   - The top values are kept exact, and the remaining "low" inputs are replaced by their conditional mean
     μ(v) = −φ(v)/Φ(v) below the last kept value.
   - Closure *c* keeps the top 4, with v4 at the latest free time position, and puts 6 lows at μ(v4). This is the
     main approximation.
   - T318 measured the integration error of R1W V1 against a Monte Carlo of closure c: 0.72 pp RMS. The closure's
     own error is what remains after that.
3. **Transverse points [A in the scored code (1c742430 and c70e22cd); T322 makes them deterministic].**
   - For each order region (which time positions hold the top values; "blind" regions are separated from the rest),
     the walk is evaluated at fixed transverse points with one v4 node (K=1).
   - At 1c742430 some transverse points were placed by seeded Monte Carlo over model-free quantities. This moves
     estimates by ≤ .008 pp (T320 audit). T322 (T317d) replaces it.
4. **Exact walks along the g12 and g23 lines [D].**
   - Along each line (the gap v1−v2 or v2−v3, the other values fixed) the network is piecewise linear in the input.
   - The walk follows segments exactly, finding each ReLU flip by a ratio test, up to the cutoff tmax = 0.5.
   - It returns the set of intervals on which the model's output is correct.
5. **Continuity restarts [D] (T317).**
   - After a flip only the neurons downstream of the flipped time step change.
   - The walk therefore recomputes from step s, charged at 2(T−s)/T of a full pass, instead of from step 0.
6. **B2 prefix tree [D] (T317).** Walks that share an input prefix share their first steps of computation.
7. **Blind g12 assumed correct [A].** In blind regions the g12 line is not walked; it is taken as correct.
8. **Product of the line sets [A].** The correct set in the (g12, g23) plane is approximated by the product of the two
   one-dimensional correct sets. This drops their interaction.
9. **Merge touching pieces [D] (T317c).** Adjacent correct intervals are merged before the masses are integrated.
   Correct sets do not change; there are fewer Φ evaluations.
10. **Exact interval masses [D].**
    - For a correct interval [a, b] in g12: A12 = Σ [Φ(v2+b) − Φ(v2+a)].
    - In g23: A23 = Σ [Φ(v2−a)^8 − Φ(v2−b)^8]/8.
11. **Outer integral over v2 [A: quadrature].** Gauss-Legendre with G=20 nodes (T317b; earlier versions used 48) against
    the top-3 law. The quadrature error is not bounded rigorously.
12. **Model-free tables [D].** Φ and φ at fixed nodes do not depend on the weights. Under the convention they are not
    charged, but they must be listed with their sizes and computed once. At 1c742430 they are recomputed inside
    `est` and not listed (T320 gap; fixed by T322).

The estimate is the order-region-weighted sum of these masses.

**Accounting for the error.** On DEV2 the bias is +0.445 pp and the RMS 0.795 pp. The main declared approximations
are the closure (2), the product of line sets (8) and blind-g12-correct (7). Their individual shares on HP2 have not
been separated. I have measured only the integration share (0.72 pp, T318).

## 3. Cost convention and the executed count
**Convention (T196/T312).**
- Every arithmetic operation whose inputs depend on the weights is charged: multiply-add = 2 FLOPs, Φ/ndtr = 20.
- Model-free tables are free but must be listed.
- Walk steps are counted as executed, including restart charges.
- Sampling is charged 5,948 FLOPs per forward pass.

**The counts for T317c.**

| count | commit | DEV2 C (FPE) | HP2 C (FPE) |
|---|---|---|---|
| frozen and first scored | 1c742430 | 5,435 | 5,455 |
| every executed ndtr charged | 1c742430 | — | 5,713 |
| **official, as executed** | **c70e22cd** (scorer 7dd19ff5) | 5,488 | **5,489** |

c70e22cd only fixes the count. The estimates are unchanged.

**Independent recount (T320).**
- With the pre-registered root rule, my instrumented re-implementation (`../t317c_confirm/cont_indep.py`) counts
  5,740 FPE against T317c's 5,271 on the same 10 DEV2 models (+8.9%). This fails the 5% agreement band.
- With the root rule equivalent to T317c's, it counts +0.1%.
- The difference is a convention choice (how much of a restart the root step costs), not an error.

**Latitude.** I report C as **5.3–5.7k FPE**. Across that band, HP2 F runs from about 6.7 to 7.3 and its upper CI from
about 8.7 to 9.5. The lenient verdict does not depend on which count is used. A recount about 14% above the official one would bring the upper CI to 10 [inferred].

## 4. Scores
F and its 95% CI come from a bootstrap with one cluster per model (10,000 resamples) unless stated otherwise.

**DEV2: 80 development retrains** (seeds 0–39 × 2 kernels, data seed 21, open 1e8-sample truths). The estimates are
identical to T317b and to R1W V1 within 1.4e-6 pp.
- Bias +0.445 pp; RMS 0.795 pp.
- C is 5,435 FPE (1c742430) or 5,488 FPE (c70e22cd).
- Mean p(1−p) = 0.0725, so n* = 1,147. **DEV2 F = 4.78 [3.53, 6.08]** at C = 5,488 (as executed) and 4.74 [3.54,
  6.01] at 5,435 (cluster bootstrap, 10,000 resamples). Reproduce with `dev2_f.py`; the T317c estimates are
  `est_c` in `r1_cont/results_d.json`.
- DEV2 was used for every design choice, so this score is not an honest test.

**HP2: 100 sealed retrains** (hp2_s100..s199, fused trainer, data seed 41, 1e8-sample truths with max SE 3.9e-5).
Hashes: weights `039e3e69…`, manifest `1bf47b3f…`, truths `1a6a44f4…`. Bias +0.53 pp; RMS 0.99 pp.

| count | C | F [95% CI] |
|---|---|---|
| 1c742430, official | 5,455 | 6.72 [4.84, 8.73] |
| 1c742430, every ndtr charged | 5,713 | 7.04 [5.07, 9.14] |
| **c70e22cd, as executed (official)** | **5,489** | **6.76 [4.87, 8.79]** |

- The lenient bar passes: point 6.76, upper CI 8.79 < 10.
- The strict bar fails: the lower CI, 4.87, is above 1.
- n* = 812, the number of samples that would give the same RMS.

**HP3: 100 fresh retrains** (seeds 200–299, max SE 4.0e-5; scored once, T321).
- **Registered (T281, 67e57f00): INDEX H-L1, CONFIRMED by the DRI (scoped to retrains; see that row for scope and
  limits).** HP3 is a fresh panel never used before; the code was frozen before it was scored.
- As executed (c70e22cd): C 5,427, **F 5.08 [3.42, 7.04]**, RMS 0.805 pp, bias +0.415 pp.
- Sensitivity: 5.05 under the T281 count at 1c742430; 5.29 with every ndtr charged.
- The plain-MC control on the same panel scores about 1 (control runs 0.72–1.07).

**T317d (T322)**, which removes the three documentation gaps (§2 steps 3 and 12), is the intended ARC-ready version.
Its HP3 score will be added here when it is in.

**HP1: original model + 5 zoo seeds, clean and with noisy copies** (8 noisy copies per base model at each level; 1e8
truths; 6-cluster bootstrap, seed 281). As executed (c70e22cd):

| set | F [95% CI] | note |
|---|---|---|
| clean (6 models) | 3.78 [0.47, 8.33] | only 6 clusters, so the CI is wide |
| relative noise 0.005 | 39.3 [12.4, 63.8] | fails |
| relative noise 0.01 | 55.4 [25.6, 83.4] | fails |
| absolute noise 0.05 | 206.1 [174.1, 239.5] | fails: RMS 6.89 pp, bias +6.64 pp |

## 5. Comparison with sampling and with the inductive reference
**Sampling.** At equal cost (about 5.5k forward passes) plain Monte Carlo has an MSE 6.76 times smaller on HP2 (5.1
times on HP3). Put the other way: the estimator's error matches what about 812 samples give, and it
costs about 5,489 sample-equivalents. **It is about 6.7x worse than sampling, not better.** What it offers instead is
a deterministic estimate with no random model inputs, built from steps that can each be checked.

**v5q, the project's inductive reference** (T274a, `../probe_v5/`). This is fixed-node quasi-Monte Carlo: it evaluates
the model at a deterministic low-discrepancy set of inputs. In ARC's terms it is inductive, because it samples inputs
(deterministically) and assumes they are representative. On HP2:

| B (FPE) | 1k | 4k | 16k | 64k | 256k |
|---|---|---|---|---|---|
| F | 1.27 | **1.06** [0.81, 1.34] | 1.48 | 3.53 | 10.47 |
| RMS pp | 1.00 | 0.46 | 0.27 | 0.21 | 0.18 |

At 4k, v5q's HP1 F is 0.70 (clean), 1.35 (abs .05), 1.15 (rel .005) and 1.25 (rel .01). Its DEV2 F is 1.2. v5q
roughly matches sampling at ~4k FPE on every panel, noisy copies included. T317c is ~6x worse on retrains and 10–200x
worse on noisy copies.

## 6. Forking-paths log
Every variant run on DEV2 in the R1 lineage, from r1_walk/README.md:122–129 and the task READMEs:

| task | variants | DEV2 models |
|---|---|---|
| T297 | 7 ladders + 4 cheap variants | 10 |
| T300 | 4 frozen designs; 3 sharing rules | 80; 10 |
| T302 | tmax {.25, .5, 1, 2, 6}; tmax {.05, .5, 1} | 10; 80 |
| T304 | 4 closures + V1/V2/V3 | 80 |
| T306 | V1 / A / B / A+B | 80 |
| T307 | B5 oracle; stopped at 8.2% | 10 |
| T311 | killed by arithmetic, not built | — |
| T313 | cost only (10,971 FPE), estimates unchanged | 80 |
| T315 | capped c = 1/2/3, F 1269/1678/903 | 80 |
| T317 | continuity + B2, G=48 (G=16 tried and rejected) | 80 |
| T317b | G scan 20/24/32 (C 6,293/6,505/6,928) | 80 |
| T317c | cost only (merging, tables), estimates unchanged | 80 |

That is about 30 estimate variants. Since T306, the estimate itself has not changed: T313 and T317–T317c changed only
the cost. The other lineage (T319, smooth verdict) has no recorded result and no HP2 submission.

**Every sealed-panel submission:**

| entry | HP2 F [95% CI] |
|---|---|
| T300 r0_shared | 9.72 [7.02, 12.97] (8.89 under the T312 count) |
| T306 V1 | 16.35 [11.77, 21.24] (recounts 14.92 / 16.83 / 13.51) |
| T317 | 9.57 [6.89, 12.43]; official 9.71 [6.99, 12.61] |
| T317b | 7.75 [5.58, 10.07]; official 7.89 [5.68, 10.25] |
| T317c | 6.72 [4.84, 8.73]; as executed 6.76 [4.87, 8.79] |
| HP3: T317c | 5.08 [3.42, 7.04] (T281, H-L1) |

**What this means for HP2.**
- HP2 was scored three times on the same estimates (T317, T317b, T317c), each time with a lower cost. The T317c
  savings were written after T317b missed the upper-CI bar by 0.07, although the DRI had pre-registered them.
- The panel is therefore no longer fresh for cost engineering. Whether anything beyond the reported F and CI
  leaked from HP2 to the developers is not recorded here; I did not audit it.
- HP3 (scored once, on the frozen code) is the cleaner test. Its point estimate is lower than HP2's (5.08 vs 6.76).

## 7. Limitations, plainly
1. **Retrains only.** The method passes the lenient bar only on fresh seeds of the training recipe. ARC names weight
   noise as the other model distribution. On noisy copies it fails by 4–20x (HP1 F 39–206). The closure and the
   assumption that blind g12 is correct are tuned to trained structure, and noise breaks that structure.
2. **The strict bar fails.** No panel's CI comes near F ≤ 1 (best lower CI 3.42 on HP3, 4.87 on HP2).
3. **About 5–7x worse than sampling** at equal compute. A random-sampling estimator with the same budget is more
   accurate.
4. **"Roughly match" at best.** The lenient factor of 10 is our convention, not ARC's. ARC's words are "match".
5. **Not fully deductive in the error budget.** Steps 2, 7, 8 and 11 are declared approximations with no rigorous
   bound. The estimate has a systematic bias of +0.4–0.5 pp on every retrain panel, the same sign each time.
6. **The cost depends on the convention.** The FPE floor, the Φ = 20 FLOPs charge, uncharged tables and the
   restart-charge rule are project choices. Independent recounts differ by up to 9%.
7. **Heavy development on DEV2** (about 30 variants) and repeated HP2 scoring (§6).
8. **One model architecture.** Nothing here has been tested beyond M_16,10.

## 8. Reproduction
**Code.** Branch `claude/t218-tie-tail-labels`, folder `m16/approaches/r1_cont/`:

| file | sha256 |
|---|---|
| `estimator_c.py` | `b3c88771b66c6cfc5d35854fe6b2e5caf005d79b442666f5fefb73aba21337ce` |
| `cont.py` | `b71f203f5caec913193a9c6c045b76b35a43b63edcd7c27148a081ffcc084d92` |
| `cont_c.py` @ 1c742430 (first frozen) | `a0dc5f4aa0189dbe4b4fc7a6d24d3cdba3e244dbf612937f633c3064e3eaf839` |
| `cont_c.py` @ c70e22cd (official count) | `066b3f6355d8d4d4e0c40e4444961a031648c76bc085f9632309842e75130211` |

**DEV2.**
- Data: branch `claude/t192-gpu-cost-law`, `m16/approaches/gpu_cost_law/hp2/dev2/{weights.npz, truths.json,
  manifest.json}`. The weights sha starts `22d1c7c2…`.
- Run `python run_c.py` in `r1_cont/`. It writes `results_c.json` and loads the weights through `r0_integral/r0.py:dev2()`,
  which checks their sha.

**HP2** (sealed; held by the scorer on branch `claude/hp1-scorer`, `m16/approaches/hp1_scorer/`):
```
PANEL=hp2 PYTHONPATH=shim python3 score.py <name> <snapshot>/m16/approaches/r1_cont/estimator_c.py est deductive 0
CI_CODE=216 python3 cost_t317.py      # official recount
bash run_hp3_t317c.sh                 # HP3
```

**Independent checks.**
- T312: `../r1w_repro/r1w_indep.py`, an R1W V1 re-implementation. It agrees within .012 pp on all 80 DEV2 models.
- T320: `../t317c_confirm/cont_indep.py` and `compare.py`. They agree within .0077 pp on 10 DEV2 models; the results
  are in `results/compare_v2.json`.

## Compute accounting
Writing this document took no new compute. All numbers come from the cited runs.

## Claims
- [measured] On 100 sealed retrains (HP2), the as-executed T317c estimator has F = 6.76 [4.87, 8.79] at C = 5,489 FPE.
  It passes the project's lenient bar and fails the strict bar.
- [measured; DRI-confirmed as H-L1, scoped] On 100 fresh retrains (HP3), F = 5.08 [3.42, 7.04].
- [measured] On noisy copies (HP1), F = 39–206. It fails.
- [measured] Two independent re-implementations (T312, T320) reproduce the estimates within .012 pp and the cost
  within 0.1–9% depending on the root rule.
- [measured] DEV2 F = 4.78 [3.53, 6.08] at 5,488 FPE (development panel, not a test).
- [inferred] The lenient verdict holds across the 5.3–5.7k FPE cost latitude.

## Alternatives not ruled out
- **Open item:** the error budget per declared approximation (closure, product set, blind g12, quadrature) has not
  been measured on any sealed panel.
- The HP2 pass could depend on cost engineering tuned to HP2's pass/fail (§6). HP3, a fresh panel scored once on
  frozen code (F 5.08), argues against this.
- The +0.4–0.5 pp bias could be a property of this training recipe. Retrains from a different recipe were not tested.
- How the error splits among the declared approximations (closure, product set, blind g12) is unmeasured on HP2.

## Failure modes
- Noisy or perturbed weights (HP1).
- Any model whose low inputs matter beyond their conditional mean (closure c).
- Cost conventions stricter than ours, such as charging the tables or using a lower FPE floor.
