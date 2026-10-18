# T323: progress note — how far a sampling-free surrogate gets on retrains of M_16,10, and why it stops there

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

**Status: final draft v3 (2026-10-18), revised after the red-team review (`review/candidate_redteam/README.md` on
`ccr-6a61f25c-wn80ii`).** This is a progress note, not a candidate solution: the estimator does not match sampling.
Author: the T241 worker. **The same author also did both re-implementation checks (T312, T320).** "Confirmed" in the
project INDEX (H-L1) is an internal label; read it as "verified internally: a second implementation and an
independent scorer". A different agent has since audited the method and measured its error budget (T324,
`claude/t292-oracle-floor`, `m16/approaches/budget_t317d/`, pre-registered e85fff85). §2 uses its results.

## Headline

> On 100 fresh retrains of AlgZoo's M_16,10 training recipe (sealed panel HP3, frozen code, scored once), a
> sampling-free estimator estimates each model's accuracy with RMS error 0.80 pp (bias +0.41 pp) at a counted cost of
> 5.4k forward-pass equivalents (5.4–7.0k under alternative FLOP conventions, §3). It walks the ReLU gates exactly
> along the two gap directions of each position triple, under six declared approximations.
> **It is sampling-free at runtime, but not deductive in Hilton's strict sense.** The approximation scheme (the
> closure, the latest-position rule for v4, the cut at gap 0.5, the single v4 node and the blind-g12 rule) was
> selected and validated by Monte Carlo on 80 separate development retrains. Its extrapolations are declared and the
> same for every model; nothing is fitted per model.
>
> **Plain sampling at the same compute would have about 5x lower MSE: F = 5.07, 95% CI 3.41–7.03.** The estimator is
> worth n* ≈ 1,070 samples. On a second sealed panel (HP2), used for earlier submissions of the same lineage, F is 6.75
> [4.86, 8.78]. **The estimator does not match sampling, and it fails on weight-noised copies (bias up to +6.6 pp).**
> The version to cite, T317e, makes no random draws at all. It is bitwise identical to the scored T317d, which still
> ran two unused seeded Monte Carlos at import.

The positive result is the information content. A constant at the development mean (0.920) scores F 18.9 [14.1, 24.2] on HP3 at the 1k-FPE floor and is
worth n* = 53 samples. This estimator is worth 1,069 on HP3, 20x more. On DEV2 its correlation
with the truth is 0.98 (R² 0.94).

## 1. The problem and ARC's bar
**Model.** M_16,10 is AlgZoo's 1-layer ReLU RNN with hidden size 16, sequence length 10 and no biases. It outputs the
position of the second-largest of 10 i.i.d. N(0,1) inputs. Its accuracy is 0.953 (`m16/truth.txt`: 0.95318).

**ARC's bar** comes from [`lit/web_arc/hilton_talk_2026-09_notes.md`](../../../lit/web_arc/hilton_talk_2026-09_notes.md),
whose quotes are approximate and taken from an auto-generated transcript.
- **The goal.** "produce a mechanistic estimate of the accuracy ... **match** random sampling in terms of the like
  amount of computation relative to like mean squared error" [33:12]. Hilton calls it "as yet completely open" [33:00].
- **Which models.** "taking a random seed or maybe adding noise to the weights" [33:22–33:31], with MSE averaged over
  "randomness over the weights" [35:29]. No noise level or seed list is given.
- **What counts as mechanistic.** "deductive rather than inductive" [24:57]; "not allowed to just like sample some
  inputs ... and assume ... other things will behave similarly" [25:05]; regression on samples is "a little bit
  cheating" [17:52]. Approximations are acceptable ("roughly zero. I'm ignoring some noise" [25:22]).
- **Compute.** Only "amount of computation" is named.

**What the bar is not.** The talk says "match". "Roughly match" and the factor of 10 are not ARC's. They are a project
convention taken from the 2.4–10x range of ARC's own M_4,3 estimators under one op count (E32), and ARC's
"roughly match" text there was about ~2x RMSE. **By ARC's word, this estimator fails, by about 5x.**

**The project's figure of merit** (TASKS Directive 8, amendment 2):
- F = mean(e²) · max(C, 1k FPE) / mean(p(1−p)), with FPE = FLOPs/5,948 (one sampled forward pass).
- Sampling scores F ≈ 1. The break-even sample count is n* = mean p(1−p)/RMS².

**The scoring rule moved three times, each time towards where the method passes:**

| rule | what it scores | why it was changed | T317c/d |
|---|---|---|---|
| Directive 7 | seeds and noisy panels (abs σ .05, rel .005/.01) | — (ARC's model distribution) | fails (noise) |
| Amendment 1 | one fixed budget of 16k/64k/256k FPE, all six groups | own-cost scoring let cheap estimators pass trivially (a constant passed) | fails (5.4k < B/2; noise) |
| Strict-wording fix | point F ≤ 1; "significant" if the upper CI ≤ 1 | with 6 clusters even sampling failed "upper CI ≤ 1" | fails |
| Amendment 2 | own cost with a 1k-FPE floor; retrains (HP2) primary, noise as robustness | the talk names no budget, so a frozen deductive estimator is one (cost, MSE) point | lenient pass |

ARC's phrasing supports own-cost scoring. The 1k floor is our choice; the principled floor is ~80 FPE.

**The denominator.** F uses the panel's p(1−p). The panels' mean accuracy is ~0.92 (p(1−p) ≈ 0.07), while M_16,10 is
0.953 (0.045). At the same MSE, a target-like denominator gives F ≈ 7.7 (DEV2), 7.9 (HP3) and 10.5 (HP2). But the
error shrinks with accuracy:
- **By accuracy on the sealed panels** (scorer `claude/hp1-scorer` 1515eb10). Each subset uses its own p(1−p):

| panel (T317c) | p > .95 | p < .90 | slope (pp per pp) |
|---|---|---|---|
| HP2 | 1.99 [1.13, 3.01] (n=23; bias +0.14, RMS 0.37 pp) | 9.41 [6.31, 12.89] (n=38) | −0.079 |
| HP3 | 4.17 [1.31, 7.67] (n=22; bias +0.18, RMS 0.53 pp) | 6.90 [2.88, 12.13] (n=20) | −0.066 |

  T317d matches within 0.03. The bias sits in the low-accuracy models (+0.8 to +0.9 pp below 0.90).
- **The target itself** (public weights and truth): T317c estimates 0.954049 against truth 0.953179, an error of
  +0.087 pp. T317d's error is +0.084 pp. On this one model n* ≈ 59–63k against C ≈ 5.8k, about 10x
  better than sampling. **This is one model, and anecdotal.** ARC scores over a distribution of models, and the panel
  F values carry the claim.

## 2. The method, step by step
The code is `m16/approaches/r1_cont/` on branch `claude/t218-tie-tail-labels`. The entry point `est(model)` returns
(estimate, FLOPs as executed).

**Deductive status (audit by T324, a different agent).** In its words: "sampling-free at runtime; the approximation
scheme was selected and validated by Monte Carlo on development retrains". "Deductive" in Hilton's stricter sense it
is not.
- No runtime step feeds random inputs to the model. Nothing is fitted per model.
- Exact verdicts along two lines per triple and region are extrapolated to the whole 3-D region (every v2, every
  value of the other gap, gaps > 0.5, blind g12). These extrapolations are declared and the same for every model.
- These were chosen by Monte Carlo on DEV2: closure c, the latest-position rule, TMAX 0.5, K = 1 and the blind rule.
- G = 20 is only a numerical setting.
- No bound for any approximation is derived from the weights. T319's attempt at one failed.
- The import-time Monte Carlo applies to T317d, not to T317e.

Labels: **D** = exact up to float64. **A** = declared approximation, followed by how it was chosen.

1. **Order statistics [D].** With no biases only the sorted values v1 > v2 > … matter. The top-3 density is
   720·φ(v1)φ(v2)φ(v3)·Φ(v3)^7.
2. **Closure c [A; chosen among 4 closures by 200k paired skeletons per DEV2 model, T304].** The top 4 values are kept
   exact, with v4 at the latest free position. The 6 low inputs are set to their conditional mean μ(v4) = −φ/Φ.
3. **One transverse point per region [A].** Each order region (nonblind v2 > 0, blind v2 ≤ 0) is evaluated at one
   transverse point (v2*, g12*, g23*), the exact conditional means of the input law.
   **3b. Single v4 node [A; K = 1, chosen by cost].** v4 sits at the one Gauss node E[v4 | v3] instead of its law.
   This shifts the bias by ±0.31 pp (T324).
   - In T317c the slice positions came from a seeded Monte Carlo of the input law: `r1w2._regions` draws 1e6×10
     normals with `default_rng(0)` at import. The quantities are model-free.
   - T317d replaces this with deterministic quadrature. Its estimates move by ≤ .0115 pp. T317e is the same
     arithmetic with no draws on any path.
4. **Exact walks along the g12 and g23 lines [D], cut at tmax = 0.5 [A; chosen by a scan, T302].** Along each gap
   line the network is piecewise linear. The walk finds every ReLU flip and returns the intervals where the output
   is correct.
5. **Continuity restarts and the B2 prefix tree [D] (T317).** After a flip at step s, the walk recomputes only from
   step s, charged 2(T−s)/T of a pass. Walks with a shared prefix share their steps.
6. **Blind g12 assumed correct [A; dropped by cost, its effect measured by MC, T304 V3].**
7. **Product of the two 1-D correct sets [A].** This ignores the interaction between the two lines.
8. **Merging touching pieces [D] (T317c).** Correct sets are unchanged; fewer Φ evaluations.
9. **Exact interval masses [D].** A12 = Σ[Φ(v2+b) − Φ(v2+a)] and A23 = Σ[Φ(v2−a)^8 − Φ(v2−b)^8]/8.
10. **Outer Gauss-Legendre over v2, G = 20 [A; G chosen by a convergence check on DEV2, T317/T317b].**

**Error budget (T324: 80 DEV2 models, 2e5 paired inputs each, open 1e8 truths).** Its pointwise form matches
`cont_d.est` within MC error: bias +.439 against +.444, RMS .796 against .794. Switching off one approximation at a
time from V1 = T317d, with MSE .633 pp²:

| switched off | change in bias (pp) | change in MSE |
|---|---|---|
| **product of the 1-D line sets** | **−.57** | **−40%** |
| tmax cut | −.18 | −18% |
| blind g12 | −.07 | −9% |
| μ linearisation of the lows | +.02 | +2% |
| one transverse point (alone) | +.27 | +25% |
| single v4 node | ±.31 (shifts the bias, moves the MSE little) | ~2% |

- **The walk approximations carry 66% of the MSE.** Closure c itself carries 34%, with a floor of RMS .46 pp and
  bias −.15 pp.
- Together the walk approximations carry +.59 pp of bias, more than V1's whole +.44. Their bias and the closure's
  −.15 partly cancel, and that cancellation is fragile.
- The product, the transverse point and the v4 node interact. Their individual sizes depend on the order of
  removal: together they carry +.03 pp of bias and ~42% of the MSE.
- The product overstates accuracy: a point correct on both central slices need not be correct jointly.

The noisy-copy failure (§4) suggests the approximations encode regularities of the training family. The nearest
precedent is ARC's M_4,3 estimator A, with its tuned q = 0.2 (E32).

**The cell cap.** `walk_cont(max_cells=400)` would truncate silently, but it is never hit on DEV2: at most 22 cells per
line, and the estimates are identical with a cap of 5,000 (T324).

## 3. Cost convention and its range
**Convention (TASKS, from T312):**
- Charge every operation on weight-dependent inputs: multiply-add = 2 FLOPs, Φ = 20.
- Model-free functions of the input law are free if they are precomputed once and listed.
- The sampling side is 5,948 FLOPs per pass (forward + ReLU + argmax + label; random-number generation excluded), as
  in ARC's own op-count rules.

**Counts:**

| count | HP2 C | HP3 C |
|---|---|---|
| T317c as executed (c70e22cd), official | 5,489 | 5,427 |
| T317d, official | — | 5,431 |
| stacking: × 1.09 (T320's root rule) × 1.04 (every Φ), and for T317c + 807 FPE of per-call tables | | |
| T317c stacked | 7,030 | 6,960 |
| T317d stacked (tables built at import, so no per-call term) | 6,225 | 6,156 |

T320's count used the root rule it pre-registered; with the rule equivalent to T317c's, the two counts agree within
0.1%.

**Effect of the stacked count** (scored by the scorer at 1515eb10):

| panel | T317c | T317d |
|---|---|---|
| HP3 | 6.51 [4.38, 9.02] | 5.75 [3.86, 7.97] |
| HP2 | 8.66 [6.23, 11.25], **upper CI over 10** | 7.65 [5.51, 9.95] (just under 10) |

The 1.04 factor is conservative for c70e22cd, which already masks its Φ calls.

**T317d's precomputation** uses a 2 × 800k-point trapezoid grid. That is ~1e8 FLOPs ≈ 20k FPE, once per run. It is
model-free and uncharged, but if it were charged per model it would dwarf C. A quadrature cheap enough to charge in
full (< 100 FPE) would remove the question.

**Wall clock is the other reading of "amount of computation."** The Python walk takes ~0.75 s per model, against
~10 ms for 5.4k vectorised forward passes (~100x). FLOP counts treat the data-dependent control flow as free.

## 4. Scores
95% CIs come from a cluster bootstrap with one cluster per model.

| panel | models | role | C (FPE) | bias / RMS (pp) | F [95% CI] | n* |
|---|---|---|---|---|---|---|
| DEV2 (T317c) | 80 retrains, seeds 0–39 × 2 kernels, data seed 21, open truths | development; every choice tuned here | 5,488 | +0.445 / 0.795 | 4.78 [3.53, 6.08] (`dev2_f.py`) | 1,147 |
| HP2 (T317c) | 100 sealed retrains, seeds 100–199, data seed 41 | selection panel (§6) | 5,489 | +0.53 / 0.99 | 6.76 [4.87, 8.79] | 812 |
| HP2 (T317d) | same | same | — | — | 6.75 [4.86, 8.78] | — |
| HP3 (T317c) | 100 fresh retrains, seeds 200–299 | scored once | 5,427 | +0.415 / 0.805 | 5.08 [3.42, 7.04] | 1,069 |
| **HP3 (T317d)** | same | scored once | 5,431 | +0.414 / 0.804 | **5.07 [3.41, 7.03]** | ≈1,071 |

- HP3 sensitivity counts: 5.05 and 5.29. The plain-MC control on HP3 scores about 1.
- Across the three panels F is 4.8 / 6.8 / 5.1, a spread of about ±30%. Pooled over the 200 sealed models, F ≈ 5.9.
- T317d's pre-registered DEV2 gate was .01 pp per model. One model moved .0115 pp, and the DRI accepted it.
- Truths are 1e8-sample Monte Carlo per model, with max SE ≤ 5e-5.

**HP1, the robustness panel** (the original model + 5 zoo seeds; 8 noisy copies of each per level; 6-cluster
bootstrap). T317c as executed:

| set | F [95% CI] |
|---|---|
| clean (6) | 3.78 [0.47, 8.33] |
| rel σ .005 | 39.3 [12.4, 63.8] |
| rel σ .01 | 55.4 [25.6, 83.4] |
| abs σ .05 | 206.1 [174.1, 239.5] (bias +6.64 pp) |

## 5. Alongside: the inductive reference v5q and the baselines
v5q (T274a, `../probe_v5/`) is fixed-node quasi-Monte Carlo: deterministic inputs, assumed representative. By ARC's
definition it is **inductive**. It is shown so the contrast is explicit.

| B (FPE) | 1k | 4k | 16k | 64k | 256k |
|---|---|---|---|---|---|
| v5q HP2 F | 1.27 | **1.06** [0.81, 1.34] | 1.48 | 3.53 | 10.47 |

- At 4k FPE, v5q's HP1 F is 0.70 / 1.35 / 1.15 / 1.25 (clean / abs .05 / rel .005 / rel .01). Its DEV2 F is 1.2.
- v5q roughly matches sampling on every panel, noisy copies included. The sampling-free estimator is ~5x worse on
  retrains and 10–200x worse on noisy copies.
- **HP3 baselines at the 1k floor** (scorer 1515eb10):

| estimator | class | F [95% CI] | n* |
|---|---|---|---|
| constant 0.920 | none | 18.9 [14.1, 24.2] | 53 |
| v5q, 1k | inductive | 0.99 [0.71, 1.29] | 1,018 |
| v5q, 4k | inductive | 0.87 [0.66, 1.09] | 4,644 |
| T317c | sampling-free | 5.08 [3.42, 7.04] | 1,069 |

  At about 1k FPE, v5q reaches the information T317c needs 5.4k FPE to reach.

## 6. Forking paths and selection
**DEV2 variants** (r1_walk/README.md:122–129 and the task READMEs), about 30 estimate variants in all:

| task | variants (DEV2 models) |
|---|---|
| T297 | 7 ladders + 4 cheap variants (10) |
| T300 | 4 designs (80); 3 sharing rules (10) |
| T302 | tmax {.25, .5, 1, 2, 6} (10); {.05, .5, 1} (80) |
| T304 | 4 closures + V1/V2/V3 (80) |
| T306 | V1 / A / B / A+B (80) |
| T307 | B5 oracle, stopped at 8.2% (10) |
| T311 | not built (killed by arithmetic) |
| T313 | cost only, 10,971 FPE (80) |
| T315 | capped c = 1/2/3: F 1269 / 1678 / 903 (80) |
| T317 | G=48; G=16 rejected (80) |
| T317b | G 20 / 24 / 32 (80) |
| T317c, T317d | cost and documentation only (80) |

Since T306 the estimates have not changed beyond .0115 pp.

**Sealed submissions:**

| entry | HP2 F [95% CI] | HP3 F |
|---|---|---|
| T300 r0_shared | 9.72 [7.02, 12.97] | — |
| T306 V1 | 16.35 [11.77, 21.24] | — |
| T317 | 9.71 [6.99, 12.61] | — |
| T317b | 7.89 [5.68, 10.25] (first 7.75 [5.58, 10.07]) | — |
| T317c | 6.76 [4.87, 8.79] | 5.08 [3.42, 7.04] |
| T317d | 6.75 [4.86, 8.78] | 5.07 [3.41, 7.03] |

**Selected on HP2 after five submissions of the lineage; confirmed once on a fresh panel.**
- T317, T317b and T317c carry identical estimates; only the cost was cut.
- T317b's HP2 score (upper CI 10.07) was posted at 10:46 UTC. T317c's pre-registration and commit followed at 10:50.
  The savings were conceived after the near miss, and the stopping rule ("upper CI ≤ 10") was conditioned on HP2.
- This does not bias F: the estimates are unchanged and the cost is real. It does make "candidate on HP2" a
  constructed threshold.
- HP3 is clean for F, given the selection. Residual caveats: it has the same recipe, data seed (41) and creator as
  HP2, and it was commissioned because HP2 passed.
- The chain from T317 to HP3 ran in 41 minutes on one day.

## 7. Limitations
1. **It does not match sampling.** It is about 5x worse at equal counted compute (7–8x with a target-like
   denominator, more in wall clock).
2. **Retrains only.** It fails on weight noise, which is half of ARC's stated model distribution.
3. **It is not deductive in Hilton's strict sense** (T324 audit, §2). It is sampling-free at runtime, but six
   approximations were chosen and validated by Monte Carlo on DEV2. Hilton's "assume other things will behave
   similarly" applies to this.
4. **The cost depends on the convention** (5.4–7.0k). At the top of that range, T317c's HP2 upper CI crosses 10
   (11.25). T317d's stays at 9.95.
5. **The scoring rule moved to where the method passes** (§1), and the method was selected on HP2 (§6).
6. **Verification is limited.** The author of this note also did both re-implementation checks (T312, T320). A
   different agent did the audit and the error budget (T324). Everything is internal to the project, and nothing has
   been tested beyond M_16,10.

## 8. Reproduction
The code is on branch `claude/t218-tie-tail-labels`, folder `m16/approaches/r1_cont/`. I recomputed the sha256s with
`git show | sha256sum`; they match the scorer's README at 29bef882.

| version | commit | entry point | sha256 |
|---|---|---|---|
| **T317e (cite this)** | 01906d33 (INDEX 23782c82) | `estimator_e.py` → `cont_e.est` | estimator_e `a9cb5ef610bbcb52ef6037097716fe4b8ff7f6aaa005133cee492094d1c6c0f8`; cont_e `50dd1c9b85512e23f037171d19de03527f8f5989758cfb74d93b441de48fda2b` |
| T317d (scored) | 5b4d4806 | `estimator_d.py` | estimator_d `f4bd3d87e0e7218ef270b54f350b9aa7a86306fa2bff0f8de08eca9709b91057`; cont_d `306061a685406111ffefa66751198f4b82f6e515905b4ae28381608438f59f64` |
| T317c | 1c742430 / c70e22cd | `estimator_c.py` (`b3c88771…37ce`) | cont_c `a0dc5f4a…f839` / `066b3f63…0211`; cont `b71f203f…4d92` |

T317e imports only numpy, `scipy.special.ndtr` and itertools, and its `est` runs with numpy's random generators
patched to raise. Its estimates and FLOPs are bitwise identical to T317d's on DEV2, and the scorer is asserting the
same on HP2/HP3.

**DEV2:**
- Data: `claude/t192-gpu-cost-law`, `m16/approaches/gpu_cost_law/hp2/dev2/`; weights sha `22d1c7c2…`.
- `python run_c.py` checks the cost.
- `python dev2_f.py results_d.json truths.json` (this folder) gives the DEV2 F.

**Sealed panels** (scorer branch `claude/hp1-scorer`, `m16/approaches/hp1_scorer/`):
```
PANEL=hp2 PYTHONPATH=shim python3 score.py <name> <snapshot>/m16/approaches/r1_cont/estimator_c.py est deductive 0
CI_CODE=216 python3 cost_t317.py      # official recount
bash run_hp3_t317c.sh                 # HP3
```

**Internal checks:**
- T312 (`../r1w_repro/r1w_indep.py`) reproduces R1W V1 within .012 pp on 80 DEV2 models.
- T320 (`../t317c_confirm/`) reproduces T317c within .0077 pp on 10 models; its count is +8.9% (root rule) or +0.1%.

## Compute accounting
None new, apart from `dev2_f.py`: arithmetic on stored estimates, under a second.

## Claims
All the numbers in §2–§5 are [measured], each with its source. None is [inferred] except the wall-clock ratio (§3).

## Alternatives not ruled out
- The +0.4 pp bias may be specific to this recipe, or to data seed 41 on HP2/HP3.
- The small error on the target (+0.087 pp) may be luck on one model. Its accuracy range is where the panels' errors
  are also small (p > .95: F 2–4), which is consistent with the target being easier, but this does not establish it.

## Failure modes
- Weight noise.
- Models whose low inputs matter beyond their mean.
- Stricter cost conventions: charging tables or precomputation, a lower floor, or wall clock.
