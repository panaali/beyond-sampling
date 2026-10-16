# Project index: clean-neurons

**DRI:** the main Claude session on branch `ccr-6a61f25c-wn80ii` (with the user).
This file is the single source of truth: what is being asked, who is doing what, where outputs live,
and **how well each claim is actually supported**. Update it whenever an experiment starts, ends or
changes status. Rules for contributors are in `CLAUDE.md`.

## 1. Questions

| id | question | status |
|---|---|---|
| Q1 | Is Cammarata's "Neural Networks Are Already Clean" right that most MLP neurons have one main job? | Answered at small scale (Qwen3-0.6B). See `RESULTS.md` |
| Q2 | Hilton/ARC: does per-neuron legibility add up to understanding? (AlgZoo models) | Answered for the 432-param model under one protocol. See E4 |
| Q3 | How does the 432-param AlgZoo RNN (M_16,10) compute the 2nd argmax? | Mostly answered. See `m16/FINDINGS.md` |
| Q4 | ARC challenge: a *mechanistic* (deductive) estimate of M_16,10's accuracy matching sampling in MSE vs compute, on noisy-weight copies | **Open.** E10 onward |

**Success criteria for Q4 (re-centred on ARC's stated intent: "a scalable, algorithmic approach to producing
mechanistic estimates"; hand analysis is "a warm-up"):**
1. **Deductive**: no random model inputs. Methods that sample inputs, however cleverly (stratified,
   quadrature with model evaluations at random-shift points, control variates), are **baselines to beat**, not solutions.
2. **General**: the same code runs unchanged, with no tables or allocations fitted per model, on the example model,
   `zoo_2nd_argmax(16,10, seed=0..4)` and noisy copies; ideally also M_4,3 as a sanity check.
3. **Competitive**: MSE at most that of sampling at equal compute, with CIs, on that model set.

## 2. Repo map

| path | what |
|---|---|
| `PREREGISTRATION.md` | Hypotheses and thresholds committed before data (Qwen study; AlgZoo E4 addenda) |
| `RESULTS.md` | Write-up of the Qwen study and the AlgZoo E4 results |
| `collect.py`, `build_items.py`, `analyze.py`, `population.py` | Qwen3-0.6B pipeline (activations -> blinded packets -> scoring) |
| `study/` | Qwen agent packets (`explain/`, `judge/`), agent outputs (`*_out/`), answer keys (`keys/`) |
| `algzoo/` | E4: label scorer (`score.py`), investigator label files, scratch dirs `work*/` |
| `m16/` | Deep analysis of M_16,10: `FINDINGS.md` (narrative), shared code `common.py`, `exact.py`, `modes.py`, ... |
| `m16/arc_eval.py` | **Shared ARC-protocol harness**: noisy copies, ground truth, estimator interface |
| `m16/leaderboard.py` | **DRI-run leaderboard**: 13 fixed models (original + 6 at sigma 0.0025 + 6 at 0.005), 4M-sample truths (s.e. ~0.011 pp), budgets 1k-256k, cost in forward-pass equivalents. Approaches expose `estimator.py`. Results in `results/leaderboard.json`. **Only leaderboard numbers count as verified.** |
| `m16/approaches/<name>/` | One folder per estimator approach (see registry) |
| `lit/` | Literature scans by era and by field (unverified citations are flagged inside) |
| `results/` | Machine-readable results (json/npy) |
| `data/` (git-ignored) | Qwen activations (2.4 GB); regenerate with `collect.py` |

## 3. Experiment registry

Status: `done` / `running` / `stopped` / `planned`. Verification: **V** = re-checked independently (another
script, agent or seed), **S** = single run by its author, **C** = claimed in an agent report, not yet checked.

| id | question | owner | status | where | headline result | verif. |
|---|---|---|---|---|---|---|
| E1 | Q1: H1-H3, blinded explainers + judges, random-direction control | main + 20 agents | done | `RESULTS.md`, `results/summary.json` | H1 61% A (borderline), H2 fail (+25pp < 30), H3 pass. Pre-registered verdict "refuted at small scale" | S |
| E2 | Q1: population stats, 896 neurons vs 112 random dirs | main | done | `population.py`, `results/population.json` | kurtosis 100 vs 0.8; mass >=20% of max 6% vs 38% | S |
| E3 | Q2: positive control M_4,3 | investigator agent | done | `algzoo/labels_4_3.py`, `results/algzoo_4_3.json` | 4/4 legible, 100% of gap recovered | V (held-out scorer) |
| E4 | Q2: M_16,10 legibility vs mechanism | investigator agent | done | `algzoo/labels.py`, `results/algzoo_16_10.json` | 14/16 legible, rebuild 54% (needs >=86.8%) | V (held-out scorer) |
| E5 | Q2: M_32,10 | investigator agent | stopped by DRI | `algzoo/labels_32_10.py` (draft) | unscored | - |
| E6 | Q3: mechanism of M_16,10 | main | done | `m16/FINDINGS.md` sections 1-4, 7 | see ledger H-M1..M6 | mixed |
| E7 | Q4: probe estimators v2-v4 | main | done | `m16/estimate.py`, FINDINGS section 5 | bias +0.74 -> +0.59 -> +0.25 pp on original; ~10-50x worse RMSE than sampling on noisy copies | S |
| E8 | Q4: exact line integration, RQMC | main | done | `m16/lineint.py`, `m16/qmc_test.py` | unbiased but ~17x worse; RQMC no gain | S (lineint validated vs brute force) |
| E9 | Q4: mechanism-informed stratified sampling | main | done | `m16/stratified.py`, `results/stratified_vs_sampling.json` | 1.2-2.8x MSE gain (model-eval accounting only) | S, see H-E3 |
| E10 | Q4: exact rank-ordering computation (deductive) | agent | done | `m16/approaches/rank_exact/` | **failed**: 8-13x worse RMSE; bias -1.0 to -1.2 pp on the original does not shrink with quadrature; on noisy copies the bias flips sign (-1.0 to +2.4 pp). Ruled out quadrature error and the lower-rank arrangement as causes | C |
| E11 | Q4: Laplace / boundary-layer integrals | agent | running | `m16/approaches/boundary/` | - | - |
| E12 | Q4: control variate (structure + randomness) | agent | running | `m16/approaches/control_variate/` | - | - |
| E13 | Q4: distribution propagation (deductive) | agent | done | `m16/approaches/propagation/` | **failed**: bias -10 to -69 pp at 2k-633k evals, insensitive to weights. Ceiling test: even *exact* cumulants of the final margin in 10k cells give -1.4 pp (Gaussian) / -0.35 pp (4 cumulants) vs a ~0.1 pp bar. Possible bug in gsf2's final step (~20 pp) flagged by its author | C (ceiling numbers are the informative part) |
| E14 | Q4: error-cone enumeration + model-free cone integration (deductive) | agent | done | `m16/approaches/cone_enum/` | **failed decisively**: 56-91x worse RMSE than sampling. >=444k error cones (82% seen once), >=2.4M gate prefixes by step 10; LP-pruned path search does not help; cone integrator reads low on thin cones (median 0.34x). 2000 cones cost 1.27M evals for 4% of the error mass. Ran a broad `pkill -f eval_noisy.py` before rule 9 existed; another agent's run may have died | C (independently consistent with H-M7) |
| E15 | Q4: tube / boundary slicing (deterministic quadrature) | agent | done | `m16/approaches/tube/` | **claims beats sampling**: MSE gain ~3x (sigma 0.0025), 1.6-2.2x (sigma 0.005), loses at 1k on sigma 0.005 (0.57x). Linear tube formula itself FAILS (overestimates 3.4x); the gain comes from exact cell-walking along tie segments plus exact radial integration. Caveats: importance tables cost ~2.8e5 evals on the original model (uncharged advice); fixed quadrature nodes shared across models; gain may be mostly tie-geometry stratification (comparable to E9). Classification: deterministic mechanism-guided quadrature, not deductive in ARC's strict sense | C (**leaderboard head-to-head vs sampling and stratified pending**) |
| E16 | Q4: Array-RQMC | agent | done | `m16/approaches/array_rqmc/` | 1.0-1.6x on original; gain mostly from exact last step; noisy runs NOT done | C |
| E17 | Q4: FORM/SORM reliability | agent | running | `m16/approaches/reliability/` | - | - |
| E18 | Q4: tie-direction line sampling | agent | running | `m16/approaches/tie_lines/` | - | - |
| E19 | Q4: certified branch-and-bound | agent | running | `m16/approaches/branch_bound/` | - | - |
| E20 | Q4: Rhee-Glynn debiased series / MLMC | agent | running | `m16/approaches/debiased_series/` | - | - |
| E21 | Q4: signal-detection observer model (deductive) | agent | done | `m16/approaches/observer/` | ~10x worse RMSE than sampling at equal compute (0.58-0.60 pp vs 0.05-0.07 at ~215k evals); original bias -0.65 pp, partly from cancelling errors. Useful: one linear map per configuration overstates errors ~5x; restricting other inputs to gate-stable intervals fixes most of it; near-tie flip rates predicted within ~15% | C (leaderboard re-run pending) |
| E22 | Q4: great-circle integration | agent | running | `m16/approaches/great_circle/` | - | - |
| R1 | Critical review of all outputs | reviewer agent | done | `review/R1/` (scripts) | 10 issues; corrections applied to RESULTS.md, FINDINGS.md and this ledger (H-Q2/3/4, H-A1, H-M3/4/5/6, H-E3/4) | V (scripts re-run data) |
| L1-L8 | Literature scans (4 eras, 4 fields) | agents | done | `lit/*.md` | converge on: enumerate error cones + Genz/Botev masses; tie-aligned preintegration; control variates | C (many citations unverified) |

## 4. Hypothesis ledger

Status: **confirmed** (independent check passed), **supported** (consistent evidence, no independent check),
**unverified**, **refuted**, **disputed**. Every row lists the alternatives **not yet ruled out**.
Nothing is "confirmed" on its author's word alone.

### Qwen study (Q1)
| id | claim | status | evidence | alternatives / weaknesses not ruled out |
|---|---|---|---|---|
| H-Q1 | 61% of Qwen3-0.6B neuron poles have "one main job" | supported (borderline) | E1, 84 poles, CI 50-71% | one explainer per item; explainer leniency; same model family as essay's tools; "one job" is a subjective category |
| H-Q2 | The "% one job" metric barely separates neurons from random directions (36% of random dirs rated A) | **disputed** (R1: random dirs are neuron-dominated, median top-1 share 29%, A-rated dirs most dominated; H2 underpowered, Fisher p=0.028 for a difference) | E1, n=28 random dirs (wide CI 18-54%) | **random directions may be dominated by a few massive-activation neurons, so they may effectively *be* neurons** (not checked); small n; random dirs and neurons were not matched on sparsity or scale |
| H-Q3 | Explanations predict held-out activations (AUROC 0.94 top, 0.80 mid) | **weakened** (R1: marked token appears verbatim in the explanation for 179/336 snippets, i.e. token lookup; per-item AUROC agrees) | E1 judges, mismatched control = 0.50 | judges may use surface lexical cues; ratings pooled across items (no per-item analysis); one judge per item |
| H-Q4 | Neurons are a privileged basis (kurtosis 100 vs 0.8) | **weakened** (R1: kurtosis shows sparsity only; the LLM-free one-job proxy shows no difference, top-token share 0.12 vs 0.11, p=0.60) | E2 | kurtosis is driven by rare massive activations / attention-sink tokens (positions 0-3 excluded, but not other sinks); fp16 storage |

### AlgZoo legibility (Q2)
| id | claim | status | evidence | alternatives / weaknesses |
|---|---|---|---|---|
| H-A1 | Legible labels do not add up to the mechanism on M_16,10 (54%) | **narrowed** (R1: driven by the 5-constant limit; one label alone drops acc to 0.79-0.95; ladder has one point) | E4 held-out scoring | **result depends on the 5-constant-per-neuron rule** (free lookup tables on the same features reach 98%); one investigator, limited effort. So it shows "short labels lose skill", not "no compact description exists" |
| H-A2 | The protocol works when understanding is achievable | confirmed | E3 positive control, held-out | M_4,3 is so small that exact formulas are trivially compact |

### M_16,10 mechanism (Q3)
| id | claim | status | evidence | alternatives / weaknesses |
|---|---|---|---|---|
| H-M1 | Fundamental-mode phase encodes time since the max (42.7 deg/step vs 42.8 predicted) | supported | phase concentration 0.86-0.98 by position | eigenbasis taken from the *most common* gate set only; not checked per gate pattern |
| H-M2 | The gated recurrence's eigenvalues form a harmonic series (periods 8.4/4.3/2.9/2) | **weakened** (DRI null test) | ratios 1.95, 2.91, 4.2 in the most common gate set; random 13-neuron subsets show this pattern only 3% of the time | that gate set covers only **16%** of states at steps 6-10; the next common gate sets give shifted periods (7.7/4.7/3.1; 6.7/4.0/3.0); 2 of 5 independently trained seeds show a similar series, 3 do not. The 'Fourier memory' is a description of one common configuration, not of the network |
| H-M3 | n3 and n13 are "higher harmonics" | **weak / withdrawn** (R1 + DRI null test) | variance shares on non-orthogonal modes (shares exceed 1) | the decomposition is not orthogonal, so the shares are not a clean attribution; no causal test |
| H-M4 | Algorithm is correct at canonical spacing (99.96%); all errors are magnitude effects | **near-tautological** (R1: canonical inputs exclude the blind region and are one point per permutation) | `frozen.py` | canonical inputs never include the blind region; "magnitude" is a residual category, not a mechanism |
| H-M5 | Error anatomy by what the model picks: blind 0.82%, picks 3rd 2.14%, picks max 0.52% (x9 below both), rank>=4 ~0.27% | confirmed (R1 recomputed), but it is grouping by the model's pick, not by measured ties | multiple scripts agree | category boundaries (e.g. "near-tie" cut-offs) are choices |
| H-M7 | The network has far too many linear regions to enumerate | confirmed (DRI count + E14 independently: >=444k error cones, >=2.4M prefixes) | 535,261 distinct full gate patterns, 435,052 seen once; top 100k cover 56% of mass; 46,738 errors in 38,743 patterns | a count from a sample underestimates the true number; it does not rule out enumeration of *coarser* classes (prefixes, last-k-step gates, rank classes) |
| H-E6 | Moment/cumulant closure at the decision-margin level cannot reach ~0.1 pp | supported (E13 ceiling test) | exact cumulants per cell: -1.4 pp (2 moments), -0.35 pp (4 cumulants) | finer cells or margins conditioned on more structure might do better; single agent, not re-run |
| H-M8 | Rank order (even full ordering + number of positives) does not fix the gates precisely enough for a ~0.1 pp estimate | supported (E10) | gates frozen at the canonical point of each full ordering+K give 94.2% vs model 95.3% (-1.1 pp); restoring the 5 worst neurons' gates only raises agreement 94.6% -> 98.0% | untested: mixtures of several gate patterns per class weighted by exact polytope masses; treating the x9-comparator gates exactly |
| H-M6 | Group roles (clocks, comparators, ...) via mean-ablation | supported for 13/16 neurons in groups (n2, n5, n9 not ablated); "all 16 causally confirmed" was an overclaim | ablation table | mean-ablation can break the network off-distribution, so damage does not prove the role; no resample-ablation or patching control |

### Estimation (Q4)
| id | claim | status | evidence | alternatives / weaknesses |
|---|---|---|---|---|
| H-E1 | Our deductive probe estimators lose to sampling | supported | E7 under the harness | the harness's noise model is our guess (see H-E4) |
| H-E2 | Exact line integration is unbiased but ~17x worse | supported | lineint validated against brute force on 6 lines | cost accounting = number of pieces; a cheaper walker could change the ratio |
| H-E3 | Stratified sampling beats sampling 1.2-2.8x | **corrected by R1, confirmed by DRI** (`m16/stratified_eval.py`: orig 2.10x): exact variance ratios 2.07x (orig), 1.78x (sigma 0.0025), 1.36x (sigma 0.005, two copies < 1); the reported 2.6-2.8x values were MC noise; candidate generation (~12.9 draws + sorts per eval) may erase the gain under FLOP accounting; the evaluation script was not committed | E9: 12 cells, all > 1 | compute counts model evals only (rejection sampling draws ~13 candidates per eval); MSE ratio from about 72 trials per cell, so about +-25% noise per cell; the allocation pilot ran on the original model; gain shrinks with noise |
| H-E4 | Our harness implements ARC's protocol | **unverified**; R1: sigma levels were our choice (code default 0.05 not reported; at 0.02-0.05 accuracy drops to 26-77%); MSE from ~48 squared errors has ~+-20% relative s.e., so ratios need CIs | ARC: "add noise enough to significantly alter accuracy" | the noise scale (sigma x matrix RMS) and the levels (0.0025, 0.005) are our choices; ARC may use seeds rather than noise; FLOPs vs evals accounting is not settled |
| H-E5 | "Someone at ARC has cracked M_16,10" | **unverified** | private message relayed by the user | not checked by design (independence) |

## 5. Open questions / next decisions
0. R1 follow-ups: re-run H2 with >=100 controls (neurons projected out, or dense rotation) and 2-3 explainers per item; n-gram dedupe for judges; sweep the label constant budget; finish M_32,10; count candidate-generation FLOPs for stratified; add CIs to all MSE ratios; report a higher-noise sigma too.
1. Check H-Q2's alternative: are random directions dominated by massive-activation neurons? (needs a Qwen re-run; deferred while CPU is saturated)
2. Orthogonal/causal test for H-M3; check H-M1's phase clock across the other common gate sets. (The H-M2 null test is done: weakened.)
3. Settle compute accounting (FLOPs incl. input generation) and re-run H-E3 with more repeats.
4. Collect approach-agent results into the registry; independently re-run the best one before believing it.
5. Reviewer report (R1): fold into the ledger.
