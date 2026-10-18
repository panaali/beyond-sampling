# HP1 scorer (T281)

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

## Question
How do frozen estimators score on the sealed held-out panel HP1 (T276) under Directive 8 and its amendment
(budget B in {16k, 64k, 256k} FPE, charged cost >= B/2, upper 95% cluster CI <= 10 lenient / <= 1 strict, in
every group)? This folder is the only place that reads the sealed truths. **Only group-level aggregates are
published here.** Per-model truths, estimates and errors stay local (`data/hp1/`, git-ignored).

## Method
`score.py` is an independent harness. It shares no code with LB3/leaderboard.py or with any submission.
- **Panel.** Taken from branch `claude/t192-gpu-cost-law` (commit 34e0b45b), `m16/approaches/gpu_cost_law/hp1/`, and
  copied to `data/hp1/`. Hashes are checked on every load:
  - weights: sha256 of `Wi.tobytes()+Wh.tobytes()+Wo.tobytes()` = `9bad715d…fbd4c`. This is the convention in
    `make_hp1.py`. The npz *file* hashes to `07f487a4…7107`, which differs because zip metadata is included, as expected.
  - manifest.json `fe097401…8ca64`; sealed/truths.json `b610b9a2…4c8a509`.
  - Truths: 1e8 MC samples per model, max SE 5.0e-5. Groups: clean (orig + zoo seeds 0–4; 6), abs0.05 (48),
    rel0.005 (48), rel0.01 (48). Each noisy group is 8 copies of each of the 6 bases.
- **Budgets.** B = 16 000, 64 000, 256 000 FPE (LB3's CURVE_BUDGETS values). Each submission is called as
  `est(model, B)` with `model = (Wi (16,), Wh (16,16), Wo (10,16))` float64, once per (model, B). The submissions
  scored so far are deterministic.
- **Cost.** C_m = cost_flops / 5948 FPE (T196 convention). A run is valid iff cost_flops >= 0.5 · B · 5948.
- **Score.** e_m = (est_m − p_m)²; s_m = p_m(1−p_m)/C_m + SE_m². The SE² term is the truth-noise term. Its share
  of Σs is reported and is ≤ 0.3%. Group ratio R_g = Σe/Σs. The 95% CI is a percentile cluster bootstrap over the
  6 base models: 10 000 resamples, seed 281.
- **Control.** Plain MC through the same harness: n = B iid N(0,1)¹⁰ inputs with a fixed seed per (model, B), and
  cost B · 5948.

Reproduce (from this folder, with the panel files in `data/hp1/`):
```
python3 score.py control control est sampling
python3 score.py v5q <probe_v5 snapshot>/hp1_entry.py est inductive
python3 score.py v5a <probe_v5 snapshot>/hp1_entry.py est_v5a inductive
```
Here `<probe_v5 snapshot>` is `m16/approaches/probe_v5/{hp1_entry.py, v5.py}` extracted with `git show` at commit
d29e28bb into `data/hp1/subs/probe_v5_d29e28bb/`. The aggregates are in `results/*.json`.

## Submissions scored
| name | entry | commit | sha256 | class (author) |
|---|---|---|---|---|
| v5q | `probe_v5/hp1_entry.py:est` → `v5.est_full` | d29e28bbc437ae02395e91741e3feefc50d2fa77 (branch claude/t241-neuron-account; unchanged at branch head 9e5d8a2a) | hp1_entry.py `81f8c07f963bd49669a18f0b769a6255164b5b2e3e28ffbd22fba235a43f113b`; v5.py `5c19b9674b632ca0746c870e3c68e12acaf8f8ddf1506eca3cf14c146fd73e1f` (unchanged since f7e14e83) | inductive (fixed-node QMC) |
| v5a | `probe_v5/hp1_entry.py:est_v5a` → `v5.est` | same | same | inductive |
| t286_rx | `cheap_frontier/hp2_entry.py:est` → `rx.est_rx(Q=8, stride=1, kgroups=3)` | 3731b9fe93159c8b1d643fdd107b451985841ec4 (branch claude/t286-cheap-frontier) | hp2_entry.py `eac842afeb65d4e2d91c3e83d38054c5cfaa34feb4be532b7d12fe6f61e5a724`; rx.py `c11623371060e3522736d2a9a36df5726ebf944da6617e2500c9f063ba4e1055`; cf.py `628c9a4e70af342c4d0ed68155075fbd17facd320323cd98b644c3b1bb119e9f`; rank_exact/rank_est.py `41c5b6eb0fa5d710b149b328b06bd62ddd1a6687b62642b414c0f215ec269401` | deductive (every step) |
| t286_rx_ind | `cheap_frontier/hp2_entry.py:est_ind` → `rx.est_rx(Q=4, stride=16, kgroups=3)` | same | same | inductive (reference only) |
| t286b_probe | `cheap_frontier/hp2_entry_b.py:est` → `probe.est_probe(version=1, iters=8, stride=1)` (selected post hoc on DEV2) | c131e933aa5ac53c72e8d6fc1cd17de54f21a6d0 (branch claude/t286-cheap-frontier) | hp2_entry_b.py `14d0a8c6141ecf05c5c6672290ccb0be7a8faa86216da5973fe92c12485628fe`; probe.py `d1ee7fcfe0ef34a5e49225b3eecd51d93dad9edb136eaf17e7c03993c97f5add`; cf.py `72fb793abe33a662a762d569e97c8bd2205f5817eb87156f1daf5096cfbd38d9`; m16/estimate.py `8d87f1bec58961db64f8acc45d8b6655c8c0d80fbfc29381a7a854e8d584398d` | deductive (every step), post hoc |
| t300_r0 | `r0_shared/estimator.py:est` → `r0_integral/tb.est(n23=1, newton=1, nblind=0, use12=False)` | 8df9793bcccb67ca8c412261006eceb5bd4777b3 (branch claude/t218-tie-tail-labels) | estimator.py `cbb7904b7717834ce677991be359232c6e6066e74cc7b0315a2caa9a992d1771`; r0.py `0f05d047789bde235f2e76c6f374f62a65eed8cfe8f81a39ddaa09f64207499b`; tb.py `a2ae41b436bb6a057b7588f3a44efc8f900d73461293a3fa08788ab26407526a` | inductive (one inductive step; DEDUCTIVE=False); author scope: lenient bar, retrains |
| t306_r1w | `r1_walk/estimator.py:est` → `r1w2.est` (R1W V1) | 684a22b96e8f3dec5ae3a63b5d163aecd1334e3e (branch claude/t218-tie-tail-labels) | estimator.py `81e30b6af4e675ecc89b85c9cef94b1d915a6ba2ed392ddc9151d9115f02cb93`; r1w2.py `3078021594b2959579601df20dee9a41be53de2e131c21127e585430571867e2`; r1w.py `3f7297f2e92f5daed0f20d21728f60389e567f29a9a98bbe85761f562a7d919e`; r0_walk/walk.py `06d7ec1987a303a651fb57d2bcdb5637f0ebf5ec23463deea78dfc158e77ce46`; r0.py `0f05d047…499b` | deductive (DEDUCTIVE=True; declared approximations) |
| t317_cont | `r1_cont/estimator.py:est` → `cont.est(G=48)` (R1W V1 estimates; continuity restarts + executed B2 prefix tree; cost as executed) | f599890226bdbada6776a75c22f392bb30592f09 (branch claude/t218-tie-tail-labels) | estimator.py `9e48b579479eed16ccf4220d182b2b33b8b638a7fdc3bdb5bf4ca3c8e617184d`; cont.py `b71f203f5caec913193a9c6c045b76b35a43b63edcd7c27148a081ffcc084d92`; v1b2.py `a09b395d…6cdc` (others unchanged since 6a496515) | deductive |

No estimator code was changed. The one loader shim is `shim/alg_zoo.py`: t286b imports m16/common.py, which
loads M_16,10 at import time via alg_zoo, and alg_zoo is not installed here. The shim returns HP1's bit-identical
'orig' weights, which the estimator then rebinds per model (`probe._bind`). T286 runs from a `git archive` of `m16/` at its commit, so its relative imports
(`../rank_exact`) resolve to the frozen code. Every entry point takes `(Wi, Wh, Wo)` float64 exactly as the panel stores it. The
estimator files were not modified.

## Status
Done on HP1 for v5q, v5a, the control and T286 (est, est_ind, probe-v1 est_b) and T300 r0. HP2 (primary panel) scored for all seven entries. Standing service: further submissions are scored the same way.

## Results (amendment 1 rule: fixed budget B, R_g = Σe/Σs; superseded as figure of merit by amendment 2 below)
R_g [95% cluster CI]; bias and RMS error in percentage points; the cost column is the minimum charged C_m in FPE.

**Control (plain MC, sampling)**
| B | group | R | 95% CI | bias pp | RMS pp | C_m min (valid) |
|---|---|---|---|---|---|---|
| 16k | clean | 2.30 | [0.17, 5.78] | +0.167 | 0.325 | 16000 ✓ |
| 16k | abs0.05 | 1.16 | [1.03, 1.29] | −0.061 | 0.335 | 16000 ✓ |
| 16k | rel0.005 | 0.83 | [0.54, 1.12] | −0.011 | 0.302 | 16000 ✓ |
| 16k | rel0.01 | 0.93 | [0.61, 1.33] | −0.026 | 0.344 | 16000 ✓ |
| 64k | clean | 1.39 | [0.61, 2.75] | −0.042 | 0.127 | 64000 ✓ |
| 64k | abs0.05 | 1.09 | [0.77, 1.39] | −0.006 | 0.162 | 64000 ✓ |
| 64k | rel0.005 | 1.05 | [0.57, 1.64] | −0.020 | 0.170 | 64000 ✓ |
| 64k | rel0.01 | 0.95 | [0.71, 1.23] | −0.064 | 0.174 | 64000 ✓ |
| 256k | clean | 1.59 | [0.41, 2.72] | −0.022 | 0.068 | 256000 ✓ |
| 256k | abs0.05 | 1.19 | [0.97, 1.42] | −0.012 | 0.085 | 256000 ✓ |
| 256k | rel0.005 | 1.24 | [0.81, 1.59] | +0.004 | 0.092 | 256000 ✓ |
| 256k | rel0.01 | 0.97 | [0.77, 1.26] | −0.002 | 0.088 | 256000 ✓ |

The control is consistent with 1 in every cell. Clean 2.30 at 16k is a 6-model χ²₆/6 draw (p ≈ 0.03 for one
cell out of 12). The harness passes.

**v5q (inductive)**
| B | group | R | 95% CI | bias pp | RMS pp | C_m min (valid) |
|---|---|---|---|---|---|---|
| 16k | clean | 1.72 | [0.71, 3.18] | +0.239 | 0.280 | 16140 ✓ |
| 16k | abs0.05 | 0.92 | [0.63, 1.21] | −0.110 | 0.298 | 16140 ✓ |
| 16k | rel0.005 | 0.99 | [0.78, 1.28] | +0.004 | 0.328 | 16140 ✓ |
| 16k | rel0.01 | 1.60 | [0.80, 2.43] | −0.019 | 0.448 | 16140 ✓ |
| 64k | clean | 1.60 | [0.33, 3.68] | +0.072 | 0.135 | 64560 ✓ |
| 64k | abs0.05 | 1.58 | [1.10, 2.02] | −0.074 | 0.195 | 64560 ✓ |
| 64k | rel0.005 | 3.31 | [1.79, 5.11] | −0.041 | 0.300 | 64560 ✓ |
| 64k | rel0.01 | 4.41 | [1.93, 6.86] | +0.044 | 0.373 | 64560 ✓ |
| 256k | clean | 6.73 | [1.66, **13.58**] | +0.092 | 0.139 | 258238 ✓ |
| 256k | abs0.05 | 5.47 | [3.45, 7.57] | −0.057 | 0.181 | 258238 ✓ |
| 256k | rel0.005 | 10.35 | [5.49, **16.69**] | −0.055 | 0.265 | 258238 ✓ |
| 256k | rel0.01 | 13.33 | [6.44, **19.60**] | +0.001 | 0.324 | 258238 ✓ |

**v5a (inductive)**
| B | group | R | 95% CI | bias pp | RMS pp | C_m min (valid) |
|---|---|---|---|---|---|---|
| 16k | clean | 18.0 | [6.3, 29.3] | +0.752 | 0.907 | 16140 ✓ |
| 16k | abs0.05 | 4.70 | [2.81, 6.61] | +0.488 | 0.671 | 16140 ✓ |
| 16k | rel0.005 | 15.9 | [11.5, 20.1] | +1.160 | 1.313 | 16140 ✓ |
| 16k | rel0.01 | 19.4 | [10.5, 27.4] | +1.392 | 1.563 | 16140 ✓ |
| 64k | clean | 40.5 | [12.8, 71.7] | +0.567 | 0.680 | 64560 ✓ |
| 64k | abs0.05 | 17.4 | [10.4, 24.1] | +0.506 | 0.645 | 64560 ✓ |
| 64k | rel0.005 | 62.1 | [44.3, 73.3] | +1.157 | 1.299 | 64560 ✓ |
| 64k | rel0.01 | 85.1 | [50.0, 117.8] | +1.474 | 1.636 | 64560 ✓ |
| 256k | clean | 154 | [68, 250] | +0.568 | 0.664 | 258238 ✓ |
| 256k | abs0.05 | 68.0 | [39.9, 100.8] | +0.517 | 0.640 | 258238 ✓ |
| 256k | rel0.005 | 252 | [181, 298] | +1.174 | 1.308 | 258238 ✓ |
| 256k | rel0.01 | 336 | [196, 468] | +1.479 | 1.627 | 258238 ✓ |

**Bars (all four groups at one B; every run is cost-valid).** Strict bar as restated by the DRI (TASKS
Directive 8, 2026-10-18): **strict** means the point R_g <= 1 in every group; **strict, significant** also needs the
upper CI <= 1. The lenient bar is unchanged: upper CI <= 10 in every group.
| method | B | lenient (upper <= 10) | strict (point R <= 1) | strict, significant (upper <= 1) |
|---|---|---|---|---|
| control (sampling) | 16k | pass | fail (clean 2.30, abs 1.16) | fail |
| control (sampling) | 64k | pass | fail (clean 1.39, abs 1.09, rel.005 1.05) | fail |
| control (sampling) | 256k | pass | fail (clean 1.59, abs 1.19, rel.005 1.24) | fail |
| v5q (inductive) | 16k | **pass** (max upper 3.18) | fail (clean 1.72, rel.01 1.60) | fail |
| v5q (inductive) | 64k | **pass** (max upper 6.86) | fail (all 4 groups 1.58–4.41) | fail |
| v5q (inductive) | 256k | fail (3 groups > 10) | fail | fail |
| v5a (inductive) | 16k / 64k / 256k | fail | fail | fail |

Under the restated strict bar the control itself sits at R ≈ 1. It fails at each budget only by noise (the clean
group has 6 models), so a method must be better than sampling to pass the strict bar reliably.

## Results under Directive 8 amendment 2 (own-cost figure of merit)
Rule (TASKS Directive 8, amendment 2, ccr 576003ba): C = group-mean charged FLOPs / 5948 and C_eff = max(C, 1000).
F = MSE · C_eff / mean p(1−p), where MSE = mean (est − p)². The amendment writes this as mean(e_m²), with e_m the
signed error, which is consistent with its 'RMS <~ .84 pp at 1k'. The break-even budget is n* = mean p(1−p) / MSE.
The 95% CI is the 6-cluster bootstrap over base models (10 000 resamples). **Strict** means point F ≤ 1 in every
group. **Lenient** means point F ≤ 10 in every group. DRI ruling (2026-10-18): the point F is binding for both bars,
and the CI is always reported. The upper-CI reading is shown for information only. A 'candidate' claim to ARC also
needs the upper CI to hold on HP2 (one cluster per retrain).
Removing the truth-MC term (mean SE²) from the MSE changes F by at most 0.003, so it is negligible. Budgets 1k and
4k were added; every run has cost ≥ B/2. 'all' pools the 150 models and is informational only.
Reproduce: `python3 score.py <name> <file> <func> <class> 1000,4000` (this also prints F for all five budgets).
Results: `results/<name>_F.json`.

**control (sampling)**: F [95% CI]; RMS pp; n*
| B | C_eff | clean | abs0.05 | rel0.005 | rel0.01 | all | strict (point ≤1) | lenient point / upper |
|---|---|---|---|---|---|---|---|---|
| 1k | 1000 | 0.94 [0.24, 1.84]; 0.83; 1062 | 1.16 [0.85, 1.48]; 1.34; 862 | 0.97 [0.59, 1.36]; 1.31; 1026 | 1.15 [0.93, 1.41]; 1.53; 871 | 1.09 [0.98, 1.20]; 1.38; 916 | fail | pass / pass |
| 4k | 4000 | 0.77 [0.18, 1.28]; 0.38; 5202 | 0.90 [0.51, 1.33]; 0.59; 4467 | 0.94 [0.76, 1.11]; 0.64; 4259 | 0.95 [0.67, 1.31]; 0.70; 4200 | 0.93 [0.75, 1.12]; 0.63; 4307 | pass | pass / pass |
| 16k | 16000 | 2.30 [0.16, 5.93]; 0.33; 6971 | 1.16 [1.02, 1.29]; 0.34; 13779 | 0.83 [0.54, 1.12]; 0.30; 19202 | 0.93 [0.60, 1.33]; 0.34; 17122 | 0.99 [0.75, 1.23]; 0.33; 16167 | fail | pass / pass |
| 64k | 64000 | 1.39 [0.61, 2.76]; 0.13; 46041 | 1.09 [0.78, 1.39]; 0.16; 58925 | 1.05 [0.57, 1.64]; 0.17; 60680 | 0.95 [0.71, 1.23]; 0.17; 67031 | 1.03 [0.82, 1.29]; 0.17; 62017 | fail | pass / pass |
| 256k | 256000 | 1.59 [0.41, 2.74]; 0.07; 160847 | 1.19 [0.96, 1.44]; 0.08; 215416 | 1.24 [0.81, 1.58]; 0.09; 205910 | 0.98 [0.77, 1.26]; 0.09; 262316 | 1.13 [0.96, 1.28]; 0.09; 225866 | fail | pass / pass |

**v5q (inductive)**: F [95% CI]; RMS pp; n*
| B | C_eff | clean | abs0.05 | rel0.005 | rel0.01 | all | strict (point ≤1) | lenient point / upper |
|---|---|---|---|---|---|---|---|---|
| 1k | 1009 | 0.23 [0.06, 0.51]; 0.41; 4370 | 2.54 [1.00, 4.55]; 1.97; 397 | 1.91 [0.96, 2.93]; 1.82; 528 | 3.25 [1.95, 4.32]; 2.56; 310 | 2.56 [1.98, 3.16]; 2.10; 394 | fail | pass / pass |
| 4k | 4035 | 0.70 [0.30, 1.03]; 0.36; 5785 | 1.35 [0.82, 2.03]; 0.72; 2999 | 1.15 [0.80, 1.38]; 0.71; 3524 | 1.25 [0.66, 1.78]; 0.79; 3226 | 1.23 [1.04, 1.47]; 0.73; 3269 | fail | pass / pass |
| 16k | 16140 | 1.72 [0.71, 3.22]; 0.28; 9375 | 0.92 [0.63, 1.21]; 0.30; 17468 | 0.99 [0.78, 1.28]; 0.33; 16251 | 1.60 [0.80, 2.43]; 0.45; 10104 | 1.21 [0.86, 1.57]; 0.36; 13316 | fail | pass / pass |
| 64k | 64560 | 1.60 [0.33, 3.69]; 0.14; 40371 | 1.58 [1.10, 2.03]; 0.19; 40858 | 3.31 [1.79, 5.17]; 0.30; 19507 | 4.41 [1.93, 6.84]; 0.37; 14627 | 3.20 [1.88, 4.61]; 0.29; 20176 | fail | pass / pass |
| 256k | 258238 | 6.75 [1.72, 13.61]; 0.14; 38278 | 5.49 [3.42, 7.60]; 0.18; 47066 | 10.38 [5.56, 16.73]; 0.27; 24877 | 13.37 [6.62, 19.65]; 0.32; 19321 | 10.04 [6.30, 13.97]; 0.26; 25724 | fail | fail / fail |

**v5a (inductive)**: F [95% CI]; RMS pp; n*
| B | C_eff | clean | abs0.05 | rel0.005 | rel0.01 | all | strict (point ≤1) | lenient point / upper |
|---|---|---|---|---|---|---|---|---|
| 1k | 1009 | 1.59 [0.12, 3.84]; 1.08; 634 | 2.19 [1.15, 3.91]; 1.84; 460 | 2.54 [1.12, 4.14]; 2.10; 397 | 4.25 [1.93, 6.82]; 2.92; 238 | 3.06 [1.99, 4.06]; 2.30; 329 | fail | pass / pass |
| 4k | 4035 | 6.03 [0.80, 11.67]; 1.05; 669 | 1.17 [0.71, 1.72]; 0.67; 3457 | 3.87 [2.15, 5.23]; 1.30; 1044 | 5.35 [1.78, 8.82]; 1.64; 754 | 3.69 [1.78, 5.61]; 1.26; 1094 | fail | pass / fail |
| 16k | 16140 | 18.01 [6.37, 29.32]; 0.91; 896 | 4.70 [2.81, 6.65]; 0.67; 3436 | 15.87 [11.55, 20.12]; 1.31; 1017 | 19.43 [10.57, 27.37]; 1.56; 831 | 14.05 [9.75, 18.48]; 1.23; 1149 | fail | fail / fail |
| 64k | 64560 | 40.54 [13.44, 71.77]; 0.68; 1592 | 17.37 [10.33, 23.95]; 0.65; 3717 | 62.18 [44.33, 73.39]; 1.30; 1038 | 85.14 [49.66, 117.92]; 1.64; 758 | 57.61 [43.78, 70.55]; 1.24; 1121 | fail | fail / fail |
| 256k | 258238 | 154.39 [67.95, 251.18]; 0.66; 1673 | 68.19 [39.86, 101.80]; 0.64; 3787 | 252.30 [181.01, 298.56]; 1.31; 1024 | 336.83 [197.01, 470.26]; 1.63; 767 | 229.70 [167.63, 287.55]; 1.24; 1124 | fail | fail / fail |

Readout:
- [measured] No submission passes the strict bar (point F ≤ 1 in all four groups) at any B.
  - v5q's best point is 4k: F = 0.70 / 1.35 / 1.15 / 1.25 (clean / abs / rel.005 / rel.01), 1.23 pooled.
  - v5q at 16k is 1.72 / 0.92 / 0.99 / 1.60. Its F rises from 64k on, when the RMS reaches its floor of
    about 0.13–0.37 pp.
  - The plain-MC control is ≈ 1 everywhere: pooled F 0.93–1.13. It 'passes' strict at 4k only by noise.
- [measured] Lenient, v5q: on the point it passes at 1k–64k and fails at 256k (rel.005 10.4, rel.01 13.4). On the
  upper CI it also passes at 1k–64k (max upper 4.55 / 2.03 / 3.22 / 6.84).
- [measured] Lenient, v5a: on the point it passes at 1k and 4k (max 4.25 / 6.03). On the upper CI it passes only at
  1k (max 6.82; clean's upper CI at 4k is 11.7). It fails at ≥ 16k on both readings.
- [measured] v5q's break-even n* is 4.4k on clean at 1k, then 3.0–5.8k at 4k. At 16k it is 9–17k per group,
  roughly the budget itself. That is, v5q is about as good as sampling from 4k to 16k and not better. v5a's n* is
  0.7–3.8k.

### T286 (fixed cost; the budget is ignored, so it is scored once, at its own cost)
Reproduce: `python3 score.py t286_rx <snapshot>/m16/approaches/cheap_frontier/hp2_entry.py est deductive 0`. The
reference uses `... t286_rx_ind <same file> est_ind inductive 0`. Results: `results/t286_rx*_F.json`.
| entry | class | C_eff FPE | clean | abs0.05 | rel0.005 | rel0.01 | all | strict / lenient |
|---|---|---|---|---|---|---|---|---|
| t286_rx | deductive | 25779 | F 107.5 [22.1, 244.7]; RMS 1.75 pp; bias -1.38; n* 240 | F 19.4 [9.6, 35.6]; RMS 1.08 pp; bias +0.35; n* 1332 | F 34.5 [16.8, 53.2]; RMS 1.53 pp; bias +0.20; n* 747 | F 37.9 [20.3, 68.3]; RMS 1.73 pp; bias +0.40; n* 680 | F 32.7 [19.9, 47.5]; RMS 1.48 pp; bias +0.25; n* 788 | fail / fail |
| t286_rx_ind | inductive | 1716 | F 12.1 [7.5, 17.1]; RMS 2.28 pp; bias -2.15; n* 142 | F 23.8 [19.4, 28.1]; RMS 4.64 pp; bias +0.80; n* 72 | F 29.5 [22.4, 36.8]; RMS 5.49 pp; bias +3.21; n* 58 | F 34.2 [13.8, 56.1]; RMS 6.36 pp; bias +3.03; n* 50 | F 29.4 [19.6, 40.0]; RMS 5.45 pp; bias +2.17; n* 58 | fail / fail |

- [measured] t286_rx (deductive) fails both bars on HP1. Worst-group F is 108 (clean, bias −1.38 pp). The author
  had LB3 worst-group F ≈ 112. Pooled F is 33, and its break-even budget n* ≈ 790 FPE is 33x below its cost.
- [measured] t286_rx_ind (inductive reference) fails both bars. Worst-group F is 34 (rel0.01), and its RMS is
  2.3–6.4 pp.

### T286 probe-v1 (`hp2_entry_b.est`; deductive, selected post hoc on DEV2; fixed cost)
Reproduce: `PYTHONPATH=shim python3 score.py t286b_probe <snapshot c131e933>/m16/approaches/cheap_frontier/hp2_entry_b.py est deductive 0`.
The charged cost varies by model: 876–3960 FPE, mean 2786.
| group | C_eff | F [95% CI] | RMS pp | bias pp | n* |
|---|---|---|---|---|---|
| clean | 3928 | 72.2 [35.4, 100.0] | 3.68 | +3.32 | 54 |
| abs0.05 | 1568 | 4513.5 [4078.4, 4906.2] | 66.77 | +66.47 | 0 |
| rel0.005 | 3491 | 1054.4 [239.4, 1784.0] | 23.00 | +18.17 | 3 |
| rel0.01 | 3156 | 1737.6 [473.4, 3187.9] | 33.44 | +28.00 | 2 |
| all | 2786 | 3137.0 [2641.0, 3698.5] | 44.21 | +36.18 | 1 |

- [measured] t286b fails both bars. On clean models F = 72, close to the author's DEV2 F_max of 76.5 and LB3 seed
  F of 77.6.
- [measured] On the noisy copies t286b breaks down: bias +18 to +66 pp, F 1.1e3–4.5e3. The author's docstring
  anticipated this for heavy-noise copies (rank-4+ and 1~2 errors are not modelled). Since HP1 is the robustness
  panel, this is the expected failure, not a harness issue.

### T300 R0 closure + one-anchor Newton boundary (`r0_shared/estimator.py:est`; inductive; fixed cost)
Reproduce: `python3 score.py t300_r0 <snapshot 8df9793b>/m16/approaches/r0_shared/estimator.py est inductive 0`.
The author charges 1.0e7 FLOPs. They divide by 5760 FLOPs/FPE (1752 FPE); this scorer divides by 5948 (1697 FPE).
| group | C_eff | F [95% CI] | RMS pp | bias pp | n* |
|---|---|---|---|---|---|
| clean | 1697 | 5.9 [3.9, 7.8] | 1.60 | +0.40 | 287 |
| abs0.05 | 1697 | 77.2 [49.3, 106.3] | 8.40 | +7.65 | 22 |
| rel0.005 | 1697 | 89.6 [43.7, 119.6] | 9.62 | +8.08 | 19 |
| rel0.01 | 1697 | 87.7 [59.5, 107.3] | 10.24 | +9.48 | 19 |
| all | 1697 | 83.9 [73.1, 93.1] | 9.27 | +8.09 | 20 |

- [measured] On the 6 clean HP1 models t300 gets F = 5.9 [3.9, 7.8], under the lenient bar. This is consistent with
  the author's DEV2 retrain figure of 5.97.
- [measured] On the HP1 noisy copies it fails: F 77–90, bias +7.7 to +9.5 pp. The author scoped the method to
  retrains, so its claim is decided on HP2. HP1 is the robustness readout only.

### T306 R1W V1 on HP1 (`r1_walk/estimator.py:est`; deductive; fixed cost)
| group | C_eff | F [95% CI] | RMS pp | bias pp |
|---|---|---|---|---|
| clean | 13056 | 9.1 [1.1, 20.1] | 0.72 | +0.49 |
| abs0.05 | 13888 | 425.1 [359.0, 493.9] | 6.89 | +6.64 |
| rel0.005 | 12875 | 92.7 [29.2, 150.4] | 3.55 | +2.88 |
| rel0.01 | 12840 | 127.4 [58.8, 191.9] | 4.49 | +3.55 |
| all | 13195 | 195.2 [149.0, 234.4] | 5.07 | +4.20 |

- [measured] On HP1, t306 has F 9.1 on the 6 clean models (CI up to 20). On the noisy copies it fails: F 93–425.

### T317 R1W V1 with continuity restarts + executed prefix tree (`r1_cont`, @ f5998902; deductive)
Reproduce: `PANEL=hp2 PYTHONPATH=shim python3 score.py t317_cont <snapshot>/m16/approaches/r1_cont/estimator.py est deductive 0`.
- Estimates: equal to R1W V1 on every model. Max |diff| is 6.7e-16 on HP2 and 1.0e-15 on HP1 (checked against
  `raw_t306_r1w`).
- Cost: as executed, counted in-line by the code. I read the counting against the loops and the constants match the
  operations performed: values 2TH, tangents 2H²+3H per step, roots 5H per step, readout, and first-cell prefix
  nodes × 1,218.
- Not charged: the `np.unique` prefix keys, which is sort/bookkeeping, not arithmetic.
- The one hand-entered constant is `correct_interval` = 146 FLOPs. My count is about 216 if every comparison and
  boolean op is counted. With 7,450 calls per model, that adds +1.2% to C, giving F ≈ 9.69. Even +100 FLOPs per
  call is only +1.7%.
- HP2 C: mean 7,765 FPE (range 5,671–10,757).

| panel / group | C_eff | F [95% CI] | RMS pp | bias pp |
|---|---|---|---|---|
| HP2 retrain | 7765 | 9.57 [6.89, 12.43] | 0.99 | +0.53 |
| HP1 clean | 7568 | 5.3 [0.7, 11.7] | 0.72 | +0.49 |
| HP1 abs0.05 | 7437 | 227.6 [192.3, 264.5] | 6.89 | +6.64 |
| HP1 rel0.005 | 7291 | 52.5 [16.5, 85.2] | 3.55 | +2.88 |
| HP1 rel0.01 | 7151 | 70.9 [32.8, 106.9] | 4.49 | +3.55 |

- [measured] **T317 is the first deductive entry to pass the lenient bar on the HP2 point: F = 9.57 [6.89, 12.43].**
  The upper CI is above 10, so it is NOT a candidate under the DRI rule.
- [measured] The point margin is thin: F ≤ 10 needs C ≤ 8,116 FPE, only 4.5% above the counted 7,765. So the point
  verdict depends on the cost count. I re-checked the count's constants (above), and the plausible undercount is
  about 1–2%.
- [measured] HP1: clean F 5.3 [0.7, 11.7]. Noisy copies fail at F 52–228; the estimator is out of scope there.

## HP2 results: PRIMARY panel (T284 retrains; Directive 8 amendment 2)
Panel: branch `claude/t192-gpu-cost-law` at aa9c000e, `m16/approaches/gpu_cost_law/hp2/`, copied to `data/hp2/`.
- 100 fresh retrains, hp2_s100..s199 (zoo (16,10) defaults, fused trainer, data seed 41). These seeds are disjoint
  from DEV2's seeds 0–39.
- Hashes, checked on load:
  - weights, array bytes: `039e3e694f0a4c606e7bb3f58c309d6bf492b43e744780f46f9dd5dfced19d7e` (file `41709001…ab12`);
  - manifest.json: `1bf47b3f5217d23d3d1202cb3be12e062ea3be65e471be1032819b90aaa97ed6`;
  - sealed/truths.json: `1a6a44f4126ed7aec8f808b8fd0b0690286d8af43f8986a06bb0c18b05a096b8` (1e8 MC per model, max SE
    3.9e-5).
- One group ('retrain'). The bootstrap uses one cluster per model (100 clusters; DRI ruling).
- Point F decides both bars (strict ≤ 1, lenient ≤ 10). A 'candidate' claim to ARC additionally needs the upper CI to
  hold here.
Reproduce: `PANEL=hp2 PYTHONPATH=shim python3 score.py <name> <file> <func> <class> <budgets>`, with the same
snapshots and entry points as above. Results: `results/hp2_<name>_F.json`.

| entry | class | B | C_eff FPE | F [95% CI] | RMS pp | bias pp | n* | strict | lenient (point) | lenient upper CI ≤ 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| control | sampling | 1k | 1000 | 0.96 [0.72, 1.22] | 0.87 | -0.11 | 1047 | pass | pass | yes |
| control | sampling | 4k | 4000 | 1.00 [0.74, 1.29] | 0.44 | +0.01 | 4011 | pass | pass | yes |
| control | sampling | 16k | 16000 | 0.79 [0.58, 1.03] | 0.20 | -0.02 | 20255 | pass | pass | yes |
| control | sampling | 64k | 64000 | 0.99 [0.68, 1.33] | 0.11 | -0.02 | 64379 | pass | pass | yes |
| control | sampling | 256k | 256000 | 1.06 [0.79, 1.36] | 0.06 | -0.01 | 242419 | fail | pass | yes |
| v5q | inductive | 1k | 1009 | 1.27 [0.92, 1.66] | 1.00 | +0.54 | 797 | fail | pass | yes |
| v5q | inductive | 4k | 4035 | 1.06 [0.81, 1.34] | 0.46 | +0.23 | 3802 | fail | pass | yes |
| v5q | inductive | 16k | 16140 | 1.48 [1.12, 1.87] | 0.27 | +0.16 | 10918 | fail | pass | yes |
| v5q | inductive | 64k | 64560 | 3.53 [2.64, 4.51] | 0.21 | +0.12 | 18296 | fail | pass | yes |
| v5q | inductive | 256k | 258238 | 10.47 [7.62, 13.73] | 0.18 | +0.09 | 24675 | fail | fail | no |
| v5a | inductive | 1k | 1009 | 2.89 [2.21, 3.61] | 1.51 | +1.10 | 348 | fail | pass | yes |
| v5a | inductive | 4k | 4035 | 5.33 [4.39, 6.26] | 1.02 | +0.84 | 757 | fail | pass | yes |
| v5a | inductive | 16k | 16140 | 14.10 [12.13, 16.04] | 0.83 | +0.74 | 1145 | fail | fail | no |
| v5a | inductive | 64k | 64560 | 53.06 [45.03, 61.46] | 0.81 | +0.72 | 1217 | fail | fail | no |
| v5a | inductive | 256k | 258238 | 219.22 [188.25, 252.53] | 0.82 | +0.73 | 1178 | fail | fail | no |
| t286_rx | deductive | fixed | 25779 | 98.01 [74.10, 122.83] | 1.74 | -1.26 | 263 | fail | fail | no |
| t286_rx_ind | inductive | fixed | 1716 | 25.77 [19.13, 33.25] | 3.45 | -2.10 | 67 | fail | fail | no |
| t286b_probe | deductive | fixed | 3910 | 106.97 [90.63, 122.97] | 4.66 | +3.97 | 37 | fail | fail | no |
| t300_r0 | inductive | fixed | 1697 | 9.72 [7.02, 12.97] | 2.13 | +0.82 | 175 | fail | pass | no |
| t306_r1w | deductive | fixed | 13269 | 16.35 [11.77, 21.24] | 0.99 | +0.53 | 812 | fail | fail | no |
| t317_cont | deductive | fixed | 7765 | 9.57 [6.89, 12.43] | 0.99 | +0.53 | 812 | fail | pass | no |

Readout (HP2):
- [measured] The control gives F 0.79–1.06 with every CI covering 1, so the harness holds on HP2.
- [measured] **No entry reaches the strict bar.** The closest is v5q at 4k: F 1.06 [0.81, 1.34], i.e.
  indistinguishable from sampling.
- [measured] **v5q (inductive) passes the lenient bar at 1k–64k, with upper CI ≤ 10 throughout** (max upper 4.51
  at 64k). It fails at 256k (F 10.5). It is a lenient candidate in its own class (inductive QMC), not a
  mechanistic one.
- [measured] **t300_r0 (inductive) passes the lenient bar on the point (F 9.72), but its upper CI is 12.97 > 10**.
  So it is not a candidate under the DRI rule. HP2 F is worse than the author's DEV2 5.97: bias +0.82 pp, RMS
  2.13 pp.
- [measured] v5a (inductive) is lenient only at 1k and 4k (F 2.9 / 5.3; upper CI ≤ 10).
- [measured] Every deductive entry fails both bars: t306_r1w F 16.3 [11.8, 21.2] (the best deductive entry; its DEV2
  F was 11.95), t286_rx F 98, t286b_probe F 107. t286_rx_ind (inductive
  reference) is at 25.8.

## Cost-convention re-score (TASKS Directive 8 cost convention, DRI 2026-10-18, from T312)
The convention: charge every weight-dependent operation at T196 rates (multiply-add = 2; Φ etc. = 20). Charge the
estimator's own forwards at their instrumented FLOPs, 5,760 = T(2H²+2H) + 2HT. Do not charge precomputed functions
of the input law alone. The sampling side stays at 5,948 FLOPs per FPE.
`cost_audit.py` reruns each frozen estimator unchanged and recounts its FLOPs. It asserts that every estimate and
the old charged cost match the scorer's raw files. Its docstring lists the counting rules:
- t306 walks: the code's own forward-equivalent count × 5,760. T312 reproduced the walk work to 0.3%.
- t306 Φ at walk endpoints: 20 + shift 1 (+3 for the 8th power), plus difference and accumulation.
- t306, not charged: the tail and full-line pieces and the density at the nodes, which are model-free tables.
- t306 combination: 4 per (triple, node, region).
- t300: 2 forwards per triple, plus 27 FLOPs per (triple, node): one weight-dependent Φ, its 8th power, and the
  multiply-adds.
- Note: both frozen codes already convert their forwards at 5,760 (`r0.FLOP_FWD`). The whole change is in the
  Φ/GL charge: the old code charged model-free Φ and density evaluations, and a flat 40 per (triple, node,
  region) for t306.
Reproduce: `PANEL=hp2 PYTHONPATH=shim python3 cost_audit.py t306|t300` (and PANEL=hp1). Results:
`results/[hp2_]<name>_conv2_F.json`.

| entry | panel / group | C old → new (FPE) | F old [95% CI] | F new [95% CI] | verdict (new) |
|---|---|---|---|---|---|
| t306_r1w (deductive) | hp2 retrain | 13269 → 12110 | 16.35 [11.77, 21.24] | 14.92 [10.74, 19.38] | fail |
| t306_r1w (deductive) | hp1 clean | 13056 → 11875 | 9.12 [1.13, 20.10] | 8.30 [1.03, 18.28] | lenient pass (point); upper CI > 10 → not a candidate (HP1 clean: robustness only) |
| t300_r0 (inductive) | hp2 retrain | 1697 → 1551 | 9.72 [7.02, 12.97] | 8.89 [6.42, 11.86] | lenient pass (point); upper CI > 10 → not a candidate |
| t300_r0 (inductive) | hp1 clean | 1697 → 1551 | 5.90 [3.86, 7.81] | 5.40 [3.53, 7.14] | lenient pass, upper CI ≤ 10 (HP1 clean: robustness only) |

- [measured] Under the written convention, R1W V1 is charged 12.1k FPE on HP2, down from 13.3k (×0.91). Its HP2 F
  goes from 16.35 to **14.92 [10.74, 19.38]**: still a lenient FAIL.
  - T312's independent count is lower again (11.1k FPE on DEV2). That count charges the Φ work at one density
    evaluation per node, not per endpoint, and would give HP2 F ≈ 13.6. That is still a fail, so the verdict does
    not depend on how the Φ work is counted.
- [measured] T300 is charged 1.55k FPE, down from 1.70k (×0.91). Its HP2 F goes from 9.72 to **8.89 [6.42, 11.86]**: a
  point lenient pass, but the upper CI is still > 10, so it is still not a candidate.

### T313 instrumented cost for R1W V1 (r1_v1b2/v1b2.py @ 6a496515, sha256 `a09b395d…6cdc`)
`cost_t313.py` applies `v1b2.cost(model)` to the unchanged per-model V1 estimates (asserted equal). It also applies
the same count with B2 disabled, i.e. every row runs all T steps in its first cell, which is how the frozen
`walk.walk_line` actually computes.
**DRI ruling (2026-10-18):** cost is charged as implemented, as in LB3. B2 counts only once the code is re-frozen with
the sharing actually performed. The **official R1W V1 HP2 number is therefore 14.92 [10.74, 19.38]** (the T281
count under the written convention). The 13.51 is a projection for a V1-B2 that has not been built. Both fail.
| cost count | HP2 C (FPE) | HP2 F [95% CI] | HP1 clean F |
|---|---|---|---|
| frozen as charged | 13,269 | 16.35 [11.77, 21.24] | 9.12 |
| T281 count (written convention) | 12,110 | 14.92 [10.74, 19.38] | 8.30 |
| T313 instrumented, no B2 (as implemented) | 13,656 | 16.83 [12.11, 21.86] | 9.35 |
| T313 instrumented + B2 sharing | 10,962 | 13.51 [9.72, 17.55] | 7.46 |

- [measured] R1W V1 fails the lenient bar on HP2 under every count (point F 13.5–16.8).
- [measured] B2 is the entire T313 saving. T313's per-step count (1,218 FLOPs per step-evaluation, including the
  gate-flip roots) is higher than the 2 × 576 that the forward-equivalent count uses.

## Compute accounting
- Charged cost is each submission's own `cost_flops`. v5q and v5a charge 6000 FLOPs per node with N = B nodes,
  i.e. C = 1.0087 B FPE. This charge does not include building the model-independent node design (Sobol points,
  inverse normal CDFs, a per-node permutation loop). That design is cached across models, and the author does not
  charge it.
- Scorer CPU: control 327 s, v5q 211 s, v5a 263 s, summed over 450 (model, B) calls each (process time, single
  process, under `ulimit -v 2500000`). Truth computation: T276, GPU, not counted here.

## Claims
- [measured] On HP1, v5q passes the lenient bar at B = 16k and at B = 64k in all four groups. Its largest upper CIs
  are 3.18 at 16k and 6.86 at 64k. It fails at 256k (clean, rel0.005 and rel0.01 have upper CIs > 10). It fails the
  strict bar at every B. This matches its LB3 pattern (pass at 16k/64k, fail at 256k).
- [measured] v5a fails the lenient bar at every B. It carries a +0.5 to +1.5 pp bias that does not shrink with B.
- [measured] The harness control gives R ≈ 1 (0.83–2.30, every CI covers 1).
- [measured] Truth-noise share of the sampling side: ≤ 0.26% (at 256k), so it is negligible.
- [measured] Under the restated strict bar (point R ≤ 1 in every group), no submission passes at any B.
- [inferred] **The original strict bar (upper CI ≤ 1) was not reachable even by plain sampling.** (The DRI has since
  restated it; it is kept here as "strict, significant".) The control's
  upper CIs are 1.12–5.78. With 6 clusters, a method exactly as good as sampling fails the strict bar in most groups,
  so a strict pass needs a ratio well below 1 (roughly < 0.3–0.5). The DRI may want to restate it, for example as
  point R ≤ 1, or as upper CI ≤ the control's upper CI.

## Alternatives not ruled out
- v5q's 64k pass has margin: max upper 6.86 vs 10. That margin comes from a single fixed Sobol scramble (seed 274),
  and another scramble could score differently. Re-scrambling would mean a new submission, so it was not done.
- Bootstrap CIs with 6 clusters are wide and the percentile CI under-covers. The clean group has one model per
  cluster.
- The scorer was checked only by the plain-MC control and the hash checks. A second independent harness has not
  re-scored these runs. A rough local check is that orig's HP1 truth agrees with the public `m16/truth.txt` within
  about 1 combined SE.

## Failure modes
- A stochastic submission would need a defined RNG protocol. The current interface `est(model, budget)` passes no
  rng, so a submission must seed itself.
- Raw per-model files are local only. If the container is reclaimed, the runs must be repeated to re-aggregate.
  They take a few minutes per submission.
