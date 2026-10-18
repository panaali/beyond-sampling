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

No loader shim was needed: the entry point takes `(Wi, Wh, Wo)` float64 exactly as the panel stores it. The
estimator files were not modified.

## Status
Done for v5q, v5a and the control. Standing service: further submissions are scored the same way.

## Results
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

**Bars (all four groups at one B; every run is cost-valid)**
| method | 16k lenient / strict | 64k lenient / strict | 256k lenient / strict |
|---|---|---|---|
| control (sampling) | pass / fail | pass / fail | pass / fail |
| v5q (inductive) | **pass** (max upper 3.18) / fail | **pass** (max upper 6.86) / fail | fail (3 groups > 10) / fail |
| v5a (inductive) | fail (3 groups) / fail | fail / fail | fail / fail |

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
- [inferred] **The strict bar, as worded (upper CI ≤ 1), is not reachable even by plain sampling.** The control's
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
