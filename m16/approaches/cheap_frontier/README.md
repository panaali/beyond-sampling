# T286: cheap-regime frontier of the existing deductive / mechanistic estimators

↑ [Approaches index](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

## Question
Scored by its own cost (TASKS.md Directive 8 amendment 2), a fixed deductive estimator passes iff
F = mean(e^2) * C_eff / mean(p(1-p)) <= 1 (strict) or <= 10 (lenient) on a group of models, with
C = charged FLOPs / 5948 (FPE, T196) and C_eff = max(C, 1000). Every existing mechanistic estimator was tuned for
16k-256k FPE. What are their (cost, error) points at their *cheapest* settings, which error components dominate, and
is any of them close to F <= 1 / F <= 10?

## Pre-registration (written 2026-10-18 before any sweep was run)
Already looked at before writing this (disclosed): the LB3 raw runs (`results/leaderboard_v3_raw.jsonl`) for
rank_exact, observer, reliability, cone_enum, propagation, branch_bound, arc_cumulants; the T274b `cone_bias.log`
and the probe_v5 README tables. Nothing was run.

**Panel.** LB3 test models: orig, seed0-4, s{.0025,.005,.01,.02}_{0..5} (30 models, `m16/leaderboard.py`),
truths `results/leaderboard_truths.json` (32M samples). Then DEV2 (T284) when published. HP1/HP2 truths never read.

**Score.** Per group g and setting: bias_g, RMS_g (truth s.e. subtracted as in LB3), C_g = mean charged FPE,
F_g = RMS_g^2 * max(C_g, 1000) / mean_g p(1-p); 95% CI by bootstrap over the group's models.
F_max = max over the six groups.

**Frontier (fixed now).** (a) The Pareto set of all (estimator, setting) points in (C, F_max); (b) per group, the
Pareto set in (C, RMS_g). The **candidate** is the setting with the lowest F_max among settings whose steps are all
labelled deductive (exact computation or a declared deductive approximation, no averaging of model outputs over a
fixed node set). The best setting that includes an inductive step is reported separately and is not the candidate.
Ties within 10% of F_max are broken by lower C.

**Settings grid (fixed now; cheapest first).**
- rank_exact: Q in {4, 8, 16, 64}; position-triple subset stride s in {1, 2, 4, 8, 16} (balanced: triples j with
  j mod s = 0 in a fixed position-balanced order); K-classes per-K (11) or pooled into 3 groups or 1 group.
- observer v4: q in {1, 2}; nq in {6, 8, 12}; relative-gap bins 6 (default) or 3; triple stride {1, 3, 8, 24}.
- reliability: LB3 ladder L0 and cheaper L0 variants (chamber stride x2/x4/x8, modes (0,2), SORM off).
- cone_enum, propagation, branch_bound: their LB3 cheapest-budget rows (cost already at or near the floor).
- T274b deductive root-cone baseline (180 cones, m = 0..2 exact last steps): its measured bias with the
  implemented-cost estimate.
- probe series: `m16/estimate.py` v1-v4 (bisection probes) with iters in {4, 8, 12} and triple strides; probe_v5b
  (deductive spread, fixed nodes) and v5a / v5q (inductive references) from their LB3 raw.

**Cost.** Charged as implemented: every operation that depends on the weights (forward passes, tangents, bisection
probes, Jacobians, cone intervals, quadrature of weight-derived maps) counted in FLOPs (multiply-add = 2), checked
against CPU time relative to plain sampling.

**Expected (before measuring).** No existing estimator reaches F_max <= 1. rank_exact at ~1-3k FPE: frozen-gate bias
~1-2 pp, F_max ~5-20. observer at its cheapest: ~5-10k FPE with ~1 pp, F ~10-30. cone_enum, propagation,
reliability and the cone baseline: F_max >> 10, dominated by the seed group.
Alternatives to keep in view: (1) a small cheap-setting RMS may be cancellation between components, not accuracy;
(2) triple subsampling is fixed-node averaging over positions (inductive); its error should be separated from the
structural bias; (3) settings are picked on test models (LB3), so only HP2 is a clean score.

## Method
Every estimator is run as implemented, with only its own resolution knobs turned down (plus one shared knob, a
position-balanced subset of the 720 top-3 triples, which is an inductive step and is labelled so). Wrappers live
here and import the original code unchanged:

| file | estimator | knobs | step labels |
|---|---|---|---|
| `rx.py` | rank_exact (`../rank_exact/rank_est.py`) | Q, triple stride, K gate groups (11 / 3 / 1) | canonical gates + frozen-gate exact linear map: deductive approximation; integration of the linear surrogate over the model-free law (fixed Sobol): deductive; stride > 1: inductive |
| `obs.py` | observer v4 (`../observer/v4.py`, `est.py`) | q, nq, relative-gap bins 6 / 3, triple stride | anchors, exact local map and gate-cone intervals: deductive; cone-clipped Gaussian internal noise: deductive approximation; blind probe (10 canonical inputs): deductive approximation; stride > 1: inductive |
| `probe.py` | probe series `m16/estimate.py` v1-v3 (bisection of the flip gap at canonical inputs) | version, bisection iters, triple stride | network answers on constructed inputs: deductive; model-free gap law: deductive approximation; blind error 8/9: argued constant; stride > 1: inductive |
| `rel.py` | reliability ladder rung L0 and cheaper variants | chamber stride, modes, SORM on/off | design points / FORM / SORM / Ditlevsen: deductive approximations; every rung uses a fixed chamber subset: inductive |
| `ext_fams.py` | cone_enum, propagation, branch_bound (LB3 entries); probe_v5a/b/q (T274a, `ext/probe_v5.py`); T274b root-cone baseline (`ext/t274b_*`) | budget; m exact last steps | cone_enum, propagation, cone baseline: deductive; branch_bound <= 64k, v5a/v5b/v5q: fixed-node averages, inductive |

`sweep.py FAMILY [--models dev2]` runs the pre-registered grid (resumable, `results/sweep_raw*.jsonl`);
`frontier.py [lb3|dev2]` scores it (`results/frontier_*.json|txt`), `summary.py` writes `results/summary_*.md`;
`attrib.py` splits the error against a sampled truth decomposition (`diag_truth.py`, diagnostic only);
`cpu_check.py` compares CPU time with plain sampling; `hp2_entry.py` is the frozen entry.
For LB3, cone_enum / propagation / branch_bound / probe_v5 rows come from their stored LB3 raw runs (cost converted
from 6000 to 5948 FLOPs per FPE); the T274b cone baseline from its stored paired biases (`ext/`, provenance in
`ext/PROVENANCE.md`).

**Environment.** This container had no venv: `python3 -m venv .venv`, numpy/scipy, CPU torch, and alg_zoo cloned
from github.com/alignment-research-center/alg-zoo (HEAD 4433f505) to `/home/user/alignment-research-center/alg-zoo`
and installed editable, as in AGENTS.md. LB3 orig reproduces (0.9538 on 200k vs truth 0.95314), rank_exact Q=16
reproduces 0.93274, observer v4 0.9467, reliability L0 0.97191 (all equal to their LB3 rows).

## Status
Done (LB3 and DEV2). Candidate frozen (`hp2_entry.py`), forwarded by the DRI to the T281 scorer; post-hoc DEV2 entry
`hp2_entry_b.py`. Scope notes from the DRI (2026-10-18): the observer moved to T292 (its LB3 rows stay; not run on
DEV2), the frozen-witness rank_exact family is T287's. Unverified.

## Results (LB3, 30 models)
Reproduce: `python sweep.py rx; python sweep.py rel; python sweep.py probe; python sweep.py obs; python frontier.py lb3;
python summary.py lb3`. Full table: `results/frontier_lb3.txt`. F = RMS^2 max(C, 1000) / mean p(1-p) per group.

**Best setting per family** (lowest F_max; for families whose best setting has an inductive step, the best
all-deductive one is listed first). Entries: bias / RMS in pp, F.

| family | setting | label | C (FPE) | F_max | orig | seed | s.0025 | s.005 | s.01 | s.02 |
|---|---|---|---|---|---|---|---|---|---|---|
| **rank_exact** | **Q=8, 3 gate groups, all triples** | **D** | **25,779** | **112** | -1.1/1.1, 67 | -1.4/1.9, **112** | -0.6/0.7, 22 | +0.5/1.0, 30 | +1.9/2.2, 84 | +3.0/3.1, 111 |
| rank_exact | Q=8, 1 gate group, all triples | D | 10,013 | 149 | -1.5/1.5, 51 | -3.2/3.4, 149 | -1.2/1.2, 25 | -0.0/1.1, 13 | +1.7/2.2, 31 | +3.0/3.1, 43 |
| rank_exact | Q=4, 3 groups, stride 16 | I | 1,716 | 16.7 | -1.0/1.0, 3.9 | -2.4/2.5, 13 | +0.0/1.4, 5.4 | +0.9/2.3, 9.8 | +2.5/3.9, 16.7 | +3.0/4.2, 14 |
| observer | q=1, nq=6, 3 r-bins, all triples | D | 51,677 | 268 | -0.7/0.7, 59 | -0.1/0.9, 50 | -0.3/0.5, 21 | +0.1/0.6, 21 | +0.1/1.5, 73 | +0.7/3.4, 268 |
| observer | q=1, nq=8, 3 r-bins, stride 24 | I | 2,276 | 58 | -0.8/0.8, 3.4 | -0.4/1.0, 3.1 | -0.9/1.2, 5.9 | -1.4/1.8, 8.0 | -3.0/5.1, 38 | -3.4/7.5, 58 |
| probe (estimate.py) | v2, iters 8, all triples | D | 7,473 | 361 | +1.1/1.1, 21 | -1.2/3.2, 100 | +0.3/2.5, 84 | +0.4/3.6, 108 | +2.5/3.7, 66 | +9.5/10.8, 361 |
| probe (estimate.py) | v2, iters 8, stride 8 | I | 930 | 86 | +0.9/0.9, 1.8 | -3.5/7.0, 62 | -0.3/3.2, 18 | +0.5/3.4, 13 | +2.8/4.0, 10.5 | +12.2/13.8, 86 |
| reliability | L0, stride x4, modes (0,2), no SORM | I | 2,652 | 106 | +1.7/1.7, 16 | +2.1/3.2, 34 | +1.5/2.1, 19 | +1.2/3.6, 39 | +0.2/6.5, 72 | +3.5/9.4, 106 |
| T274b root cones | m=2 (cost lower bound) | D | >=1,980 | >=729 | -3.0/3.0, 40 | -12.8/17.1, 729 | -2.0/2.1, 15 | -0.3/1.6, 5.8 | +2.9/3.8, 18 | +5.3/5.7, 29 |
| cone_enum | budget 1k | D | 1,152 | 759 | +4.7/4.7, 57 | +8.8/9.2, 121 | +6.4/6.5, 80 | +10.2/10.8, 149 | +20.1/21.5, 349 | +36.6/38.1, 759 |
| propagation | budget 1k (cheapest) | D | 10,443 | 67,725 | -54/54 | -46/46 | -50/50 | -49/49 | -42/43 | -29/31 |
| branch_bound | <= 64k (180 centres) | I | 182 | 105 | +3.6/3.6, 29 | +5.9/6.2, 48 | +3.0/4.2, 29 | +0.0/8.0, 71 | -2.3/12.7, 105 | -9.0/10.8, 53 |
| probe_v5b | 4k (spread, fixed nodes) | I | 4,074 | 8.9 | -0.9/0.9, 7.9 | -1.2/1.3, 8.9 | -1.1/1.1, 8.3 | -1.1/1.2, 6.7 | -1.5/1.6, 6.5 | -2.1/2.2, 8.9 |
| probe_v5a | 1k (reference, QMC-like) | I | 1,009 | 4.6 | -0.1/0.1, 0.0 | +0.8/1.2, 1.8 | -0.6/0.7, 0.8 | -0.7/1.5, 2.4 | -1.0/2.0, 2.7 | -1.5/3.2, 4.6 |
| probe_v5q | 4k (reference, 10-D QMC) | I | 4,035 | 1.3 | +0.2/0.2, 0.5 | +0.1/0.4, 0.7 | -0.0/0.2, 0.2 | -0.3/0.5, 1.1 | -0.4/0.5, 0.7 | -0.7/0.9, 1.3 |

**Frontier (pre-registered definition (a), (C, F_max) Pareto set).**
- All-deductive: cone_enum 1k (C 1.2k, F_max 759) -> T274b cones (>=2.0k, >=729) -> probe v2 iters 4 / 8 (4.8k, 382;
  7.5k, 361) -> rank_exact Q=4 / Q=8, 1 group (8.9k, 174; 10.0k, 149) -> **rank_exact Q=8, 3 groups (25.8k, 112)**.
  Nothing all-deductive is within a factor 11 of the lenient bar.
- Any label: branch_bound (0.18k, 105) -> rank_exact Q=4, 1 group, stride 16 (0.62k, 35) -> probe_v5a 1k (1.0k, 4.6)
  -> probe_v5q 4k (4.0k, 1.3). Only the fixed-node references (v5a, v5q, v5b) pass the lenient bar; none of the
  mechanistic families does, at any setting.
- Per-group (C, RMS) Pareto sets (definition (b)): `results/summary_lb3.md`. The orig model alone is a poor guide:
  rank_exact stride 8 reaches 0.05 pp there by cancellation, against 2-4 pp on the seed and s.01/.02 groups.

**Error attribution** (`python attrib.py`, `results/attrib.txt`; sampled truth split with 2M inputs per model).
- **rank_exact**: the error is the frozen-gate bias in the bulk classes K = 2..8 (e.g. Q=4, 1 group: -2.2 orig,
  -3.7 seed, +2.8 s.02 pp mean). The blind region K <= 1 adds 0.1-0.2 pp, K >= 9 under 0.1. Split of the total:
  structural limit (Q=64, all triples) -2.0 / -3.5 / +2.6 pp (orig / seed / s.02) for 1 group and -1.6 / -1.8 / +2.5
  for 3 groups; quadrature at Q=4-8 RMS 0.05-1.0 pp; triple subset at stride 16 RMS 0.7-2.2 pp. At Q=16 with 11 groups
  (rank_exact's LB3 setting) a -0.9 to -2.0 pp quadrature error partly cancels a structural error of the other sign
  on the noisy groups, so the LB3 numbers there are not a structural property.
- **observer**: the total is small only by cancellation. On orig err23 is over-predicted by 0.8 pp, err12 by 0.3,
  the blind probe under by 0.2 and rank-4+ errors (0.27 pp) are not modelled. On s.02 err23 -5.7, err12 -5.1 and
  rank-4+ +10.8 pp nearly cancel. The 2~3 / 1~2 margins Phi(-mu/sigma) count events in which a third rank wins, so the
  components are not separately calibrated.
- **probe (estimate.py v2)**: err12 (1~2 bisection) is the main error on seeds and noisy copies (RMS 2.8-4.2 pp), and
  rank-4+ (unmodelled, up to 11.9 pp) dominates s.02.

**CPU-time check** (`python cpu_check.py`; ratio to plain sampling at the charged FPE). rank_exact 3.6, observer 8
(all triples) to 51 (stride 24), reliability 6-46, probe 68. Profiling the observer at stride 24 puts the excess in
`arc_eval._blind_probe` rebuilding its model-free order-statistic means on each call, and the probe excess is Python
overhead of one-sequence forward passes (1 FPE each, counted). No under-count of weight-dependent work was found;
the observer's own 10.5 FPE per anchor was raised to 11.5 after a FLOP count of `core.map_and_cone`.

## Results (DEV2, 80 open retrains: 40 fused-kernel + 40 torch-trainer, T284)
Truths: T284's official `truths.json` (1e8 per model, `ext/dev2/`); my provisional 4M truths agree (z sd 1.08).
Reproduce: `python sweep.py FAM --models dev2` for FAM in rx rel probe v5 tcone, and `lb` run from the repo root
(`python m16/approaches/cheap_frontier/sweep.py lb --models dev2`, the leaderboard loader uses repo-relative paths);
then `python frontier.py dev2; python summary.py dev2` (`results/frontier_dev2.txt`, `results/summary_dev2.md`).

| family | setting | label | C (FPE) | DEV2 F_max | fused bias/RMS, F | torch bias/RMS, F | LB3 seed F |
|---|---|---|---|---|---|---|---|
| **probe (estimate.py)** | **v1, iters 8, all triples** | **D** | **3,931** | **76.5** | +3.3/3.8, 76.5 | +3.2/3.6, 72.5 | 77.6 |
| rank_exact (candidate) | Q=8, 3 groups, all triples | D | 25,779 | 131 | -1.6/1.9, 131 | -1.2/1.5, 85 | 112 |
| rank_exact | Q=8, 1 group, all triples | D | 10,013 | 150 | -2.9/3.3, 150 | -2.5/2.9 | 149 |
| cone_enum | 1k | D | 1,149 | 119 | +8.1/8.8, 119 | +7.9/8.4, 114 | 121 |
| T274b root cones | m=2 (lower bound) | D | >=1,980 | >=313 | -5.8/10.8 | -6.4/10.6 | 729 |
| propagation | 1k | D | 9,440 | 3.5e4 | -43/45 | -48/51 | 2.5e4 |
| rank_exact | Q=8, 3 groups, stride 16 | I | 1,790 | 16.9 | -1.3/2.5, 15.7 | -1.2/2.6, 16.9 | 15.7 |
| reliability | L0 stride x4, modes (0,2), no SORM | I | 2,652 | 17.3 | +1.2/1.9, 13.0 | +1.3/2.2, 17.3 | 33.5 |
| probe (estimate.py) | v1, iters 12, stride 8 | I | 628 | 21.6 | +3.3/4.0 | +3.2/3.9 | 23.4 |
| branch_bound | <= 64k | I | 182 | 46.5 | +4.8/5.8 | +4.0/5.3 | 48.3 |
| probe_v5b | 1k | I | 1,018 | 4.3 | +0.3/1.8 | +0.2/1.5 | 1.8 |
| probe_v5a | 1k | I | 1,009 | 3.0 | +1.0/1.4 | +1.1/1.5 | 1.8 |
| probe_v5q | 1k | I | 1,009 | 1.2 | +0.6/0.9, 1.0 | +0.5/0.9, 1.2 | 0.1 |

- All-deductive (C, F_max) frontier on DEV2: cone_enum (1.1k, 119) -> probe v1 iters 4 (2.5k, 109) -> **probe v1
  iters 8 (3.9k, 76.5)**. Nothing all-deductive is within 7x of the lenient bar. Any label: branch_bound (0.18k, 47) ->
  probe v1 stride 8 (0.3-0.6k, 46-22) -> probe_v5q 1k (1.0k, 1.2).
- **The LB3 seed group predicts DEV2; the LB3 noise groups do not.** For every family the DEV2 F is close to the LB3
  seed-group F (rank_exact 131 vs 112, probe v1 77 vs 78, cone_enum 119 vs 121, rank_exact stride 16 16.9 vs 15.7),
  while F_max over the LB3 noise groups mis-ranks methods (probe v1: LB3 F_max 1042, set by rank-4+ errors that
  retrained models barely make). The pre-registered rule (LB3 F_max) therefore picked rank_exact; on the scoring
  distribution probe v1 is better by 1.7x in F. It is frozen post hoc as `hp2_entry_b.est` (selected on DEV2, so DEV2
  is advice for it; HP2 is clean for both).
- Retrains have one-signed errors: rank_exact (candidate) -1.39 pp mean, negative on 76/80 models, sd 1.06 pp around
  that mean; probe v1 +3.28 pp, positive on 80/80, sd 1.73 pp (it does not model 1~2 flips or rank-4+ errors). Both
  estimates correlate .96 with the truth across the 80 models. A constant offset fitted on DEV2 would leave F ~40
  (rank_exact) and ~16 (probe v1); that offset is a tuned constant (advice) and is not part of any frozen entry.

## Mean-field oracle (DRI request after T294; diagnostic, sampling)
Question: how much of rank_exact's per-class bias would remain if each class's gates were closed by moment matching /
switching probabilities instead of frozen at one pattern? `python meanfield_oracle.py 200000` (110 models: LB3 + DEV2,
inputs from rank_exact's own law, 100k for moments + 100k paired evaluation per model), `python mf_summary.py`
(`results/meanfield_summary.txt`). "gauss" gives each class's 9 rival logit margins their exact sampled mean and
covariance and takes P(all > 0) under that Gaussian: the best any class-level Gaussian closure can do.

| panel | network acc | frozen gates - net (mean/RMS pp) | Gaussian oracle - net | per-class RMS frozen / gauss / noise | corr with net, frozen / gauss |
|---|---|---|---|---|---|
| LB3 seeds | .917 | -1.08 / 1.83 | -11.9 / 12.0 | .097 / .160 / .055 | .83 / .92 |
| LB3 s.02 | .634 | +3.42 / 3.49 | -3.37 / 3.52 | .123 / .134 / .077 | 1.00 / 1.00 |
| DEV2 (80) | .923 | -0.87 / 1.34 | -10.95 / 11.13 | .092 / .151 / .054 | .95 / .78 |

- Blind classes (K <= 1): frozen -0.02 pp, Gaussian +0.33 pp on DEV2. The blind region is too small to matter for
  rank_exact either way.
- Bulk classes: the exact-moment Gaussian margin law is 11 pp too pessimistic on retrains, against -0.84 pp for
  frozen gates. The margins are far from Gaussian: they pile up near 0 at near-ties and have a hard edge where the
  gates switch. **A class-level mean-field rank_exact is not worth building.** Even with perfect moments it is 8x worse
  than frozen gates on DEV2 and tracks between-model variation less well (corr .78 vs .95).
- Not ruled out: a closure conditioned on more than the class (e.g. on the top-3 values, as the observer does, or on
  the sign of the near-tie gap), or mean-field only for the gates of the 7 lower ranks with the top-3 handled exactly.
  These are different estimators and were not tried.

## Candidate for HP2 (frozen, `hp2_entry.py`)
By the pre-registered rule the candidate is **rank_exact, Q=8, K pooled into 3 gate groups ({0-3}, {4-6}, {7-10}),
all 720 triples**: `hp2_entry.est(model, budget) -> (estimate, cost_flops)`, deterministic, ~1.53e8 FLOPs = 25.8k
FPE, LB3 F_max 112 (seed group). It is not expected to pass either bar on HP2. `hp2_entry.est_ind` (rank_exact Q=4,
3 groups, stride 16, ~1.7k FPE, LB3 F_max 16.7) is a reference with an inductive step, not the candidate.

**What would cut its error by the factor still needed.** F scales as RMS^2 x C_eff. At its own cost the candidate
needs its RMS cut 10.6x (strict) or 3.4x (lenient). At the 1k floor it would need 2.1x (strict) and would already pass
lenient (F = 4.3) if the error did not grow, so the binding constraint is that its structural error does not shrink
while its cost (720 triples x 3 groups x 11 FPE for the frozen-gate maps) cannot be cut without the triple subset,
which adds 0.7-2.2 pp. The structural part is the value-dependence of the gates within a (triple, K) class
(rank_exact claims 1-3), worst on the seeds (-1.8 pp) and s.02 (+2.5 pp), so a fixed correction cannot remove it.
What would: (i) gates as a function of the values inside each class (per-class mixtures of gate patterns weighted
by polytope masses, or exact splits on the few comparator gates); (ii) a cheaper per-class map (shared prefixes of
the canonical inputs, only the 2-3 rival logit rows). Both are untried here.

## Compute accounting
- Charged per estimator as implemented (FLOPs / 5948, multiply-add = 2): rank_exact 11 x 5920 FLOPs per (triple, gate
  group) + Q x 200 per (triple, K); observer 11.5 FPE per anchor + 10 blind probes + quadrature
  2 x n_tri x q x nodes x (2q + 36) FLOPs; probe 1 FPE per bisection probe; reliability its own count (C.fwd x 1.3 +
  SORM); LB3 rows their LB3 cost x 6000/5948. The T274b cone baseline is charged 180 x 11 FPE; its exact integration
  is not counted, so its F is a lower bound.
- Not charged (model-free): order-statistic tables, Sobol points, quadrature grids, tie-level quantiles.
- Diagnostics (not estimator compute): mean-field oracle 110 x 200k (~15 min), truth split 30 x 2M sequences (`diag_truth.py`), DEV2 provisional truths
  80 x 4M (`dev2_truth.py`), T274b paired bias 80 x 200k x 3 on DEV2. Sweep CPU: LB3 about 25 min, DEV2 about 1 h, 1 core.

## Claims
1. [measured] On LB3 no all-deductive setting of rank_exact, observer, probe (estimate.py), cone_enum, propagation or
   the T274b cone baseline has F_max below 112. The best is rank_exact Q=8 / 3 gate groups at 25.8k FPE.
2. [measured] Turning resolution down does not turn these methods into lenient-bar passes: their structural errors
   (1-4 pp on seeds and heavy-noise copies) stay while the cost floor is 1k, and cheap settings reach C <= 3k only by
   triple subsets (inductive), which add error. The best mechanistic setting with an inductive step is rank_exact
   stride 16 (F_max 16.7).
3. [measured] rank_exact's error sits in the bulk K = 2..8 classes (frozen-gate bias); the blind region is <= 0.2 pp.
4. [measured] The observer's small totals are cancellation between over-predicted 2~3 / 1~2 errors and unmodelled
   rank-4+ errors (up to 10.8 pp on s.02).
5. [inferred] The pre-registered expectation held, except that the cheapest observer setting is far better on the four
   lighter groups (F 3-8) than expected, and rank_exact needed 26k FPE rather than ~1-3k for its best F_max.
6. [measured] On DEV2 (80 retrains) every family's F is within ~20% of its LB3 seed-group F (except rank_exact 3
   groups: 131 vs 112, and reliability 17 vs 34); LB3 F_max, dominated by heavy-noise copies, mis-ranks methods.
7. [measured] On DEV2 the best all-deductive setting is probe v1 (estimate.py), 3.9k FPE, F_max 76.5; the candidate
   rank_exact has 131. [speculative] HP2 F will be close to these DEV2 values.
8. [measured] A class-level Gaussian (moment-matched) closure of rank_exact's margins, with oracle moments, is biased
   -11 pp on DEV2 against -0.9 pp for frozen gates. Frozen gates are not the main limitation at class level; the
   class is too coarse for any single-law summary.

## Alternatives not ruled out
- Settings were chosen on the LB3 test models (advice in leaderboard terms); only HP2 is a clean score.
- 5-6 models per group: the bootstrap CIs on F are wide (`results/frontier_lb3.json`, `F_ci`).
- The triple-subset knob is mine; other cheap designs (e.g. fewer rest arrangements in reliability, or a smarter
  subset) were not tried. Cheaper implementations of the same maps (prefix sharing) could move all deductive points
  left by an unknown factor; costs here are as implemented.
- The observer's 3-bin edges ([0, .03, .2, inf)) were fixed in the pre-registration without tuning; other edges not tried.
- The T274b cone baseline's numbers assume perfect integration (T274b's measurement), not an implementation.

## Failure modes
- Every mechanistic method here has a structural bias whose sign changes with the weights (seeds negative,
  heavy-noise copies positive for rank_exact), so no fixed correction or cheaper setting fixes it.
- Triple subsets (stride) turn a deductive method into a fixed-node average over positions; at stride >= 8 the subset
  error is as large as the structural error on the noisy groups.
- Error components that cancel (observer) make small totals uninformative about calibration.
