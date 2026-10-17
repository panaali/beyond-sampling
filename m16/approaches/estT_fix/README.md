# estT_fix — T131 (mass bug), T164 (re-score (8,4)), T185 ((8,5) with 5-D masses)
**Question.** T131: why do (8,4) exact-mode leaf masses sum to 1.000888? T164: how does the frozen recipe108 score at
(8,4) against a correct reference, and do its splits change the answer? T185: same at (8,5). **Status:** done. **Method:** deductive;
fixed copy `estT.py` (original `understand_m43/estT.py` untouched); scripts named below with **Results**.
## T131: the defect is the d = 4 Plackett quadrature in `orthant`, not the geometry (final3's guess was wrong)
`diag1.py 0` (original code): on zoo0's 35098 error simplices the fixed rules ((12,2) / (48,3)) are off by **+1.113e-3
summed** (max 3.9e-6 each, 2984 negative masses) = the whole bias. Fix: (1) log nodes 1-t = e^-v, v in [0,70], 128 GL;
(2) cofactor facet normals, atan2 angles/partial corr (no sqrt(eps) loss near |rho| = 1); (3) |det R| < 1e-6: exact form
|det R|/(2 pi^2) int_simplex |R^T u|^-4 du (10^3 rule) — it removed seed 1's last 2.2e-9 (45 negative simplices, `diag2.py 1`).
Seeds 0-4 (`leafsum.py s`, `mc.py`): leaf sum − 1 = 1.1e-11 / 8.6e-12 / 1.6e-11 / 1.2e-11 / 1.2e-11; exact acc .97616081 /
.98746001 / .97185215 / .97671350 / .98539792, z vs 1e7 MC −0.30 / +0.53 / −0.06 / +0.25 / −1.20 (noisy |z| <= 0.51).
zoo0 equals the Codex T131 target .9761608075 (not on this branch).
## T164: frozen recipe108 (eps .1, as final3/run.py) vs fixed exact — `rescore.py <name>`, `summary.py` -> `summary.json`
| model | err pp (fixed estT) | ops | MSE ratio | old final3 est: err pp / ratio | cells | regions ≥ |
|---|---|---|---|---|---|---|
| zoo0 | −0.00001 | 8.87e8 | 1e-6 | −0.091 / 10.7 | 6182 | 1131 |
| zoo1 | −0.00057 | 6.28e8 | 0.0025 | −0.086 / 13.3 | 4573 | 980 |
| zoo2 | −0.00725 | 8.63e8 | 0.25 | −0.091 / 8.9 | 6145 | 1062 |
| zoo3 | −0.00000 | 9.23e8 | 2e-9 | −0.079 / 8.6 | 6799 | 1309 |
| zoo4 | −0.00491 | 1.07e9 | 0.27 | −0.133 / 45.8 | 7679 | 1374 |
| zoo0 noisy 0-3 | −0.0001..+0.00005 | 8.4-9.3e8 | ≤ 3e-5 | −0.078..−0.083 / 4.4-6.4 | 5874-6032 | 1089-1145 |
Ratio = err^2/(p(1-p)/N), N = ops/fpe_flops. Cells as final3; ops 4.4x final3's (costlier fixed orthant).
**Over-split** (DIAG hook: per resolved split, 64 Dirichlet points + rays of the minority side, true vs gate frozen; 9 models
pooled): steps 1/2/3 have 3369/10940/29088 splits, with no class change seen in **33%/43%/34%** (no correct/wrong change
38%/49%/40%); upper bounds (sampling). Not dominant (< 50%), so no certificate tightening was tried.
## T185: (8,5) — `val5.py`, `run5.py <name> recipe108|exact [leafsum]`, `mc5.py` -> `run5/*.json`, `val5.json`
5-D orthant `_orth5`: Plackett reduces d = 5 to a 1-D integral with a closed-form conditional 3-D orthant (conditional
corr of (k,l) given (i,j) only, from cofactors of the 4x4 block {i,j,k,l}; a first version used the full 5x5 inverse =
given (i,j,m) and was wrong, z up to 4300); same log nodes; |det| < 1e-6: 3|det R|/(8 pi^2) int |R^T u|^-5 du (6^4 rule).
Validation: 6 random simplices and 3 non-simplicial halves vs 1e7 MC |z| <= 2.2; 480 halves sum to 1 - 7.5e-14; thin
rule vs Plackett agree to <= 3e-8 abs (mass 3e-6), 1e-15 at smaller det; recipe leaf cells sum to 1 + 2.0e-9 (4.9M simplices).
| zoo0 (8,5) | acc | ops | cells | regions ≥ |
|---|---|---|---|---|
| exact | .8358057 (1e7 MC .8357964, z +0.08) | 3.12e11 | 203685 | 3433 |
| recipe108 | .8358035 (err −0.00023 pp, MSE ratio 0.0097) | 2.18e11 | 114155 | |
**Stopped after zoo0** (brief: stop if > 1e11 ops/model): 2.2e11 ops, 61 min. Recipe ops (fixed masses) along T at H = 8:
1.7e5 / 8.7e8 / 2.2e11 for T = 3/4/5 (x5200, x250). **M_16,10 projection:** even at the smallest observed cell growth
(x13.5/step) and constant 1.9e6 ops/cell, T = 10 at H = 8 needs >= 1e17 ops (the ops trend gives ~1e23); H = 16 adds more.
## Claims
- [measured] The E45 (8,4) bias (−0.08..−0.13 pp, both recipe and "exact") was the d = 4 orthant quadrature; with it fixed,
  exact mode matches 1e7 MC (|z| <= 1.2, 9 models) and leaf masses sum to 1 within 2e-11 (5 seeds).
- [measured] The unchanged recipe108 geometry with the fixed masses beats sampling at (8,4) on all 9 models (ratio <= 0.27,
  7/9 below 0.003), even at 4.4x the ops. E45's "recipe does not scale to T = 4" at (8,4) is refuted (DRI to re-register).
- [measured] 57-67% of resolved splits per step change the class somewhere; cells/regions ≈ 5.5x [inferred: mostly needed].
- [measured] (8,5) zoo0: recipe accurate (ratio 0.0097, 1 model) but 2.2e11 ops = 70% of exact enumeration.
- [inferred] Per-cell recipes stay accurate along T but cost grows >= x13/step; M_16,10 is out of reach (>= 1e17).
**Compute.** Ops counted by `m43.Ops` (estimator only; DIAG and MC excluded). Exact runs 0.96-1.35e9 ops, ~1 min each;
recipe 0.63-1.07e9; (8,5) above; MC 10 x 1e7 forward passes (diagnostic only).
**Alternatives not ruled out.** Orthant ops not optimised (could cut ~2-4x, not the trend); thresholds/rule sizes tuned on
zoo0 simplices; over-split counts sampled; (8,5) only zoo0 (no other seeds / noisy copies).
**Failure modes.** Simplices with |R^T u| near 0 (cones near a half-space) would break the thin rule's accuracy; none seen.
