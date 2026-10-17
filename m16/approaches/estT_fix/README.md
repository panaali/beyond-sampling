# estT_fix — T131 (exact-mode mass bug) + T164 (re-score (8,4), over-split diagnosis)

**Question.** T131: why do estT exact-mode leaf masses sum to 1.000888 at (8,4)? T164: with a correct reference, how does
the frozen recipe108 score at (8,4), and do its splits change the answer? **Status:** done. **Method:** deductive (no
random inputs); fixed copy `estT.py` (original `understand_m43/estT.py` untouched); scripts named below. **Results:**
## T131: bug and fix (only `orthant`, d = 4; geometry/clipping/triangulation unchanged)
The defect is the 4-D Plackett quadrature, not the cell geometry (final3's guess was wrong). `diag1.py 0` (original code)
reproduces acc .9750473 / leaf sum 1.000888; on its 35098 error simplices the fixed rules ((12,2) if max|rho|<=.95 else
(48,3)) differ from a fine rule by **+1.113e-3 summed** (max 3.9e-6 per simplex, 2984 negative masses) = the whole bias.
Fixes in `orthant`: (1) log substitution 1-t = e^-v, v in [0,70], 128 GL nodes, Rt = C + (1-t)(I-C); (2) facet normals
from cofactors, asin(C_ij) and asin(partial corr) via atan2 (removes sqrt(eps) loss near |rho| = 1); (3) |det R| < 1e-6:
exact ray-space form mass = |det R|/(2 pi^2) * int_simplex |R^T u|^-4 du (10^3 collapsed Gauss; matches Plackett to 7e-7
relative at 1e-6..1e-3). After (1)+(2) seed 1 still lost 2.2e-9, all in 45 negative-mass simplices with det ~1e-11 (`diag2.py 1`); (3) removed it.

| zoo (8,4) seed | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| leaf sum − 1 (`leafsum.py s`) | 1.1e-11 | 8.6e-12 | 1.6e-11 | 1.2e-11 | 1.2e-11 |
| fixed exact acc | .97616081 | .98746001 | .97185215 | .97671350 | .98539792 |
| z vs 1e7 MC (`mc.py`, rng 2026) | −0.30 | +0.53 | −0.06 | +0.25 | −1.20 | zoo0 reproduces the Codex T131 target .9761608075 (not on this branch) to 10 digits. Noisy copies: |z| <= 0.51.
## T164: frozen recipe108 (eps .1, as final3/run.py) vs fixed exact — `rescore.py <name>`, `summary.py` -> `summary.json`
| model | err pp (fixed estT) | ops | MSE ratio | old final3 est: err pp / ratio | cells | regions ≥ |
|---|---|---|---|---|---|---|
| zoo0 | −0.00001 | 8.87e8 | 1e-6 | −0.091 / 10.7 | 6182 | 1131 |
| zoo1 | −0.00057 | 6.28e8 | 0.0025 | −0.086 / 13.3 | 4573 | 980 |
| zoo2 | −0.00725 | 8.63e8 | 0.25 | −0.091 / 8.9 | 6145 | 1062 |
| zoo3 | −0.00000 | 9.23e8 | 2e-9 | −0.079 / 8.6 | 6799 | 1309 |
| zoo4 | −0.00491 | 1.07e9 | 0.27 | −0.133 / 45.8 | 7679 | 1374 |
| zoo0 noisy 0-3 | −0.0001..+0.00005 | 8.4-9.3e8 | ≤ 3e-5 | −0.078..−0.083 / 4.4-6.4 | 5874-6032 | 1089-1145 |
Ratio = err^2 / (p(1-p)/N), N = ops / fpe_flops(8,4). Cells are identical to final3 (same geometry); ops are 4.4x
final3's because the fixed orthant costs more (mostly the 10^3-point thin rule).
**Over-split diagnosis** (DIAG hook, per resolved split: 64 Dirichlet points + rays on the minority side, true vs gate
frozen; pooled over the 9 models). Step t (0-based input index) 1 / 2 / 3: 3369 / 10940 / 29088 splits; with no class
change seen: **33% / 43% / 34%**; no correct/wrong change: 38% / 49% / 40%. Sampling can miss small changes, so these are upper
bounds. Over-splitting does not dominate (< 50% at every step), so per the brief no certificate tightening was tried.
## Claims
- [measured] The E45 (8,4) bias (−0.08..−0.13 pp, both recipe and "exact") was the d = 4 orthant quadrature; with it fixed,
  exact mode matches 1e7 MC (|z| <= 1.2, 9 models) and leaf masses sum to 1 within 2e-11 (5 seeds).
- [measured] The unchanged recipe108 geometry with the fixed masses beats sampling at (8,4) on all 9 models (ratio <= 0.27,
  7/9 below 0.003), even at 4.4x the ops. E45's "recipe does not scale to T = 4" at (8,4) is refuted (DRI to re-register).
- [measured] 57-67% of resolved splits per step change the class somewhere in the cell; cells/regions ≈ 5.5x.
- [inferred] The remaining cell excess over regions is mostly needed splits (cells are convex pieces of non-convex regions,
  plus answer changes in tiny slivers), not certificate slack.

**Compute.** Ops counted by `m43.Ops` (estimator only; DIAG and MC excluded). Exact runs 0.96-1.35e9 ops, ~1 min each;
recipe 0.63-1.07e9; MC 9 x 1e7 forward passes (diagnostic only). 1 core.
**Alternatives not ruled out.** Ops of the new orthant are not optimised (a cheaper thin rule could cut ops ~2-4x); the
1e-6 det threshold and rule sizes were chosen on zoo0 simplices only; over-split counts use samples, not exact tests;
(8,5) and M_16,10 projections in final3 used the buggy masses and should be re-run.
**Failure modes.** Simplices with |R^T u| near 0 (cones near a half-space) would break the thin rule's accuracy; none seen.
