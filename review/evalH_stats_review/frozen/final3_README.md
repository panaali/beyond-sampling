# final3: RECIPE112 + multi-step certificate (T108) along T at H=8, and the M_16,10 projection

**Question.** Does the frozen recipe (estT mode `recipe108`: margin split at every step with the multi-step Lipschitz
certificate |Wh|^k, |ΔWo|) still beat sampling at (8,4), (8,5), and is its cost growth along T compatible with M_16,10?

**Method.** Deductive (no random inputs to the estimator). `run.py` (recipe + MC truth 4e6 + region lower bound from
distinct gate patterns in 2e5 samples), `exact` mode of estT for comparison, `cells5.py` (T=5: estT has no 5-D cone
mass, so splitting is run with `mass()` stubbed -> cells and non-mass ops, a cost lower bound), `summary.py`.
Sampling MSE = p(1-p)/N, N = ops / `leaderboard.fpe_flops(H,T)`. Noisy copies: `leaderboard._noisy(.., 0.005, seed 7)`.

**Status.** Done (negative). Truth/ratio per model: `res_H8_T4.json`; exact mode: `exact_H8_T4.json`; all: `summary.json`.

## Exp 1: (8,4)  — `OMP_NUM_THREADS=1 python run.py 8 4 0,1,2,3,4 4`
| model | err pp | ops | MSE ratio | ratio if mass bug fixed* | cells | regions ≥ |
|---|---|---|---|---|---|---|
| zoo0 | -0.081 | 2.01e8 | 8.5 | 0.55 | 6182 | 1131 |
| zoo1 | -0.085 | 1.47e8 | 12.8 | 1.82 | 4573 | 980 |
| zoo2 | -0.097 | 1.96e8 | 10.2 | 0.67 | 6145 | 1062 |
| zoo3 | -0.076 | 2.10e8 | 8.0 | 0.48 | 6799 | 1309 |
| zoo4 | -0.137 | 2.48e8 | 48.5 | 4.21 | 7679 | 1374 |
| zoo0 noisy 0-3 | -0.067..-0.082 | 1.9-2.1e8 | 3.3-5.4 | – | ~5960 | ~1120 |

*(recipe − estT exact)²/sampling MSE. estT `exact` mode (2.1-3.1e8 ops, 8.4-11.3k leaves) is itself off by
−0.09..−0.18 pp vs MC (independent torch MC 1e7 on zoo0: 0.97617 vs numpy MC 0.97606, exact mode 0.97505).
**Root cause located:** the exact-mode leaf cells of zoo0 have total Gaussian mass 1.000888 (should be 1): leaves
overlap / clip pieces over-count by 8.9e-4 ≈ the bias; spread over many half-chambers (each +4e-5, `check108.py`).
4-D simplicial masses agree with MC (8 random cones, |Δ| ≤ 2.6 SE) and the 96 halves sum to 1.000000, so the defect
is in clipping/triangulation of thin non-simplicial cells, not in the Plackett orthant formula.

## Exp 2: (8,5) — `python cells5.py 8 5 0` (stopped after seed 0: answer clear)
| T (H=8) | recipe cells | regions ≥ (2e5 samples) | ops |
|---|---|---|---|
| 3 (evalH_8.json, mean 5 seeds) | 466 | 165 (exact 238 on zoo0) | 1.67e5 |
| 4 (mean 5 seeds) | 6276 | 1171 | 2.0e8 |
| 5 (zoo0) | 114155 | 3433 | ≥2.63e9 without masses (~15% of total at T=4) |
Per T step: cells ×13.5 then ×18; regions ×7 then ×3; ops ×1200 then ≥×13 (non-mass only; ×85 non-mass to non-mass).
Log-slope cells vs regions: 1.3 (T3→4), 2.6 (T4→5).

## Exp 3: M_16,10 — not run (projection)
At H=8 the cost already exceeds 1e9 at T=5 (≥2.6e9 without masses, and 5-D masses are unimplemented). With
≥×13 per extra step, T=10 at H=8 projects to ≥1e15 ops; H=16 only adds to it. Far above the ~1e9 gate.

## Claims
- [measured] At (8,4) the frozen recipe fails vs sampling on all 5 seeds and 4 noisy copies (ratio 3.3-48.5), with a
  systematic negative bias (−0.07..−0.14 pp), at ~80% of exact-enumeration cost.
- [measured] Most of that bias is shared with estT exact mode: leaf masses over-cover by 8.9e-4 (zoo0).
- [inferred] With that defect fixed, the T108 certificate's own error would give ratio 0.5-4.2 (passes 3/5 seeds):
  borderline, and it saves only ~20% vs exact enumeration.
- [measured] Recipe cells grow faster than linear regions along T (cells/regions 2.8 -> 5.4 -> 33); cost does NOT grow
  slower than regions.
- [inferred] The recipe cannot reach M_16,10 within 1e9 FLOPs; per-cell splitting along T is the wrong axis to scale.

## Failure modes
Thin-cell clipping/triangulation over-counts mass at H=8,T=4 (bug in understand_m43/estT.py; owner should test
mass(parent) vs mass(in)+mass(out) per clip). No 5-D cone mass. Region counts are sample lower bounds.
