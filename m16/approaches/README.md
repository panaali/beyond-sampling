# Approaches to ARC's challenge on M_16,10

↑ [Project README](../../README.md) · [STATUS](../../STATUS.md) · [INDEX (registry + ledger)](../../INDEX.md) ·
[FINDINGS (mechanism)](../FINDINGS.md) · [Leaderboard code](../leaderboard.py)

Each folder is one attempt, run by one agent, with a `README.md` in the template from [CLAUDE.md](../../CLAUDE.md).
The **ID** is the experiment's row in the [INDEX registry](../../INDEX.md#3-experiment-registry). "Deductive" means no
random inputs are fed to the model. MSE ratios are relative to plain sampling at equal compute (below 1 is better).
Only the leaderboard (LB2) numbers were run by the DRI; the rest are the agents' own.

## Deductive attempts (ARC's actual bar): all lose to sampling

| Folder | ID | Idea | Result |
|---|---|---|---|
| [rank_exact](rank_exact/README.md) | E10 | Freeze gates per rank-order class; integrate exactly | 8-13x worse; gates not fixed by rank order (H-M8) |
| [propagation](propagation/README.md) | E13 | Propagate Gaussian mixtures / cumulants through the RNN | Off by 10-69 pp; cumulant ceiling (H-E6) |
| [cone_enum](cone_enum/README.md) | E14 | Enumerate error cones; model-free cone masses | 56-91x worse; >=444k error cones (H-M7) |
| [reliability](reliability/README.md) | E17 | FORM/SORM per rank chamber | 20-50x worse; lowest deductive bias (+0.15 pp) |
| [branch_bound](branch_bound/README.md) | E19 | Certified probabilistic branch-and-bound | Certified nothing at 1e11 FLOPs |
| [observer](observer/README.md) | E21 | Signal-detection "observer" quadrature | ~10x worse |
| [sci_laplace](sci_laplace/README.md) | E26 | Laplace: asymptotics of tie bands | Oracle additive model misses by 4% (H-E8); fallback fails on seeds |
| [sci_poincare](sci_poincare/README.md) | E27 | Poincare: symbolic dynamics over records | Ceiling +0.07-0.75 pp, always positive (H-E10) |
| [sci_feynman](sci_feynman/README.md) | E24 | Feynman: exact partition + segment walks | Deterministic quadrature ~1-3x; not deductive in ARC's sense |
| [sci_tao](sci_tao/README.md) | E25 | Tao: exact structure + averaged pseudorandom part | 507x at T=3, collapses at T~5 (H-E11); ~1.5x at T=10 |

## Sampling-based and quadrature methods (baselines, not solutions)

| Folder | ID | Idea | Result (LB2 at 64k: orig / seeds / sigma 0.0025 / sigma 0.005) |
|---|---|---|---|
| [control_variate](control_variate/README.md) | E12 | **srb**: exact last step + near-tie strata | **0.42 / 0.45 / 0.50 / 0.61: best, CI < 1 everywhere** |
| [tie_lines](tie_lines/README.md) | E18 | "joint": gap strata + tie-swap antithetics | 0.34 / 0.72 / 0.51 / 0.53 |
| [boundary](boundary/README.md) | E11 | Boundary zones + exact 1-D band integrals | 0.61 / 0.68 / 0.52 / 0.68 |
| [tube](tube/README.md) | E15 | Deterministic tie-slicing quadrature | 0.14 / **1.23** / 0.46 / 0.75 (fails on seeds; bias floor) |
| [debiased_series](debiased_series/README.md) | E20 | Exact E[surrogate] + sampled residual | Loses to stratified below 128k |
| [great_circle](great_circle/README.md) | E22 | Exact integration around random great circles | 3-10x worse (H-E7) |
| [array_rqmc](array_rqmc/README.md) | E16 | Array-RQMC with an exact last step | ~1.0-1.6x |

## Diagnostics

| Folder | ID | Question | Answer |
|---|---|---|---|
| [bandwidth](bandwidth/README.md) | E23 | Is the tie band width compactly describable? | No: learnable but not compact; uncorrelated across seeds |

Earlier DRI-run estimators live directly in [`m16/`](../README.md): probe v2-v4 (`estimate.py`), exact lines
(`lineint.py`), RQMC (`qmc_test.py`), stratified sampling (`stratified.py`).
