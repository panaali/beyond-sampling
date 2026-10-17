# Array-RQMC for M_16,10 (partial: original model only)

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [STATUS](../../../STATUS.md)

Files: `core.py` (chain, exact last-step integration, sort keys, estimators), `check_cmc.py` (checks the
last step against brute force), `experiment.py` (RMSE vs n over independent randomisations), `analyze.py`
(cost model and gains), `diag_var.py` (where the variance lives), `results/` (logs, cached truths for
noisy_models(6, .0025/.005) at 4M samples, V.npz).

## Method
- The chain state is (h_t, running top-2 values and positions, plus m3 for the sort keys only).
- **Conditional MC at the last step.** Given h_9, the logits are piecewise linear in x_9 (16 gate
  breakpoints) and the target changes only at m2 and m1. P(correct | x_0..x_8) is then a sum of Gaussian
  masses over at most 19 intervals. Checked against 200k-sample brute force on 20 states (all within 2.5 sd)
  and unbiased overall. This halves the per-chain variance: 0.0215 vs p(1-p) = 0.0445.
- **Array-RQMC** (L'Ecuyer, Lécot and Tuffin, 2008). n = 2^k chains. At steps 1-8 the chains are sorted by
  a key and given x_t from a fresh scrambled Sobol set, matched by rank. Keys tried: m1; the relative top-2
  gap; a linear value function fitted on the ORIGINAL model only; 2-D batch sorts on (m1, m2), (m1, V) and
  (m2, top-3 tie); and the min top-3 tie gap. The baselines are classical RQMC (9-D scrambled Sobol plus the
  exact last step) and MC plus the exact last step.

## Results (original model, 24 randomisations each; n*Var, lower is better; plain MC = 0.0445)
| method | n=1024 | 4096 | 16384 |
|---|---|---|---|
| MC + exact last step | 0.034 | 0.025 | 0.018 |
| RQMC(9-D) + exact last step | 0.016 | 0.011 | 0.020 |
| array, m1 | 0.022 | 0.014 | 0.019 |
| array, top-2 gap | 0.018 | 0.021 | 0.013 |
| array, value function | 0.020 | 0.014 | 0.026 |
| array 2-D (m1, m2) | 0.015 | 0.023 | 0.012 |
| array 2-D (m1, V) | 0.019 | - | - |

- **Convergence rate:** n*Var stays flat for every variant, so RMSE goes as about n^{-1/2}. This is no
  better than MC. The noise in a variance estimated from 24 randomisations is about ±30%.
- **Per-chain gain over plain MC:** about 2-3x, and it comes almost entirely from the exact last step. The
  array sorting adds at most about 1.0-1.5x, which is within noise. Cost per chain is about 1.9 forward
  passes (last step charged as 1.0; sorting adds 0.1-0.2). The compute-adjusted MSE gain over sampling is
  therefore about 1.0-1.6x. That is no better than stratified.py (1.2-2.8x by model-eval count).
- **Why it fails:** 94% of Var(Y) comes from the 7% of inputs with Y < 0.9. These are near-ties among the
  top 3 of x_0..x_8 (2nd/3rd: 45%; 1st/2nd: 26%) and the blind region (24%). Even binned (m1, m2, p1, p2)
  explains only R^2 0.19-0.27 of Y. The fitted value function reaches R^2 0.001-0.14 at steps 1-8. No 1-D
  or 2-D sort captures this, so the per-step 2-D RQMC cannot flatten the variance.

## Not done (the run was cut short)
- Noisy-model RMSE tables. The truths are cached in results/, and running `experiment.py s0025|s005 ...`
  produces them.
- The sort-key variants arr_tie and arr2_m2tie, and runs at n = 65536.

Caveat: this is still sampling, not a deductive estimate.
