# tie_lines: line sampling along tie directions (and what actually paid off)

## Question
Can conditional Monte Carlo along the direction that crosses the error bands, d = (e_i - e_j)/sqrt2 for
the pair holding ranks 2,3 (or 1,2), estimate M_16,10's accuracy with lower compute-adjusted MSE than plain
sampling and than `m16/stratified.py`, under ARC's protocol (noisy copies, RMSE vs compute)?

**Short answer.** Exact integration along tie lines is unbiased and removes 3.6-4.9x of the variance per
input. But each line costs 2x16-20 forward-pass equivalents, so per unit of compute it **loses** to sampling
(0.1-0.9x) and to stratification. What does pay off is the cheapest version of the same idea: a 2-point
antithetic along the tie line (t -> -t, i.e. swapping the two near-tied values), combined with finer
model-free strata on the joint gaps (g12, g23). This **joint** estimator gives about **2.8x / 2.35x / 1.47x**
compute-adjusted MSE gain over sampling at sigma = 0 / 0.0025 / 0.005. That is **1.35x / 1.34x / 1.13x over
stratified.py**, even though here input-generation cost is counted for joint and not for stratified.py.
It is still sampling, not a deductive estimate.

## Method
Key files: `tielines.py` (line integrator and the constructions), `joint.py` (final estimator and design),
`estimator.py` (leaderboard entry), `estimators.py` (all variants with ARC's interface), `run_eval.py` and
`joint_analytic.py` (evaluation).

1. **Exact tie-line conditional MC** (`tielines.py`). Draw x. Let f(x) = {i,j} be a selected pair (for
   example the positions of ranks 2,3). Write x = u + t d with t = d.x ~ N(0,1) independent of u. The set
   W(u) = {t : f(u + t d) = {i,j}} is explicit and model-free. For the 2-3 pair it is
   |t| < sqrt2 min(o1 - m, m - o2). The estimator is Z = P(correct, t in W | u) / P(t in W | u). It is
   unbiased because E[1{f=ij} | u_ij] = P(W | u_ij); the derivation is in the docstring. The numerator is
   exact: walk the network's linear pieces inside W only, solve where the target logit wins (logits are
   affine per piece, and the target switches at t = 0), and integrate the Gaussian density in closed form.
   The integrator generalises `m16/lineint.py` to any direction and is batched over lines.
   Variants:
   - (a) the full window;
   - (b) narrow windows (select only when the gap is < c x midpoint; plain indicator otherwise; the 12-pair
     window excludes the 23 selection region analytically);
   - (c) windows restricted to the stratified.py stratum and the min-gap pair kind (found by bisection on
     monotone conditions), used inside stratification.
2. **Tie swap** (2-point antithetic on the same line). s(x) swaps the positions holding the two values with
   the smaller relative top gap. s is a measure-preserving involution and leaves the sorted values (hence
   every value-based stratum) unchanged, so (g(x) + g(s x))/2 is unbiased in any stratum. It costs 2 evals.
3. **Joint strata** (`joint.py`): blind (v2 <= 0), plus a 13x13 grid on (g12, g23) relative gaps. Cells with
   mass < 2e-3 are greedily merged into grid neighbours, giving 97 strata. Masses come from 4e7 model-free
   draws. Allocation is Neyman with cost (n_h ~ P_h sigma_h / sqrt c_h, sigma floor 0.02), using sigma from a
   1e6-draw pilot on the **original model only**. Swap is used in the 43 strata where the pilot says
   sigma_swap sqrt2 < sigma_plain. Inputs for all strata are drawn from one shared rejection pool, with every
   candidate counted.
4. Also tried: scrambled-Sobol RQMC with the swap (`est_swap_rqmc`), and finer strata that add x9's rank or
   v2's magnitude (`finer.py`).

## Status
Done. `estimator.py` exposes `est` (joint) with `DEDUCTIVE = False`.

## Results
Gain = compute-adjusted MSE gain over plain sampling = [acc(1-acc)/cost] / MSE (>1 beats sampling).
Original model unless stated otherwise.

**Tie lines (exact) lose at equal compute** [measured]:

| variant | Var per input (plain 0.044) | pieces/line | gain (piece = 2 evals) |
|---|---|---|---|
| full window, 2-3 pair | 0.0122 | 16.4 | 0.11 |
| full window, 1-2 pair | 0.0092 | 20.0 | 0.12 |
| narrow windows c23=c12=0.02 / 0.05 / 0.1 | 0.033 / 0.026 / 0.021 | 3.8-7.4 | 0.86 / 0.63 / 0.44 |
| inside stratified.py strata 1-2 (oracle Neyman) | sd 0.21 vs swap 0.25 vs plain 0.48 | 3.3 (cost 6.7) | 1.70 overall vs 2.04 for plain strat |

Validation: line integrals match 400k-point brute force on 6 lines to 1e-4, and the means of Z are
unbiased (0.95328 +- 0.00078 and 0.95188 +- 0.00068 vs 0.95318).
Commands: `python check1.py`, `python check2.py`, `python check5.py`, `python analytic.py 20000`, run from
this folder with the venv; check1/2 were run from the repo root.

**Swap and strata, precise analytic gains** (`python joint_analytic.py`; per-stratum variances from an
independent 1e6 sample per model, with rounding and generation cost; pooled over copies as
mean(acc(1-acc)) / mean(acc(1-acc)/gain)) [measured]:

| | sigma=0 | sigma=0.0025 (6 copies) | sigma=0.005 (6 copies) |
|---|---|---|---|
| stratified.py (no generation cost) | 2.08 | 1.75 | 1.30 |
| stratified.py (its rejection cost counted, ~0.43 evals/eval) | 1.45 | 1.22 | 0.91 |
| stratified.py strata + swap | 2.25 | 1.92 | 1.42 |
| **joint (g12 x g23) + swap, generation counted** | **2.81** | **2.35** | **1.47** |

Joint per copy: 1.84-2.69 at sigma=0.0025 and 0.93-2.28 at sigma=0.005. Its worst copy (acc 0.823) is
0.93, where stratified.py gets 0.97: allocation tuned on the original fits a heavily changed copy badly. The
gain is flat in budget (1k-64k).

**Rep-based protocol check** (`python run_eval.py <sigma> <est> 4000,16000,64000 8`, 6 copies x 8 reps; the
original model has 40 reps; truth n=4e6 per copy in `truth.json`; rows in `noisy.out` and `orig.out`)
[measured]. These gains are noisy (s.e. about 20%). They agree with the analytic table:

| sigma | budget | sampling RMSE | strat RMSE | joint RMSE (gain) |
|---|---|---|---|---|
| 0 | 4k / 16k / 64k | (exact 0.0033 / 0.0017 / 0.00084) | 0.0024 / 0.0012 / 0.00051 | 0.0019 (3.1) / 0.0013 (1.8) / 0.00047 (3.2) |
| 0.0025 | 4k / 16k / 64k | 0.0036 / 0.0020 / 0.00096 | 0.0030 / 0.0014 / 0.00071 | 0.0024 (2.5) / 0.0013 (2.1) / 0.00069 (2.0) |
| 0.005 | 4k / 16k / 64k | 0.0047 / 0.0023 / 0.0010 | 0.0050 / 0.0021 / 0.00080 | 0.0038 (1.5) / 0.0018 (1.8) / 0.0011 (1.2) |

Unbiasedness: across all 9 joint runs, bias/s.e. ranges from -2.0 to +1.4, which is consistent with zero.
On the original model at 64k the bias is -0.00007 +- 0.00007.

**RQMC** (`python rqmc_swap2.py`, 48 scrambles): raw RQMC gives 1.04 / 1.37x and swap+RQMC 1.18 / 0.98x
(the gain counts the swap's cost) at n = 4096 / 16384. That is no material gain. An earlier "2-3x" reading
came from a bool+bool bug and was retracted. [measured]

**Finer strata** (`python finer.py 1000000 6`): adding x9's rank or v2's magnitude to the grid helps the
original but overfits and loses on noisy copies (s0.0025_1: 0.82-0.99 vs 1.68 for the plain grid). [measured]

## Compute accounting
- One forward pass of the RNN on one sequence = 1. A tangent pass (forward plus directional derivative, used
  by the line walker) = 2.
- Input generation is counted for joint: every candidate N(0,I_10) draw, rejected ones included, is
  200 FLOPs / 6e3 = 0.033 evals. The shared pool needs about 3.3 candidates per eval, about 10% overhead,
  and the budget is pre-scaled so that total cost is about the budget. For the stratified.py rows it is not
  counted unless stated.
- Precomputation reused across models (not counted, declared): stratum masses (4e7 model-free draws,
  `joint_cells_P.json`), and the allocation pilot on the original model (1e6 model evals plus 1e6 swap
  evals, `joint_design.json`). This matches stratified.py's convention. A per-copy pilot is not used.
- Totals for this folder's experiments: roughly 1e8 model evaluations, mostly truths (12 x 4e6) and
  analytic variance runs (13 x 2e6).

## Claims
1. [measured] Tie-direction line integration is exact and unbiased, and cuts per-input variance 3.6x (2-3
   pair) and 4.9x (1-2 pair), versus about 2x for coordinate lines (FINDINGS). It still loses at equal
   compute (0.1x full window, at best 0.86x narrow), because every line crosses the record-changing tie,
   which forces at least 2-4 linear pieces, and the variance that remains is between lines.
2. [measured] Inside the tie strata, most of the remaining variance is between u (different other
   coordinates), not along t. The exact line (sd 0.21) barely beats the 2-point swap (sd 0.25) at 3x its
   cost.
3. [measured] Joint (g12 x g23) strata plus swap beat stratified.py by about 1.35x (sigma <= 0.0025) and
   1.13x (sigma = 0.005) in compute-adjusted MSE, with generation cost counted for joint only. Counting it
   for both, the margin is about 1.9x / 1.6x.
4. [inferred] The swap works because, near a tie, the network's choice depends mainly on positions and not
   on the tiny value gap. g(x) and g(sx) are then strongly anti-correlated: one is right and one is wrong.
5. [speculative] Joint's advantage over stratified.py should shrink further for noisier copies, because the
   allocation is fitted to the original model.

## Alternatives not ruled out
- For claim 3, two alternatives:
  - (i) The gain comes from the finer strata alone, not the swap. Tested: grid without swap gives 2.84 vs
    3.25 with swap on the original (`finer.py`), so both contribute.
  - (ii) The analytic gain is optimistic, because sigma is estimated from the same data as the evaluation.
    Tested: the design pilot uses seed 99, the analytic evaluation uses seed 31, and rep-based MSEs on
    independent seeds agree within noise. One rep row (sigma=0.005 at 64k: 1.16 +- 0.24 vs analytic 1.47)
    sits low, at about 1.3 s.e.
  - (iii) Not tested: whether a per-copy pilot (cost counted) would beat a fixed design on very noisy copies.
- For claim 4, the anti-correlation could also come from gain mismatch (FINDINGS: gain ratio 0.98-1.0)
  rather than position preference. I did not separate the two.
- RQMC was tested only at two n with 48 scrambles. Small gains (<1.3x) cannot be excluded.
- The line integrator could be made cheaper (secant pieces without tangents, about 1 eval per piece). By the
  numbers above this would still not beat the swap (claim 2), but it was not built.

## Failure modes
- It is sampling. The mechanism only decides where to sample and which antithetic pair to use. It is not a
  deductive estimate, and it does not beat sampling by orders of magnitude.
- The allocation is fixed from the original model. On a copy whose errors move away from top-gap ties
  (s0.005_1, acc 0.82) it falls to about 0.93x sampling.
- Small budgets: each of the 97 strata gets at least 1 sample, which distorts the allocation below about 500
  evals.
- Stratum masses come from 4e7 draws. The induced bias is about 5e-5 or less, which is negligible against an
  RMSE >= 4.7e-4 at 64k, but it would matter at budgets above about 1e7.
- Exact tie lines: the window-walk caps at 400 pieces, which was never hit, and they cost 6-40 evals per line.
