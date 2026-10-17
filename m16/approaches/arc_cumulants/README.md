# arc_cumulants (T35): ARC's Hermite-corrected cumulant propagation on the 2nd-argmax RNN

↑ [Approaches index](../README.md) · [INDEX (registry + ledger)](../../../INDEX.md) · [TASKS](../../../TASKS.md)

Sources read (all logged in [lit/web_log.md](../../../lit/web_log.md)): arXiv:2605.05179v2 (full PDF, main text,
appendices A-J, S.1, S.3.3-S.4.3), the code repo
[alignment-research-center/mlp_cumulant_propagation](https://github.com/alignment-research-center/mlp_cumulant_propagation)
(commit 93d091a, vendored here unmodified except one FLOP-counter assert, see `vendor/SOURCE.txt`), the blog post of
2026-05-07, the LessWrong/AF crosspost and its 5 comments, and the 2025-11-18 "Competing with sampling" post.

## Question
ARC reports an input-free estimator ("cumulant propagation") that beats Monte Carlo at estimating E[M(x)] for wide
random ReLU MLPs. (1) What exactly is it? (2) Applied faithfully to M_H,T (the 2nd-argmax RNN), how close does it
get to the accuracy, and how does it compare with sampling on leaderboard v3? (3) If it fails, why: narrowness,
training, or the architecture? (4) Is it useful as a conditional proxy (control variate / band-width proxy, T33)?

## 1. ARC's method, precisely (from the paper)

**Problem.** "Given weights θ ... drawn independently from N(0, 2/n) ..., estimate E_{X∼N(0,I_n)}[M_θ(X)]", with
"mean squared error O(ε²) over the choice of θ for each output neuron" (Sec. 2.2). MLP without biases, equal widths,
no final activation (Def. 2.1). Loss functions: "Our methods can easily be adapted to other closed-form loss functions
that only depend on the network's output, by treating the loss function as an additional activation function. For
more complex formally-defined loss functions that also depend on the network's input, it is often still possible to
adapt our methods, although more work may be required" (Sec. 2.2). Our target is of the second kind.

**What is propagated.** Estimated joint cumulant tensors κ_r of the activations, r ≤ K (plus, for odd K, "a single
number corresponding to the full trace of the (K+1)-th-order cumulant tensor"), stored in a harmonic decomposition
(Sec. 4.1, S.4.1). K is the only accuracy knob ("we treat K as a hyperparameter"). Input: κ_2 = I, all others 0.

**Linear step.** Exact multilinearity: κ_r[Wx] = κ_r[x] contracted with r copies of W, O(n^{r+1}) (S.1.2).

**Non-linear step ("Hermite correction").** The cumulants of φ(Z) are written by the Hermite-based diagram summation
formula (Thm A.2): a sum over connected "[≤2]-mixed" diagrams of products of input cumulants, weighted by 1-D Hermite
coefficients φ̂_k^{(μ_i,σ_i²)} = σ^{-k} E[φ(Y) He_k((Y-μ)/σ)], **taken separately for each neuron around a Gaussian
with that neuron's own mean and variance**. Two adjustments are essential (Sec. 4.1, ablation Sec. 6.4: without them
MSE is O(1) instead of O(n^{-K})):
1. **Power cumulants**: expand powers φ^p, so cumulants with repeated indices (e.g. variances) are computed exactly
   from 1-D integrals ("when K = 2, we compute Gaussian variances exactly, while continuing to use an approximation
   for off-diagonal Gaussian covariances").
2. **Extra full trace for odd K.**
The diagram sum is truncated by power counting in the width: keep diagrams with k(ν) ≤ K (eq. 13-14), because
off-diagonal cumulants of a wide random net are O(n^{-1/2}) per extra index. For K = 2 the off-diagonal covariance
update is the *leading term only*: Σ_ij ← Σ_ij c_i c_j with c_i = E[φ'(Y_i)] (Algorithm 2, line 12), not the full
series of Cor. A.3 (Wright et al. 2024). For ReLU all coefficients are closed form (Prop. S.3.8).
Variants: *augmented* (track more harmonic parts; same leading FLOPs, better constant), *factorized* (rank-factored
top cumulant; a factor n cheaper, same output).

**Indicator / threshold outputs.** Only 1-D thresholds: "we replace the final ReLU activation in our networks with the
threshold function 1_{z>3}" (Sec. 6.6), handled as one more coordinatewise activation (Heaviside Hermite coefficients).
Nothing on argmax, multi-coordinate indicators or input-dependent targets.

**Compute accounting.** FLOPs from "an extension of PyTorch's FLOP counter", adding entrywise +-*/; FLOPs on
symmetric tensors are *discounted* to what "an implementation with efficient symmetric tensor CUDA kernels" would use
(factors α, β of App. J, e.g. β_4 → 5/32). Sampling: N forward passes. Wall-clock is reported separately
(App. I): "In most cases, our algorithms underperform sampling" in wall-clock.

**MSE-vs-compute protocol.** "For each configuration, we calculate mean squared error (MSE) over all output neurons and
over 5 different random seeds. To calculate MSE, we estimate ground truth using Monte Carlo sampling with 2^34 ≈ 2×10^10
samples ... We use these same samples to estimate the variance of a single forward pass, and divide MSE by this
variance to obtain variance-normalized MSE. This choice of units means that Monte Carlo sampling with N samples has a
variance-normalized MSE of exactly 1/N" (Sec. 6.1). Comparison is variance-normalised MSE against FLOPs.

**Theory.** MSE O(ε²) in time O(n²/ε²) (basic) or O(n/ε²) (factorized, when ε² = O(n^-2)), fixed depth, polynomial
activations (Thm 5.1, 5.2); conjectured MSE ≤ c_K (L/n)^K.

**Narrow / deep / trained / RNN / AlgZoo.** Narrow: "Our algorithms perform poorly at low width, especially when the
number of hidden layers L is large, and can even get worse with the maximum cumulant order K in this setting" (App. C);
at width 16 with 8-12 hidden layers points are missing because "variance-normalized mean squared error greater than 10".
Open problems: "we do not know whether it is possible to match the performance of random sampling at large depth ...
or for narrow networks (i.e., asymptotically as the error tolerance ε → 0 while the width n is held fixed)" (Sec. 7.3).
Trained nets: "our methods rely on the networks being randomly initialized" and for trained networks "specific
higher-order deviations can become much more important" (blog); a "structure versus randomness" treatment is left to
future work (Sec. 7.2). Nothing on RNNs or AlgZoo in the paper, blog or comments (Hilton mentions a coming contest).

## 2. Method here

Key files: `arc_kprop.py` (wrapper), `estimator.py` (leaderboard), `selftest.py`, `validate_mlp.py`, `diag_layers.py`,
`diag_width.py`, `diag_skeleton.py`; `vendor/mlp_kprop` (ARC's code). Extra deps (git-ignored):
`pip install --target data/arc_kprop_deps jaxtyping einops opt_einsum joblib`.

**The RNN as an ARC network.** State s = (h ∈ R^H, x_0..x_{T-1}); x enters as pass-through coordinates with identity
activation, so step t is one layer: h ← ReLU(Wh h + Wi x_t), x ← x. Input cumulants κ_2 = diag(0_H, I_T). ARC's own
`linear_kprop` / `nonlin_kprop` (basic = SIMPLE, or AUGMENT) are applied unchanged, with per-unit Hermite coefficients
(identity / ReLU / Heaviside) and the exact metric W M W^T (not ARC's random-weight average E[WW^T]).

**Expressing accuracy (input-dependent target).** Two choices, both exact as networks (`selftest.py`: 0 mismatches
on 200k inputs at T = 4, 5, 10):
- `uncond`: read-out of 3 Heaviside layers on (h_T, x): pairwise 1[l_i>l_j] and 1[x_i>x_j] (2·C(T,2) units), counts
  → A_j = 1[argmax l = j], G1_j, G2_j (1[x_j is 2nd] = G1_j − G2_j), then C_j = 1[A_j + G1_j − G2_j ≥ 2], output Σ_j C_j.
- `cell`: condition on (p1, p2) = positions of the top-2 inputs; accuracy = Σ_cells P(cell)·P(l_{p2} > l_i ∀i | cell),
  P(cell) = 1/(T(T−1)). x | cell is non-Gaussian; its cumulants (orders 1-4) are model-free, precomputed from 4M exact
  conditional draws (sort, place top-2, permute the rest) and symmetrised; orders ≥ 5 set to 0. Read-out: 9 Heavisides
  + 1. This is the "per position of 2nd max" decomposition; E13 conditioned on the same (p1, p2).

**Leaderboard estimator.** Untuned rule: highest K ∈ {1..4} whose counted cost fits the budget, cell mode preferred at
equal K. Cost = raw FLOPs from ARC's counter (no symmetric discount, i.e. as implemented) / FPE. Out-of-range estimates are clipped to [0, 1] and NaN → 0.5 (model-free post-processing; it can only help the method).

## Status
**Done (negative as a whole-model estimator; negative as a skeleton-conditional proxy, see 3d).** The faithful
implementation is 41-100 pp off on every (H, T) and every K, and 2,900x to 1.5Mx worse than sampling in MSE on the
leaderboard. The cause is identified and tested: AlgZoo's RNN injects a **scalar** input per step, so hidden units are
perfectly (rank-1 at step 1) correlated; ARC's diagram truncation assumes O(n^-1/2) off-diagonal cumulants. On
*random* RNNs the same code works when the input per step has dimension H and is stuck at ~40-600 samples' worth of
accuracy when it is scalar, whatever the width (3b).

## 3. Results
All commands from the repo root with the venv active, `OMP_NUM_THREADS=1`.

**3a. Wrapper validation.** `python m16/approaches/arc_cumulants/validate_mlp.py 64` [measured]: on a He-initialised
width-64 ReLU MLP with 2 hidden layers our `propagate` equals ARC's `mlp_kprop` to 0.0 (K = 1, 2, 3) and gives
variance-normalised MSE 1.9e-3 / 6.7e-5 / 3.2e-6 (= sampling with 542 / 14,840 / 312,163 samples), i.e. the paper's
behaviour is reproduced. Identity pass-through layers keep the mean exact (4e-16). `python
m16/approaches/arc_cumulants/selftest.py`: both read-out networks equal `leaderboard.correct` on 200k inputs (0
mismatches) at T = 4, 5, 10.

**3b. Why it fails: scalar input injection (random RNNs, linear read-out, MC truth 1M).**
`python m16/approaches/arc_cumulants/diag_width.py T H1,H2 2,3` (outputs `out_width_*.txt`) [measured].
Entries: variance-normalised MSE (equivalent number of samples), mean of 3 seeds.

| T | H | input dim d = 1 (AlgZoo-like), K=2 / K=3 | input dim d = H, K=2 / K=3 |
|---|---|---|---|
| 4 | 16 | 6.1e-3 (164) / 4.9e-3 (206) | 2.1e-4 (4,679) / 2.0e-5 (50,269) |
| 4 | 64 | 6.6e-3 (151) / 6.7e-3 (150) | 3.4e-5 (29,837) / 1.2e-6 (817,168, MC floor) |
| 4 | 128 | 5.8e-3 (173) / 3.0e-2 (33) | 7.5e-6 (132,479) / 1.3e-6 (MC floor) |
| 10 | 16 | 2.3e-2 (44) / 8.8e-3 (114) | 2.1e-4 (4,794) / 1.8e-4 (5,675) |
| 10 | 64 | 5.8e-3 (174) / 1.6e-3 (623) | 2.3e-4 (4,419) / 3.9e-6 (254,268) |

With vector input the method improves with width and K as the paper predicts; with scalar input it does not improve
with width at T = 4 and barely at T = 10, and K = 3 can be worse than K = 2. This is a smooth (linear) read-out on
*random* weights; the trained models and the indicator read-out are harder still.

**3c. Faithful estimator on the trained models (base model of each (H, T)).**
`python m16/approaches/arc_cumulants/config_table.py T SIMPLE,AUGMENT uncond,cell`.
Cost = raw counted FLOPs / FPE (symmetric-discounted count in brackets in the logs) [measured].

| (H,T), truth | mode | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|---|
| (16,4) seed3, 0.99831 | uncond | 0.311 (8 FPE) | 0.182 (91) | 0.000 (3,659) | 0.217 (79,926) |
| | cell | 0.148 (85) | 0.232 (943) | 0.417 (59k) | 0.200 (1.05M) |
| | uncond AUGMENT | 0.311 | 0.000 | 3.000 | NaN (747k) |
| | cell AUGMENT | 0.148 | 0.366 | 0.583 | 0.583 (9.7M) |
| (16,10) orig, 0.95315 | uncond | 0.000 (14) | 0.017 (402) | 0.112 (45k) | **0.9495 (3.64M)** |
| | cell | 0.006 (888) | 0.181 (14.3k) | 0.108 (950k) | not run |

The one close number (orig, K = 4, -0.37 pp) is a coincidence: on DEV copies the same configuration gives
0.684 / 0.054 / 1.082 (truths 0.94 / 0.93 / 0.93, sigma 0.0025) and 0.193 / 0.270 / 1.493 (truths 0.62 / 0.52 / 0.57,
sigma 0.02), i.e. -88 to +93 pp; a 0.25% weight perturbation moves it by 25-90 pp
(`python m16/approaches/arc_cumulants/dev_check.py 10 4 uncond 6`, `out_dev_T10_K4_uncond.txt`) [measured].

Layer-by-layer (`python m16/approaches/arc_cumulants/diag_layers.py 16 4 K`) [measured]: h_1 is exact for K >= 2
(mean error 0.005 on a scale of 1.7); already at h_2 the mean error is 3.9 (K = 2) / 43 (K = 3) on a scale of 2.7, and
variances blow up (K = 3: 2e9 at h_3). The break is in the off-diagonal covariance of h_1, whose units are all
functions of the one scalar x_0 (correlation ±1), where Algorithm 2's leading-order update Σ_ij c_i c_j is O(1) wrong.

**3d. Leaderboard v3** (`python m16/leaderboard.py run arc_cumulants --H 16 --T 4 --budgets 1000,4000,16000`, the same
with `--T 5`, and `--T 10 --budgets 1000,4000,16000,64000`; ratio = MSE / sampling MSE at equal cost, all groups;
class unreviewed, deterministic, memoised so the time column after the first budget is meaningless) [measured]:

| (H,T) | budget → config (cost FPE) | bias pp by group (seed, s0.0025 ... s0.02) | MSE ratio range |
|---|---|---|---|
| (16,4) | 1000 → cell K2 (943) | -66.8, -75.4, -73.7, -70.1, -61.9 | 2,868 - 195,475 |
| (16,4) | 4000, 16000 → uncond K3 (3,659) | -93.1, -98.5, -96.9, -93.3, -85.1 | 20,997 - 1,483,056 |
| (16,5) | 1000 → uncond K2 (114) | -87.2, -88.3, -86.5, -82.6, -74.2 | 467 - 8,950 |
| (16,5) | 4000 → cell K2 (1,788) | -64.5, -78.0, -76.3, -72.4, -64.2 | 5,468 - 78,879 |
| (16,5) | 16000 → uncond K3 (5,040) | -83.3, -98.0, -91.8, -83.5, -57.7 | 12,448 - 368,560 |
T10_PLACEHOLDER

SKEL_PLACEHOLDER

## 4. What ARC's method does that E13 did not (and vice versa)
- ARC tracks **higher cumulants** (K = 3, 4) with a principled truncation and the power-cumulant / full-trace
  bookkeeping that makes same-neuron terms exact; E13 stopped at Gaussian (mixtures).
- ARC expands each neuron around **its own** Gaussian (μ_i, σ_i²) and sums diagrams; E13 used exact bivariate-normal
  ReLU moments, i.e. the *full* Hermite series for pairs (Wright et al.). At K = 2 ARC keeps only the leading
  off-diagonal term, so on this model ARC-K2 is *less* accurate per step than E13's closure.
- ARC is **unconditional** (one cumulant tower, no mixture, no discrete state); E13 conditioned on (p1, p2) and gate
  patterns. Our `cell` mode adds E13's (p1, p2) conditioning to ARC; it does not rescue it.
- ARC's error control is entirely asymptotic in width (MSE ~ c_K (L/n)^K). Here n_eff = 26, L = 13 including the
  read-out, and the scalar input makes correlations O(1) at any width (3b), so the expansion parameter is not small.
Both fail; E13's exact-cumulant ceiling (-0.35 pp with 4 cumulants of the *final margin* per 10k cells) says that even
perfect propagation would need a better read-out than any 4-cumulant closure.

## Compute accounting
Counted: every tensor operation in ARC's linear and non-linear steps, Hermite coefficients, power-cumulant
conversions and harmonic projections, via ARC's `NamedFlopCounter` (PyTorch FlopCounterMode + entrywise ops), **raw**
(we do not apply ARC's symmetric-tensor discount, rule 7 "as implemented"; the discounted count is 0.3-0.7x of raw and
is printed in the logs). erf/exp inside Φ/φ count as entrywise ops of the counter (not separately weighted).
Converted at `fpe_flops(H,T)`. Model-free precomputation (cell input cumulants, 4M exact conditional draws per T, about
8e10 FLOPs at T = 10; the per-(H,T) cost table on random weights) is declared in `PRECOMPUTE_FLOPS`. No model
evaluations on random inputs in the estimator. Diagnostics (MC references) are not part of any estimator.
CPU: the Python implementation runs at 10-90x the CPU time of sampling at equal counted cost (leaderboard time column).

## Claims
- [measured] ARC's algorithm, applied faithfully (their code, unchanged) to M_16,4, M_16,5 and M_16,10 with an exact
  network form of the 0/1 correctness, is 41-100 pp off for all K in 1..4, both variants, both encodings of the
  target; MSE is 470x-1.5Mx sampling's at equal cost on leaderboard v3.
- [measured] The single near-hit (M_16,10 orig, K = 4, -0.37 pp) does not survive DEV copies (-88..+93 pp).
- [measured] On random RNNs the method works (beats sampling-equivalents by 10-100x more as H grows) when each step's
  input has dimension H, and stalls at ~40-600 sample-equivalents when the input is scalar, independent of width.
- [inferred] The failure on AlgZoo is architectural (scalar input ⇒ perfectly correlated hidden units ⇒ ARC's
  leading-order off-diagonal truncation is O(1) wrong), not specific to training; narrowness and depth add to it.
- [inferred] Whatever ARC's unpublished M_16,10 solution is, it is not this algorithm applied as-is; it would have to
  condition on structure (the "structure versus randomness" split they mention) rather than expand around Gaussians.
  [speculative] beyond that.

## Alternatives not ruled out
1. *A bug in our wrapper.* Separated by 3a: identical to ARC's `mlp_kprop` on MLPs, exact on identity layers,
   correct network form of the target. Not tested: rectangular W against an independent cumulant implementation
   (ARC's library supports it but its FLOP assert did not, which we removed).
2. *A better encoding of the target would rescue it* (e.g. max via ReLU trees, a margin + single threshold, or
   order-statistic features). Not tested; 3b shows the hidden-state propagation itself is already O(1) wrong at h_2,
   before any read-out, so a read-out change cannot fix it.
3. *Re-parametrising the scalar input* (h_1 = a·x_0⁺ + b·x_0⁻ is exactly 2-dimensional) and propagating in a basis where
   cumulants are small. Not tested; this is "structure versus randomness" and no longer ARC's method.
4. *The full pairwise Hermite series (Wright et al.) instead of the leading term.* At K = 2 this is E13's single-
   component Gaussian closure, which was -52..-69 pp; not re-run here.
5. *Width alone*: 3b shows scalar-input failure does not shrink from H = 16 to 128 at T = 4; it was not tested at
   larger H or T.

## Failure modes
- Perfectly correlated units (rank-1 input injection) break the 1/n power counting from the first layer on.
- Truncated series give negative variances; ARC's code clamps to 1e-10, then Hermite coefficients σ^{-k} explode and
  Heaviside read-outs saturate to 0/1 (estimates such as 0.000, 3.000, NaN, 5/12, 7/12).
- Raising K makes things worse at this width (as the paper warns for narrow nets).
- Cost grows as n^{K+1} with n = the widest layer: the T = 10 unconditional read-out has 90 units, so K = 4 costs 3.6M FPE
  and several hundred MB-GB of memory (not run under the 2 GB cap in the estimator).

