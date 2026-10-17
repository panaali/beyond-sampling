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
equal K. Cost = raw FLOPs from ARC's counter (no symmetric discount, i.e. as implemented) / FPE. NaN → 0.5.

RESULTS_PLACEHOLDER
