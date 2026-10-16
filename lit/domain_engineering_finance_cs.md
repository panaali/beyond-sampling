# Methods from engineering, quant finance, OR and formal methods for estimating the RNN's 2nd-argmax accuracy

Problem recap: f(x) = 1[argmax W_o h_10(x) = rank-2 position of x], x ~ N(0, I_10). f is piecewise constant on polyhedral cones (f is ReLU-PL and positively homogeneous). The target is E f ≈ 0.953. Errors sit in thin wedges near ties x_(1)≈x_(2) and x_(2)≈x_(3). The baseline to beat is mechanism-informed stratified sampling, with a 1.2–2.8x MSE gain over plain MC.

## Two structural facts that drive the ranking

1. **Homogeneity removes the radial dimension.** f(rx) = f(x) for r > 0, so E f equals the normalized solid angle of the "correct" cones. Directional simulation, the structural-reliability workhorse, therefore gains nothing on its own, because the radial integral is trivially 1 or 0. Gains have to come from integrating exactly over an angular subspace of dimension ≥ 1.
2. **The failure probability is not rare (p ≈ 0.047).** Subset simulation, cross-entropy importance sampling and FORM/SORM are built for p ≈ 1e-4 to 1e-9 and give little here. What matters is smoothing the discontinuity (conditional MC / preintegration) and control variates with exactly known means. That is the toolkit quant finance uses for digital and barrier payoffs.

---

## Ranked methods

### 1. Conditional MC / preintegration along the tie direction (finance: digital-option smoothing)
**Citations.** Griewank, Kuo, Leövey, Sloan, "High dimensional integration of kinks and jumps — smoothing by preintegration", J. Comput. Appl. Math. 2018, arXiv:1712.00920. Bayer, Ben Hammouda, Tempone, "Numerical smoothing with hierarchical adaptive sparse grids and QMC for efficient option pricing", Quant. Finance 2023, arXiv:2111.01874. Gilbert, Kuo, Sloan, "Preintegration is not smoothing when monotonicity fails", arXiv:2112.11621.

**Idea.** Pick one direction v along which the discontinuity is crossed transversally. Integrate the indicator exactly along v: find the kinks by root-finding, then sum Gaussian CDF differences. Then apply MC or QMC to the remaining d−1 variables. The result is a smooth integrand with lower variance, and it is also where QMC starts to work.

**Recipe.**
- Do not use coordinate lines. For each outer sample, condition on the sorted-order configuration and integrate along the gap direction between ranks 2 and 3 (and/or 1 and 2). One concrete scheme: write x = z + t·u with u = (e_{(2)} − e_{(3)})/√2 and z ⊥ u, so t ~ N(0, 1) independent of z.
- Along the line, h_t(x) is piecewise linear in t. Track the activation pattern exactly: in each piece every pre-activation is a + b·t, so the next breakpoint is min over units of −a/b. This gives the exact 1-D measure in O(#pieces) forward passes with no bisection.
- This is "exact 1-D integration" (already tried), but aimed at the band. That turns the conditional variance from Bernoulli-like into a smooth function of z, which is close to constant away from ties.
- The finance literature reports that monotonicity along v is what makes the remaining integrand smooth. Here the rank-2 output is "monotone" in the gap coordinate only locally. Gilbert–Kuo–Sloan warn that smoothing fails without monotonicity, so check it.

**Expected gain.** Each line costs about 5–20 forward passes. If the conditional probability along the tie direction has variance well below p(1−p), for example about 0.1·p(1−p), the net work-normalized gain is 3–10x over plain MC. Pairing it with RQMC on the outer variables (which failed on the raw indicator) may add another 2–5x, since the integrand is now smooth. **Highest promise. Feasible now.**

### 2. Great-circle (2-D angular) conditional MC: directional simulation adapted to homogeneity
**Citations.** Bjerager, "Probability integration by directional simulation", J. Eng. Mech. 114(8), 1988. Ditlevsen, Melchers, Gluver, "General multi-dimensional probability integration by directional simulation", Comput. Struct. 36, 1990. Related: Koutsourelakis, Pradlwarter, Schuëller, "Reliability of structures in high dimensions, Part I", Prob. Eng. Mech. 19, 2004 (line sampling).

**Idea.** Directional simulation samples directions and integrates radially in closed form (a chi distribution). Because f is homogeneous, the radial step is trivial, so lift it by one dimension. Sample a random 2-plane P = span(a, b). The Gaussian restricted to P is rotationally symmetric, so the angle θ is uniform. The exact fraction of θ ∈ [0, 2π) where f = 1 is computable by tracking breakpoints on the circle. Each pre-activation is α cos θ + β sin θ within a linear region, so the crossings are closed-form arctan expressions.

**Recipe.**
- Sample a ~ N(0, I).
- Take b = a component along the tie direction (e_{(1)} − e_{(2)} or e_{(2)} − e_{(3)} of a), orthogonalized against a. This is the "important direction" of line sampling, chosen per sample and made unbiased by symmetrization or by randomizing which tie to use.
- Estimator: mean over planes of the correct arc-fraction.
- Unbiasedness: if b is chosen as a measurable function of a that keeps the plane's law rotation-invariant (e.g. a random plane), the estimator is unbiased. With tie-aligned b, P is not uniformly distributed. Weight via the Blaschke–Petkantschin / Crofton formula, or keep the plane random and just exploit that every circle crosses about 2·(number of tie hyperplanes) bands.

**Expected gain.** A circle crosses every tie hyperplane x_i = x_j twice. Each arc-fraction therefore already averages over many orderings, which suppresses the Bernoulli variance strongly. Gain is likely ≥ 5x per unit compute, at about 50–200 forward-pass-equivalents per circle. **High promise. Feasible.** This generalizes "1-D lines" by exploiting homogeneity instead of fighting it.

### 3. Control variate from a surrogate with an exactly computable Gaussian measure (finance CV + Genz MVN)
**Citations.** Genz and Bretz, *Computation of Multivariate Normal and t Probabilities*, Springer LNS 195, 2009 (chapter on linear inequality constraints and singular cases; R `mvtnorm`, SciPy `multivariate_normal.cdf`). Gessner, Kanjilal, Hennig, "Integrals over Gaussians under linear domain constraints" (LIN-ESS + HDR), AISTATS 2020, arXiv:1910.09328. Glasserman, *Monte Carlo Methods in Financial Engineering*, 2004, ch. 4 (control variates).

**Idea.** Build a surrogate g with E g known to high precision, correlated with f. Then estimate E f = E g + E[f − g]. The variance falls by 1/(1 − ρ²).

**Recipe.**
- g = 1[argmax(M x) = rank-2 index], where M = W_o · J is the network frozen at its dominant activation pattern (or the pattern at the canonical ordered input, permuted per ordering). E g is a sum over the 90 (top, 2nd) pairs of Gaussian probabilities of polyhedral cones in R^10: 9 ordering constraints plus 9 logit constraints. Compute each cone with Genz's sequential-conditioning algorithm on the 18 × 10 constraint system (singular MVN), or with HDR/LIN-ESS, to about 1e-5. Exploit the permutation symmetry of the input distribution to cut the cone count.
- A better g: a "two-pattern" surrogate that also captures the main mechanism's failure mode near ties, taken from the existing mechanism analysis.
- The residual f − g is nonzero only where the true network leaves the frozen pattern and disagrees. Estimate it with the stratified sampler or with method 1.

**Expected gain.** If g reproduces about 70–80% of the error mass at the right locations (ρ ≈ 0.8–0.9), the gain is 3–5x on top of whatever estimator handles the residual, and it composes multiplicatively with methods 1 and 2. **High promise. Moderate effort.** It is also deductive in spirit: the bulk of the answer comes from a closed-form quadrature.

### 4. Exact region enumeration plus solid-angle evaluation (computational geometry; ProbStar)
**Citations.** Tran et al., "Quantitative verification for neural networks using ProbStars", HSCC 2023, and the StarV tool (CAV 2024). Ribando, "Measuring solid angles beyond dimension three", DCG 36, 2006. Fitisone and Zhou, "Solid angle measure of polyhedral cones", arXiv:2304.11102. "ReLU networks as random functions" (arXiv:2503.22082): the probability of each affine piece is a Gaussian orthant probability.

**Idea.** Propagate the input set exactly through the ReLUs, splitting at each sign ambiguity. Each leaf is a star set (polyhedral cone plus affine map), so its classification is decided by one more linear split. Sum the Gaussian measures of the "correct" leaves. For cones, the measure is a normalized solid angle: closed form in 2-D/3-D, a hypergeometric series or Genz MVN in 10-D.

**Recipe.**
- Unroll the 10 steps × 16 units (160 ReLUs) and use symmetry: condition on the input ordering, which is 1 of 10! orderings with equal mass.
- Split only on neurons whose sign is ambiguous over the current cone, checked with an LP.
- Prune any cone whose classification is provably constant (LP on the logit margins).
- Accept remaining mass below ε as an interval bound.

**Expected gain.** This is a deterministic answer with certified bounds, but the leaf count may be exponential. Feasible only if, per ordering cone, most neurons are sign-stable. Mechanism analysis (99.96% correct at typical inputs) suggests most are. The 10! orderings make full enumeration impossible unless the network is permutation-equivariant enough to reuse leaves. Hybrid option: enumerate exactly within strata near ties and sample orderings elsewhere. **Medium promise. Higher effort.** The 2026 hardness result (arXiv:2607.14375: smoothed exact ReLU verification has no polynomial expected time unless NP ⊆ BPP) caps the worst case.

### 5. Branch-and-bound probabilistic verification with anytime bounds (formal methods)
**Citation.** Boetius, Leue, Sutter, "Solving probabilistic verification problems of neural networks using branch and bound", ICML 2025, PMLR 267:4660–4699, arXiv:2405.17556 (code: PyBaB / probspecs).

**Idea.** Branch the input space into hyper-rectangles or cones. On each piece, use bound propagation (CROWN/IBP) to decide "always correct", "always wrong" or "undecided". Add the decided pieces' Gaussian mass to the lower and upper bounds, and refine the undecided pieces chosen heuristically. The procedure is sound and anytime.

**Recipe.**
- Branch on the ordering first (cone splits on x_i − x_j), then on input coordinates within tie bands.
- Report [L, U]. When U − L is small, sample only the undecided leaves (a stratified estimator whose strata are certified pure). This is a principled version of the current stratified sampler: pure strata contribute zero variance.

**Expected gain.** If B&B certifies even 90% of the mass as pure, the variance confined to undecided strata of mass m drops roughly in proportion to m, which could be a 10x+ MSE gain at matched compute after the B&B overhead. Recurrent depth (10 steps) makes the CROWN bounds loose, so expect many undecided boxes near ties. **Medium-high promise. Moderate effort** with existing auto_LiRPA.

### 6. Line sampling with an adaptive important direction (structural reliability)
**Citations.** Koutsourelakis, Pradlwarter, Schuëller 2004 (above). Pradlwarter et al. 2007 (adaptive line sampling). Papaioannou and Straub, "Combination line sampling for structural reliability analysis", Struct. Safety 88, 2021.

**Idea.** Pick a direction α pointing to the failure domain. For each sample of the orthogonal complement, solve the 1-D failure probability along α exactly. When the limit state is nearly flat along α, the variance is much smaller than MC's.

**Recipe.** This is essentially method 1, with the twist that α is fixed per stratum: within the stratum "positions (i, j, k) are ranks 1, 2, 3", set α = e_j − e_k. Use combination line sampling to adapt α from the samples (Papaioannou–Straub average the failure points). The point to take from this literature is how to adapt the direction and combine estimators across directions without bias.

**Expected gain.** Same order as method 1. Mainly a source of engineering tricks. **Promise tied to method 1.**

### 7. Neyman / optimal-allocation stratification plus post-stratification on order-statistic gaps (OR)
**Citations.** Glasserman 2004, §4.3. Cochran, *Sampling Techniques*, 1977 (Neyman allocation). Etoré and Jourdain, "Adaptive optimal allocation in stratified sampling methods", Methodol. Comput. Appl. Probab. 2010.

**Idea.** Choose strata where the conditional error rate varies most. Allocate n_h ∝ P_h σ_h, estimating σ_h adaptively in a pilot phase (Etoré–Jourdain prove the adaptive scheme converges to the optimal allocation).

**Recipe.**
- Stratify on the normalized gaps g12 = (x_(1) − x_(2))/s and g23. Their joint law is known in closed form from order-statistic theory, so the stratum probabilities P_h are exact.
- Use 2-D strata on (g12, g23) with about 10×10 cells and log-spaced breakpoints near 0.
- Sample within a stratum via the order-statistic / exponential-spacings representation (or rejection), and allocate adaptively.
- Most mass sits where σ_h ≈ 0.02 (p ≈ 0.0004), so nearly all samples go to the tie corners.

**Expected gain.** Ideal stratification on gaps captures most of the variance. If about 95% of error mass sits where both gaps are below about 0.1σ, gains of 3–8x over plain MC are plausible, versus the current 1.2–2.8x. The likely shortfall today is allocation and stratification variables, not the idea. **High feasibility. Medium promise.** It composes with method 1 (smoothing inside each stratum).

### 8. Multilevel / multi-fidelity MC with a cheap surrogate (finance MLMC)
**Citations.** Giles, "Multilevel Monte Carlo methods", Acta Numerica 24, 2015. Giles, Nagapetyan, Ritter, "Multilevel MC approximation of distribution functions and densities", SIAM/ASA JUQ 2015 (smoothing of indicator payoffs). Peherstorfer, Willcox, Gunzburger, "Optimal model management for multifidelity UQ", SIAM J. Sci. Comput. 2016.

**Idea.** E f = E g_0 + E[f − g_0], with many cheap samples of g_0 and few expensive samples of the difference.

**Recipe.**
- g_0 can be the frozen-pattern linear classifier (one 10×10 matmul plus argmax, versus a full 10-step RNN).
- If g_0's mean cannot be computed exactly (method 3), estimate it with 100x more samples.
- Correlation between f and g_0 decides everything. Per-sample cost ratio is about 10–20.

**Expected gain.** With ρ ≈ 0.85 and a cost ratio of 15, the multifidelity gain is about 3x. **Medium promise. Easy.** Strictly dominated by method 3 if g_0's mean is computed exactly.

### 9. Holmes–Diaconis–Ross / LIN-ESS nested estimation for the residual error cones
**Citation.** Gessner, Kanjilal, Hennig, AISTATS 2020 (above). Underlying: subset simulation, Au and Beck, Prob. Eng. Mech. 16, 2001.

**Idea.** Write a small Gaussian mass in a polyhedron as a product of conditional probabilities over nested relaxed domains. Sample each level with the rejection-free elliptical slice sampler, whose ellipse–hyperplane intersections are analytic.

**Recipe.** Use it to compute the exact mass of the error cones of a surrogate (method 3) or of individual error leaves (method 4) when Genz's method becomes slow at small masses. It is not useful directly on the net, since 0.047 is not rare.

**Expected gain.** An enabling tool only. **Low direct promise.**

### 10. Randomized volume / Gaussian-measure algorithms (Gaussian cooling, hit-and-run)
**Citations.** Cousins and Vempala, "Gaussian cooling and O*(n^3) algorithms for volume and Gaussian volume", SIAM J. Comput. 2018, arXiv:1409.6011. Lovász and Vempala, "Hit-and-run from a corner", 2006.

**Assessment.** These are designed for exponentially small volumes in high dimension. In d = 10 with masses around 1e-2 to 1e-3 per cone, direct MC or Genz is faster. **Not recommended**, except as a cross-check in method 4.

### 11. Weighted model integration / #SMT (formal methods)
**Citations.** Morettin, Passerini, Sebastiani, "Advanced SMT techniques for weighted model integration", AIJ 2019. Zeng and Van den Broeck, "Efficient search-based weighted model integration", UAI 2019.

**Assessment.** WMI handles polynomial weights over LRA formulas, not Gaussian densities. Gaussian weights need numeric integration per cell, which reduces to method 4 with an SMT solver doing the enumeration. Possibly useful as the enumeration engine (all-SMT over activation patterns with ordering constraints), but no gain over a hand-rolled LP-based star-set enumerator. **Low promise.**

### 12. FORM/SORM and adaptive importance sampling (cross-entropy, sequential directional IS)
**Citations.** Rackwitz and Fiessler 1978; Breitung 1984 (SORM). Rubinstein and Kroese, *The Cross-Entropy Method*, 2004. Cheng, Papaioannou et al., "Rare event estimation with sequential directional importance sampling", arXiv:2202.05683.

**Assessment.**
- FORM/SORM linearize the limit-state surface at the design point. Here the failure set is a union of many thin wedges through the origin with no design point (it contains arbitrarily small x). The result is biased, which matches the experience with sensitivity probing.
- IS that oversamples tie bands is useful only as a defensive mixture, IS density q = 0.5·φ + 0.5·(tie-concentrated). Even then it is a weaker version of stratification (method 7).

**Low promise.**

---

## Bottom-line ranking (promise × feasibility)

| Rank | Method | Expected MSE gain vs plain MC | Effort |
|---|---|---|---|
| 1 | Tie-direction preintegration (conditional MC), then RQMC on the rest | 5–30x | low–med |
| 2 | Great-circle exact angular integration (homogeneity-adapted directional simulation) | 5–20x | med |
| 3 | Control variate: frozen-pattern surrogate, mean via Genz MVN over polyhedral cones | ×3–5, composes with others | med |
| 4 | Branch-and-bound certified-pure strata (Boetius et al. 2025) | 10x+ if bounds are tight; anytime [L, U] | med–high |
| 5 | Gap-stratified Neyman allocation with exact stratum probabilities | 3–8x | low |
| 6 | Exact star-set enumeration plus solid angles | deterministic if tractable | high |
| 7 | Multifidelity MC | ~3x | low |

Recommended pipeline: stratify on (g12, g23) with exact stratum masses (5), plus a frozen-pattern control variate (3), plus within-stratum preintegration along the active tie direction (1), plus RQMC on the outer variables. Each layer is unbiased and the gains multiply roughly.

## Sources
- [Griewank et al., smoothing by preintegration (arXiv:1712.00920)](https://arxiv.org/pdf/1712.00920)
- [Bayer, Ben Hammouda, Tempone, numerical smoothing (arXiv:2111.01874)](https://arxiv.org/pdf/2111.01874)
- [Gilbert, Kuo, Sloan, preintegration and monotonicity (arXiv:2112.11621)](https://ar5iv.arxiv.org/html/2112.11621)
- [Glasserman and Staum, Conditioning on one-step survival, Oper. Res. 2001](https://pubsonline.informs.org/doi/10.1287/opre.49.6.923.10018)
- [Sequential directional IS (arXiv:2202.05683)](https://arxiv.org/pdf/2202.05683)
- [Line sampling overview](https://en.wikipedia.org/wiki/Line_sampling)
- [Koutsourelakis et al. 2004, Part I](https://researchportal.hw.ac.uk/en/publications/reliability-of-structures-in-high-dimensions-part-i-algorithms-an/)
- [Gessner, Kanjilal, Hennig, AISTATS 2020](https://arxiv.org/pdf/1910.09328)
- [Genz and Bretz, Computation of MVN and t Probabilities](https://ebook.library.uitm.edu.my/Record/7113/TOC)
- [Boetius, Leue, Sutter, ICML 2025](https://proceedings.mlr.press/v267/boetius25a.html)
- [Tran et al., ProbStar / StarV](https://par.nsf.gov/biblio/10649230-starv-qualitative-andquantitative-verification-tool-forlearning-enabled-systems)
- [ReLU networks as random functions (arXiv:2503.22082)](https://arxiv.org/html/2503.22082v1)
- [Exact bounds for the output distribution of NNs (arXiv:2502.11672)](https://arxiv.org/pdf/2502.11672)
- [Smoothed ReLU verification hardness (arXiv:2607.14375)](https://papers.cool/arxiv/2607.14375)
- [Fitisone and Zhou, solid angle of polyhedral cones (arXiv:2304.11102)](https://arxiv.org/pdf/2304.11102)
- [Ribando 2006](https://scholarworks.uni.edu/facpub/2853)
- [Cousins and Vempala, Gaussian cooling (arXiv:1409.6011)](https://arxiv.org/pdf/1409.6011)
