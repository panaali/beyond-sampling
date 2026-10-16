# Physics-side methods for computing the accuracy of the 2nd-argmax ReLU RNN

Problem recap: accuracy = Gaussian measure of a union of polyhedral cones in R^10. The network is positively homogeneous and piecewise linear. Errors are about 4.7% of the mass, sitting in thin bands near ties of the top order statistics plus a region with at most one positive input. Our perturbative expansion around canonical inputs gives a bias of 0.59 then 0.25 pp, and the cost doubles per order. It blows up when a perturbed network's "typical" point is atypical.

Diagnosis in physics language: we expanded around a single saddle (the canonical point). The integrand is a sum of indicators with boundaries that are sharp but not smooth, so the series is asymptotic at best. The physics fixes are of three kinds:
(a) do the hard directions exactly and sample only the smooth ones (Rao-Blackwellize, which is "integrate out the fast mode");
(b) make the expansion a control variate rather than the answer;
(c) treat the error mass as a rare-event / large-deviation problem with nested or tilted measures.

All three keep the result unbiased, so a weight perturbation can raise the variance but can never produce a blow-up.

Ranking: promise x feasibility, best first.

---

## 1. Directional + line sampling: integrate the radius and the tie coordinate exactly (conditional MC)
**Citations.** Bjerager, "Probability integration by directional simulation", J. Eng. Mech. 114(8):1288 (1988). Ditlevsen, Melchers & Gluver, Comput. Struct. 36(2):355 (1990). Koutsourelakis, Pradlwarter & Schuëller, "Reliability of structures in high dimensions, Part I", Prob. Eng. Mech. 19:409 (2004) (line sampling). Papaioannou & Straub, "Combination line sampling" (2021).

**Idea.** Sample a direction or a hyperplane point, then integrate the 1-D remainder analytically along a line, using the 1-D Gaussian CDF between the exact limit-state crossings. When the line is aligned with the failure direction, the variance drops by orders of magnitude.

**Recipe.**
- *Radius is free.* By positive homogeneity, correctness depends only on x/|x|. Plain MC already exploits this implicitly, so it gives nothing new on its own. It does mean we can work on the sphere or on any 2-D plane through the origin.
- *Line in the tie direction.* For a sample x with top-two indices (a, b), and third index c, parametrize the line x(t) = x + t·(e_a − e_b)/√2. The network is piecewise linear in t. Track the activation pattern along t and compute every breakpoint exactly (each ReLU pre-activation is affine in t inside a region). This takes at most ~16×10 crossings and is cheap. Within each segment the argmax logit is decided by linear inequalities. Return the exact Gaussian mass of the correct segments: the conditional probability given the orthogonal complement.
- *Plane version.* Do the same in the 2-D plane spanned by (e_a − e_b) and (e_b − e_c), the two tie directions that dominate the errors. Use homogeneity so that only angles matter: the region in the plane is a union of wedges whose boundaries come from piecewise-linear root finding. The mass of each wedge is exact (angle/2π times the chi-squared-2 factor, which equals 1 for a cone through the origin when the plane passes through the origin; otherwise use an affine slice plus 1-D Gaussian integrals).
- *Low-positive-input region.* Use a line along the all-ones direction or along the largest coordinate.

**Why it fixes our problem.** Our perturbation theory is essentially a Taylor expansion of the flip threshold t*(rest) in the other coordinates. Line sampling instead computes t*(rest) exactly per sample, to all orders, and averages over "rest" by MC. The integrand becomes smooth: a CDF of the exact threshold, not an indicator. Variance per sample falls by roughly (band-width)^-1, with no truncation bias and no failure on atypical weights, because nothing is expanded. This is the direct non-perturbative upgrade of what we already have.

## 2. Use the perturbative estimate as a control variate (lattice-QCD / "improved estimator" style)
**Citations.** Standard control variates (e.g. Glasserman, *Monte Carlo Methods in Financial Engineering*, 2003, ch. 4). Lattice usage: Bali, Collins & Schäfer, "Effective noise reduction techniques for disconnected loops", CPC 181:1570 (2010). Multilevel: Giles, "Multilevel Monte Carlo methods", Acta Numerica 24:259 (2015).

**Idea.** If a cheap surrogate f̃ has a known exact mean and correlates with f, then estimating E[f − f̃] + E[f̃] removes most of the variance and none of the bias protection.

**Recipe.** Take the first-order perturbative classifier. Its flip thresholds are linear in the other inputs, so its "correct" region is a polyhedral set whose Gaussian mass is computable to high precision by Genz/Botev (item 4) or in closed form per order-statistic band. Take f̃(x) = 1[surrogate correct at x] or, better, the smooth surrogate probability from item 1's line integral under the linearized threshold. Run MC on the difference f − f̃, which is nonzero only where the surrogate and the network disagree (a tiny set). The truncation bias of 0.25–0.59 pp becomes a quantity *estimated* without bias, and its variance scales with the disagreement mass (≈ bias magnitude), not with p(1−p) ≈ 0.045. This is a 2-level MLMC with level 0 exact, and orders 1 and 2 can be added as further levels (Giles telescoping).

**Why it fixes our problem.** Slow convergence of the series no longer matters: each order only needs to reduce the variance of the next correction. On atypical perturbed weights the surrogate is poor, so the variance rises gracefully to that of plain MC, but the estimate stays unbiased. This removes the fragility.

## 3. Randomized-truncation debiasing of the perturbation series (Rhee–Glynn / von Neumann–Ulam)
**Citations.** Rhee & Glynn, "Unbiased estimation with square root convergence for SDE models", Oper. Res. 63(5):1026 (2015). McLeish, "A general method for debiasing a Monte Carlo estimator", MCMA 17 (2011). Beatson & Adams, "Randomized telescoping sums", ICML 2019 (arXiv:1905.07006).

**Idea.** Given approximations Y_0, Y_1, … → Y, draw a random level N and output Σ_{k≤N} (Y_k − Y_{k−1})/P(N ≥ k). The result is unbiased for the limit Y with finite variance, provided the increments' second moments decay faster than the cost grows.

**Recipe.** Let Y_k be the order-k perturbative per-sample correctness, or the order-k line-sampling threshold from item 1. Our numbers give a bias ratio of about 0.42 per order, and cost ×2 per order. Choose P(N ≥ k) ∝ 2^{-k·γ} with γ between log2(1/0.42²) ≈ 2.5 and 1 (cost), which gives finite variance and finite expected cost. Also add the "exact" level N = ∞ ≡ full network evaluation at a sample point, which is cheap here. The truncation to infinity is therefore well defined, and the sum is a coupled MLMC.

**Why it fixes our problem.** It turns a slowly converging *biased* series into an *unbiased* estimator with O(1/√cost) error. You never need to reach high order, and an atypical point simply produces a large but finite increment.

## 4. Exact Gaussian mass of each linear region: Genz / Botev minimax tilting, summed over gate patterns ("sum over switching histories")
**Citations.** Genz, "Numerical computation of multivariate normal probabilities", JCGS 1:141 (1992). Botev, "The normal law under linear restrictions: simulation and estimation via minimax tilting", JRSS-B 79:125 (2017) (arXiv:1603.04166; R `TruncatedNormal`, Python ports). Gessner, Kanjilal & Hennig, "Integrals over Gaussians under linear domain constraints", AISTATS 2020 (arXiv:1910.09328; LIN-ESS + Holmes–Diaconis–Ross, in BoTorch).

**Idea.** For a polyhedron {Ax ≤ 0}, Genz's sequential conditioning plus QMC gives the Gaussian mass with relative error ~1e-4 in 10-D in milliseconds. Botev's tilted version has vanishing relative error even for tiny masses. LIN-ESS/HDR handles masses down to 1e-10 or below via nested domains.

**Recipe.** This is the path-integral picture: accuracy = Σ over activation histories σ (10 steps × 16 units) and over the true rank permutation π of μ(cone_{σ,π} ∩ {correct}).
- Use the 10! permutation symmetry of the input distribution to fix π up to the network's asymmetry. The full sum over π is still needed, but each term is a cone.
- Enumerate the *dominant* histories by sampling (a few thousand samples reveal the histories that carry ~99% of the error mass).
- Integrate each error cone exactly with Botev/Genz QMC, which gives about 1e-4 relative error per cone.
- Treat the leftover histories (the "tail") with plain MC or item 1.

This gives a "head exact + tail sampled" estimator. The head is deterministic, with an error bar from QMC. The tail is unbiased.

**Why it fixes our problem.** No expansion point. The answer is exact per region, so the only error is from unseen regions, and that error is measurable. Feasibility depends on the number of distinct error cones. Collapse histories that give the same final linear map W_o·Π(D_t W_h)… restricted to the decision, i.e. group by the logit-difference linear forms, to keep the count manageable.

## 5. Subset simulation / Holmes–Diaconis–Ross nested domains for the thin error bands
**Citations.** Au & Beck, "Estimation of small failure probabilities in high dimensions by subset simulation", Prob. Eng. Mech. 16(4):263 (2001). Cérou, Del Moral, Furon & Guyader, "Sequential Monte Carlo for rare event estimation", Stat. Comput. 22:795 (2012). Gessner et al. 2020 (above).

**Idea.** Write P(F) = Π P(F_{i+1} | F_i) using nested level sets of a continuous "margin" score. Estimate each conditional by MCMC that stays inside F_i. Relative error grows only like log(1/P).

**Recipe.** Define the margin m(x) = logit_correct − max_other logit, divided by |x| for homogeneity. Errors are {m < 0}. Use levels m < τ_1 > τ_2 > … > 0 chosen adaptively, each with conditional probability 0.1–0.3. Use LIN-ESS within a fixed activation region (closed-form ellipse–hyperplane intersections), switching regions when crossings occur. Each error sub-type (tie of ranks 1–2, tie of ranks 2–3, the low-positive region) can be its own run (multichannel, item 7).

**Why it fixes our problem.** The error mass is ~0.047 overall, but the hard part is that it is concentrated in thin bands. Splitting spends samples near the boundary instead of in the 95% bulk. It is less helpful than items 1–2 here, because 0.047 is not that rare. It becomes important if we want to resolve sub-bands or very accurate networks.

## 6. Instantons / large deviations: FORM/SORM around the *most probable error point*, not the canonical point
**Citations.** Breitung, "Asymptotic approximations for multinormal integrals", J. Eng. Mech. 110:357 (1984). Hohenbichler & Rackwitz (1988), SORM improvement. Freidlin–Wentzell large-deviation theory, and in physics the "optimal fluctuation"/instanton method, e.g. Grafke & Vanden-Eijnden, "Numerical computation of rare events via large deviation theory", Chaos 29:063118 (2019). Siegmund, "Importance sampling in the Monte Carlo study of sequential tests", Ann. Stat. 4:673 (1976), for exponential tilting.

**Idea.** Rare-event mass is dominated by the minimum-norm point x* on the failure boundary (the instanton). A Laplace or SORM expansion around x*, using the boundary's curvature, gives the leading asymptotics. Mixture importance sampling centered at the x*'s gives asymptotically efficient MC.

**Recipe.** Errors sit near tie hyperplanes, so the "instanton" is the minimum-norm point on each piecewise-linear decision boundary facet, found by an LP or QP per activation region. On the unit sphere (homogeneity), each error component has a centroid direction. Build a mixture proposal: Gaussian shifted toward each facet's foot point, or (better) the von Mises–Fisher on the sphere, with weights from SORM estimates. This replaces our single "typical point" with *many* expansion points, one per error channel, found by optimization rather than assumed from order-statistic means.

**Why it fixes our problem.** Our blow-ups occurred exactly when the expansion point was atypical. Instantons are *computed* per network, so a perturbed network automatically gets new expansion points. Used as IS proposals rather than as a series, the method remains unbiased.

## 7. Multichannel adaptive importance sampling (VEGAS+, Kleiss–Pittau multichannel) aligned to tie coordinates
**Citations.** Lepage, "Adaptive multidimensional integration: VEGAS enhanced", J. Comput. Phys. 439:110386 (2021) (arXiv:2009.05112; `vegas` Python package). Kleiss & Pittau, "Weight optimization in multichannel Monte Carlo", CPC 83:141 (1994). Ohl, "Vegas revisited" (hep-ph/9806432). Normalizing-flow successor: Gao et al., "i-flow" / "Event generation with normalizing flows", PRD 101:076002 (2020).

**Idea.** VEGAS adapts a separable grid map, and VEGAS+ adds adaptive stratification. Multichannel sampling uses one map per singular structure, with optimized channel weights. Both concentrate samples where the integrand's variance lives.

**Recipe.** VEGAS needs separability, and our singularities are on the diagonals x_a = x_b, which is the classic failure mode. Fix this the way particle physicists do with propagator peaks: change variables per channel to *order-statistic spacings*. Sort x and use (x_(1), gap_12 = x_(1) − x_(2), gap_23, …), plus the permutation as a discrete channel variable. Errors live at small gap_12 or gap_23, which are now axis-aligned, so VEGAS grids resolve them. Integrate f − f̃ (item 2) rather than f. Run channels for (i) rank-1/2 tie, (ii) rank-2/3 tie, (iii) ≤1 positive input, with Kleiss–Pittau weight optimization.

**Why it fixes our problem.** It needs no expansion: the adaptation is learned from the actual network, so it is robust across perturbed weights. Spacing coordinates turn our "thin bands" into "near-boundary strips" in 1-D marginals, where VEGAS is strongest. Expect a 5–50× variance reduction over plain MC. It is easy to implement with `vegas`.

## 8. Resummation of the existing series: Padé / Shanks / Richardson, and "RG-improved" expansion point
**Citations.** Baker & Graves-Morris, *Padé Approximants* (1996). Shanks, J. Math. Phys. 34:1 (1955). Bender & Orszag, *Advanced Mathematical Methods*, ch. 8 (1978). Kleinert & Schulte-Frohlinde, *Critical Properties of φ⁴ Theories*, variational perturbation theory (2001).

**Idea.** A series with geometric-like convergence (our bias ratio is ~0.42) can be accelerated: Aitken/Shanks on partial sums, or Padé in an auxiliary coupling. Variational or RG improvement picks the expansion point (here the "canonical point") to optimize convergence, rather than fixing it a priori (principle of minimal sensitivity).

**Recipe.**
- *Shanks/Aitken.* Partial sums S0, S1, S2 → S2 − (S2 − S1)²/(S2 − 2S1 + S0). For our numbers this gives roughly −0.07 pp residual for the cost of no extra orders.
- *Coupling λ.* Introduce λ scaling the deviation of the non-tied inputs from their canonical values. Expand in λ, then Padé[1/1] at λ = 1.
- *PMS.* Instead of expected order statistics, choose the expansion point per rank pattern to make the 2nd-order term vanish (minimal sensitivity), or expand around the instanton of item 6.

**Why it fixes our problem.** It is cheap, but it is a heuristic. Use it only to improve the control variate in item 2, never as the final answer, because a resummed asymptotic series still carries uncontrolled bias on atypical weights.

## 9. Expectation propagation / cavity (TAP) approximation for Gaussian mass of polyhedra
**Citations.** Cunningham, Hennig & Lacoste-Julien, "Gaussian probabilities and expectation propagation" (arXiv:1111.6832, 2011). Opper & Winther, "Expectation consistent approximate inference", JMLR 6:2177 (2005) (TAP/cavity connection). Mézard & Montanari, *Information, Physics, and Computation* (2009).

**Idea.** Approximate a truncated Gaussian on {a_i·x ≤ b_i} by a Gaussian that matches moments factor by factor (cavity method). The normalizer estimates the polyhedron's mass. It is excellent for boxes, but on general polyhedra it is "often accurate but can be arbitrarily wrong".

**Recipe.** For each dominant error cone (item 4), run EP to get an estimate of its mass and a Gaussian approximation of the conditional. Use the EP Gaussian as the proposal for importance sampling inside the cone (EP-IS), or as the shift in Botev tilting. Treat the RNN unrolled over 10 steps as a factor graph with ReLU-gate factors. Belief propagation / generalized AMP over time (as in multilayer GAMP, Manoel et al. ISIT 2017) can propagate Gaussian messages through the gated linear recurrence for a fixed gating history.

**Why it fixes our problem.** It gives fast deterministic per-cone estimates, which are useful as proposals and control variates. On its own it is not trustworthy enough (the paper's own caveat), so it is ranked low as a final estimator.

## 10. Moment closure / Fokker–Planck propagation through the gated recurrence (cumulant expansion)
**Citations.** Frey & Hinton, "Variational learning in nonlinear Gaussian belief networks", Neural Comput. 11:193 (1999) (rectified-Gaussian moment matching). Wu et al., "Deterministic variational inference for robust Bayesian neural networks", ICLR 2019 (closed-form ReLU moments). Kuehn, "Moment closure – a brief review" (2016, arXiv:1505.01987). Pilipovsky et al., "Probabilistic verification of ReLU neural networks via characteristic functions", L4DC 2023 (arXiv:2212.01544).

**Idea.** Propagate the distribution of h_t forward using exact ReLU moment formulas for Gaussian pre-activations, closing at the second cumulant (or with a Gaussian mixture). The characteristic-function approach propagates the full law.

**Recipe.** Condition on the rank pattern of x, or on the gap variables of item 7, so the input is near-deterministic up to small fluctuations. Propagate mean and covariance of h_t through 10 steps using the damped rotating modes as a near-linear backbone (most gates stay fixed), and compute P(correct logit wins) from the final Gaussian or mixture logit-difference law. Our perturbation theory is a crude version of this. Moment closure resums all orders in fluctuation amplitude for the fixed-gate part and treats gate flips as the only non-Gaussian effect.

**Why it fixes our problem.** It partially resums, but closure errors are uncontrolled, especially at the tie boundaries that matter. Use it at most as a control variate.

## 11. Diagrammatic Monte Carlo: stochastically sample the expansion itself
**Citations.** Prokof'ev & Svistunov, "Polaron problem by diagrammatic quantum Monte Carlo", PRL 81:2514 (1998). Prokof'ev & Svistunov, "Bold diagrammatic Monte Carlo", PRB 77:125101 (2008). Van Houcke et al., Nature Phys. 8:366 (2012).

**Idea.** Rather than computing each order deterministically (with cost ×2 per order), sample *order and diagram* jointly by Metropolis, so high orders cost the same per sample as low ones. Bold lines resum classes of diagrams.

**Recipe.** Index the "diagram" as a set of gate flips relative to the canonical activation history (each flip = a vertex). The weight is the Gaussian mass of the region with exactly those flips, which can be estimated per configuration. A Metropolis walk over flip sets with add/remove updates gives the sum over switching histories. The "bold" version uses the line-sampled exact threshold of item 1 as the dressed propagator.

**Why it fixes our problem.** It conceptually addresses the "cost doubles per order" issue. Items 1–3 do the same job much more simply for a 10-D integral, so this is rated low on feasibility and is mostly a framing.

---

## Overall ranking
| Rank | Method | Promise | Feasibility | Fixes bias? | Fixes fragility? |
|---|---|---|---|---|---|
| 1 | Line/plane conditional MC along tie directions (item 1) | very high | high (exact PL root-finding along a line) | yes (no expansion) | yes |
| 2 | Perturbative surrogate as control variate / MLMC (item 2) | very high | high | yes (unbiased) | yes (degrades to MC) |
| 3 | Rhee–Glynn randomized truncation of the series (item 3) | high | high | yes | yes |
| 4 | Exact per-cone Genz/Botev QMC over dominant gate histories + sampled tail (item 4) | high | medium (cone count) | yes | yes |
| 5 | VEGAS+/multichannel in order-statistic spacing coordinates (item 7) | medium-high | high (`vegas` package) | n/a (MC) | yes |
| 6 | Instanton / most-probable-error-point mixture IS (item 6) | medium | medium | yes, as IS | yes |
| 7 | Subset simulation / HDR + LIN-ESS (item 5) | medium | medium | yes | yes |
| 8 | Shanks/Padé/PMS resummation (item 8) | low-medium | trivial | partially | no |
| 9 | EP / cavity (item 9) | low-medium | medium | no | no |
| 10 | Moment closure (item 10) | low | medium | no | no |
| 11 | Diagrammatic MC (item 11) | low | low | yes | yes |

Suggested combination: item 1 (exact 1-D or 2-D integration across the tie band) applied to the residual f − f̃ of item 2, with f̃ the 1st/2nd-order surrogate whose mean is computed exactly by Genz/Botev. Stratify by rank pattern or spacing channels (item 7). Every component is unbiased, and each layer multiplies the variance reduction.
