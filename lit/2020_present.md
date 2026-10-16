# Literature 2020–present for the AlgZoo "2nd-argmax" RNN accuracy-estimation problem

Problem recap: bias-free ReLU RNN (16 hidden, 10 steps, 432 params), x ~ N(0, I_10), target = argpos of
2nd-largest input, true acc ~95.3%. Goal: mechanistic (no input sampling) estimate whose MSE over
perturbed weights is at most that of Monte Carlo (MC) at equal compute.

Structural facts used below:
- Network is positively homogeneous and piecewise linear, so accuracy is the Gaussian measure of a union of
  polyhedral cones. That equals the normalized solid angle on S^9, and the radius integrates out exactly.
- Error mass (~4.7%) lives in thin wedges near ties of the top order statistics, plus the region with at most
  one positive input.
- Gates are mostly a function of the rank order / record history.

Constraint respected: no ARC AlgZoo posts or papers after Jan 2026 were read. ARC general work (heuristic
arguments, cumulant propagation, "competing with sampling", Nov 2025) is included.

Ranked by promise for this specific problem.

---

## 1. Preintegration ("smoothing by preintegration") + randomized QMC
**Citations.**
- Gilbert, Kuo, Sloan (2023), "Analysis of preintegration followed by quasi-Monte Carlo integration for
  distribution functions and densities", SIAM J. Numer. Anal. 61(1). arXiv:2112.10308
- Gilbert, Kuo, Sloan (2022), "Preintegration is not smoothing when monotonicity fails", in *Advances in
  Modeling and Simulation*, Springer, pp. 169–191. arXiv:2112.11621
- Background: Griewank, Kuo, Leövey, Sloan (2018), "High dimensional integration of kinks and jumps —
  smoothing by preintegration", arXiv:1712.00920

**Method.** To integrate a discontinuous indicator 1{phi(x) > 0} against a Gaussian, first integrate out one
well-chosen variable analytically or exactly in 1-D. The remaining (d-1)-dimensional integrand becomes smooth,
so (randomized) QMC gets close to O(1/N) convergence instead of O(N^{-1/2}). The 2022 paper warns that
smoothness needs monotonicity in the preintegrated variable. Without it, square-root singularities appear,
but the estimator stays unbiased and has lower variance.

**Gaps.** (4) deterministic or variance-reduced MC; (3) boundary-concentrated probabilities.

**How to apply.** This is the cheapest strong baseline and is natural here.
- Homogeneity already gives exact radial integration.
- Go further. Pick a random 2-D plane through the origin (a great circle on S^9). Along that circle, every
  pre-activation is a·cosθ + b·sinθ, so all ReLU switch points on the circle can be found exactly by
  tracking the 10-step recursion piecewise. Correctness is then a union of arcs, and its exact length gives
  the conditional probability with zero variance in two dimensions.
- Preintegrating along a direction that crosses the tie bands, such as e_{(1)} − e_{(2)}, the difference of
  the top two coordinates, is close to monotone, so it smooths well.
- Feed the outer 8 dimensions to scrambled Sobol points. Expect large MSE gains because the error is
  boundary mass.

## 2. Line sampling / combination line sampling (structural-reliability rare-event methods)
**Citation.** Papaioannou & Straub (2021), "Combination line sampling for structural reliability analysis",
*Structural Safety* 88:102025, doi:10.1016/j.strusafe.2020.102025. Preprint:
https://mediatum.ub.tum.de/doc/1612140/c0hrm4jrvecjwf7ti60sgzst0.2021_Papaioannou_CLS.pdf

**Method.** Choose an important direction α. For each sample of the orthogonal complement, solve the 1-D root
problem along α and add Φ(−root) exactly. Combination LS adaptively updates α and combines the per-direction
estimators with improved weights. This is conditional MC that removes the variance along the direction normal
to the failure boundary.

**Gaps.** (3), (4).

**How to apply.** Failure here is not one half-space but a union of thin wedges at the ties x_(1) ≈ x_(2) and
x_(2) ≈ x_(3). Use a mixture of line directions:
- Stratify on which coordinates are the top three (720 ordered triples, or exploit symmetry under weight
  perturbation).
- Use α = (e_i − e_j)/√2 for the relevant tie.
- Along each line, the network is piecewise linear, so roots are exact breakpoints rather than numerical
  solves.

This is item 1 specialized to rare-event geometry. The tie band's width is set by the network's margins.

## 3. Cumulant propagation / "competing with sampling" (ARC general program, pre-2026)
**Citations.**
- Christiano, Neyman, Xu (2022), "Formalizing the presumption of independence", arXiv:2211.06738
- ARC blog (Nov 18, 2025), "Competing with sampling", https://www.alignment.org/blog/competing-with-sampling/
  (defines the matching-sampling criterion E_c[(G−E_x M)^2] ≤ ε²·E_c[Var_x M] at O(1/ε²·Time(M)) compute;
  cumulant propagation with Hermite-polynomial activations; random half-space intersection warm-up)
- Wu, Lecomte, Winer, Robinson, Hilton, Christiano (2026), "Estimating the expected output of wide random MLPs
  more efficiently than sampling", arXiv:2605.05179. Code:
  https://github.com/alignment-research-center/mlp_cumulant_propagation. This is about random MLPs, not
  AlgZoo; only its abstract was read.
- Christiano, Hilton, Neyman et al. (2024), "Towards a law of iterated expectations for heuristic estimators",
  arXiv:2410.01290

**Method.** Represent each layer's activation distribution by low-order joint cumulants, or by Hermite
coefficients. Push them through by expanding ReLU in Hermite polynomials and using Wick/cumulant rules,
truncating at order d with n^d ≈ 1/ε². Truncation is principled because higher cumulants are presumed
independent or zero unless an argument says otherwise.

**Gaps.** (5), and directly the evaluation criterion.

**How to apply.** Width 16 is far from the wide regime, and the hidden state is strongly non-Gaussian, being
order-statistic driven. So naive cumulant propagation of h_t will be poor. Use the framework at the level of
the mechanism instead:
- Condition on the discrete "record history" or rank-pattern variables, which you enumerate.
- Inside each stratum, propagate cumulants of the few continuous quantities that matter, such as the gaps
  between the running top-2 and the current input.
- Finish with exact Gaussian integrals.

The ε²·Var criterion and compute accounting give the right yardstick for every method here.

## 4. Gessner, Kanjilal, Hennig (2020): integrals over Gaussians under linear domain constraints (LIN-ESS + HDR)
**Citation.** AISTATS 2020, PMLR 108:2764–2774. https://proceedings.mlr.press/v108/gessner20a.html,
arXiv:1910.09328. Code (Python): https://github.com/alpiges/LinConGauss

**Method.** Estimates the Gaussian mass of a polytope {x: Ax + b ≥ 0}, even a tiny one, with
Holmes–Diaconis–Ross nested subset simulation. Constraints are relaxed by a shift that shrinks level by level,
and each conditional probability is estimated by rejection-free elliptical slice sampling. The ellipse/polytope
intersections are closed form, which is exactly what our cones are.

**Gaps.** (1), (3).

**How to apply.** For each activation pattern, or merged pattern, that produces an error, the error cone is
{x: A_p x ≥ 0} with homogeneous constraints. Use LinConGauss to get its mass with low relative error even
when it is small, then sum over the dominant error patterns.

Better: use its analytic ellipse–polytope intersection as the inner kernel of a deterministic arc-length
computation, as in item 1. An ellipse through the origin-centered Gaussian is a great-circle slice. The
nested-shift idea also yields a boundary-layer expansion: P(margin < s) as a function of s.

## 5. Probabilistic verification by branch and bound (PV)
**Citation.** Boetius, Leue, Sutter (2025), "Solving probabilistic verification problems of neural networks
using branch and bound", ICML 2025, PMLR 267. arXiv:2405.17556,
https://proceedings.mlr.press/v267/boetius25a.html. Code is linked from the paper (`probspecs`).

**Method.**
- Computes certified lower and upper bounds on P_{x~D}[property of NN(x)] for input distributions such as a
  Gaussian or a product distribution.
- Recursively splits the input domain or ReLU states, and uses linear-relaxation bound propagation
  (CROWN-style) to certify sub-boxes as fully satisfying or fully violating.
- Probability mass of the certified pieces is accumulated exactly. The undecided remainder gives the gap.
- Sound, and complete under mild conditions. Runtimes drop from tens of minutes to tens of seconds.

**Gaps.** (2), (5), and partly (1).

**How to apply.**
- Unroll the RNN into a 10-layer, 16-wide MLP with shared weights.
- Split by rank order first, i.e. by the (Weyl-chamber) simplex cones {x_{σ1} > … > x_{σ10}}. Each has Gaussian
  measure 1/10!, and within it most gates are fixed.
- B&B then only refines near ties and in the region with few positive inputs.

Even if the bounds are loose, the undecided mass tells you where to spend sampling or exact computation. It
is a natural "where's the residual" tool for a hybrid estimator.

## 6. Linear-region / polyhedral-complex enumeration
**Citations.**
- Berzins (2023), "Polyhedral complex extraction from ReLU networks using edge subdivision", ICML 2023, PMLR
  202:2234–2244. arXiv:2306.07212. Code: https://github.com/arturs-berzins/relu_edge_subdivision
- Masden (2022), "Algorithmic determination of the combinatorial structure of the linear regions of ReLU
  neural networks", arXiv:2207.07696, with code on GitHub. Gives the full face structure from vertices plus
  sign vectors, polynomial in the number of neurons.
- Humayun, Balestriero, Balakrishnan, Baraniuk (2023), "SplineCam: exact visualization and characterization of
  deep network geometry and decision boundaries", CVPR 2023. Exact region partitions on 2-D slices.
- Context on hardness: "The computational complexity of counting linear regions in ReLU neural networks"
  (NeurIPS 2025, arXiv:2505.16716).

**Method.** Edge subdivision tracks sign vectors on the edges of the current complex and splits only edges
that a new neuron's hyperplane crosses. It is GPU-vectorized and handles millions of cells. Masden recovers
the entire canonical polyhedral complex.

**Gaps.** (2).

**How to apply.**
- Full enumeration in R^10 with 160 neurons is likely too big, but the domain is a cone, so you can work on
  S^9 or on a 9-D affine slice x_{10} = 1.
- Restricting to a rank-order chamber cuts it drastically.
- SplineCam/Berzins on 2-D planes is exactly the inner routine needed for item 1's exact arc-length
  computation.
- Use region counts per chamber to test how well "gates = f(record history)" holds, and to find the exceptions.

## 7. Exact / series solid-angle formulas for polyhedral cones
**Citations.**
- Fitisone & Zhou (2023), "Solid angle measure of polyhedral cones", arXiv:2304.11102. A SageMath
  implementation is mentioned in Zhou's SageDays talk.
- Fitisone (2024), PhD thesis, "Solid angle measure approximation methods for polyhedral cones", Univ. of
  Kentucky.
- Builds on the Ribando (2006) / Aomoto (1977) hypergeometric series.

**Method.** The normalized solid angle of a simplicial cone, which equals the Gaussian measure of the cone, has
a multivariate power series in the off-diagonal Gram entries. It converges when an associated matrix is
positive definite. They decompose arbitrary cones into simplicial cones satisfying that condition, and use
tridiagonal decompositions to cut the number of series variables.

**Gaps.** (1), (6).

**How to apply.**
- Each linear region is a polyhedral cone, so its Gaussian mass is its solid angle.
- Rank-order chambers are simplicial cones with Gram entries 0 or ±1/2. The exact gap/tie wedges are
  simplicial cones that are slight perturbations of those.
- For perturbed weights, the series in small perturbations of a known cone gives the derivative structure.
  This is a route to perturbative, deterministic estimates of how accuracy moves with weight noise.
- In 10-D, a pure series is expensive, so use it for the few dominant error cones and use Genz-style QMC
  (item 8) for the rest.

## 8. Fast multivariate-normal (orthant/polytope) probabilities: separation of variables + QMC + reordering
**Citations.**
- Cao, Genton, Keyes, Turkiyyah (2021), "Exploiting low rank covariance structures for computing
  high-dimensional normal and Student-t probabilities", *Statistics and Computing* 31:2. arXiv:2003.11183.
  R package `tlrmvnmvt`.
- Background (pre-2020, still the workhorse): Genz & Bretz SOV algorithm (`mvtnorm`, `scipy.stats.
  multivariate_normal.cdf`); Botev (2017, JRSS-B), "minimax tilting" (`TruncatedNormal`, R/Matlab/Python),
  which gives accurate small probabilities with vanishing relative error.

**Method.** Write P(a ≤ Lz ≤ b) via a Cholesky factor and integrate one variable at a time with conditional
1-D normal CDFs. This is itself preintegration in every coordinate. Run randomized QMC over the remaining
uniforms, and reorder variables to put the most constrained ones first. Botev adds an exponential tilt that is
optimal in a minimax sense for rare polytopes.

**Gaps.** (1), (6), (3).

**How to apply.** Any cone defined by ≤10 linear constraints in R^10 is an MVN orthant probability of
dimension ≤10: P(A x ≥ 0) with A x ~ N(0, A Aᵀ). This takes milliseconds with relative error ~1e-4 by QMC.
For more than 10 constraints, use Botev's tilting with polytope constraints, or Genz on a chosen 10-constraint
subset plus correction. This is the engine for summing over enumerated error cones (items 5, 6, 9).

Order-statistic probabilities are also MVN orthants. Examples: P(x_i is 2nd largest and x_(1) − x_(2) < δ),
and P(at most one positive input) = 11/1024 exactly. That makes the "ideal algorithm minus band" control
variate computable exactly.

## 9. Compact proofs of model performance via mechanistic interpretability
**Citation.** Gross, Agrawal, Kwa, Ong, Yip, Gibson, Noubir, Chan (2024), NeurIPS 2024. arXiv:2406.11779.
Code: https://github.com/JasonGross/guarantees-based-mechanistic-interpretability

**Method.** For 151 one-layer transformers on **Max-of-K**, they convert reverse-engineered circuits into
computer-checked lower bounds on accuracy. Inputs are grouped by (max token, largest non-max token, query),
and worst-case or averaged bounds are taken per group. They map a frontier of proof length against bound
tightness. Their key obstacle is "compounding structureless noise".

**Gaps.** (7), (2).

**How to apply.** This is the closest published analogue: an order-statistic task, a tiny model, and accuracy
deduced from weights. Borrow the methodology:
1. Partition input space by the mechanistically relevant discrete variables: which index is max, which is 2nd,
   and the record times.
2. Within each part, bound or compute the accuracy using the mechanism.
3. Measure "explanation length" vs. estimate error.

Their discussion of how residual noise destroys tightness tells you which terms must be computed exactly
(the tie bands) and which can be treated as noise. The difference is that we want an unbiased estimate under
weight perturbations, not a bound.

## 10. Exact and approximate moment matching through ReLU layers
**Citations.**
- Wright, Nakahira, Moura (2024), "An analytic solution to covariance propagation in neural networks",
  AISTATS 2024, PMLR 238:4087–4095. arXiv:2403.16163. Closed-form full covariance of ReLU/Heaviside outputs
  for jointly Gaussian pre-activations.
- Kuang & Lin (2026), "Exact Gaussian moment matching for residual networks: a second-order method",
  arXiv:2601.22307. Exact mean/covariance matching for ReLU, Heaviside, probit and GeLU, with an error bound.
  Code is in the arXiv ancillary files (`neural-uncertainty`).
- Petersen, Mishra, Kuehne, Borgelt, Deussen, Yurochkin (2024), "Uncertainty quantification via stable
  distribution propagation", ICLR 2024.
- Krapf, Hagn, Miethaner, Schiller, Luttner, Heinrich (2024), "Piecewise linear transformation — propagating
  aleatoric uncertainty in neural networks", AAAI 2024, doi:10.1609/aaai.v38i18.30029. Sample-free
  propagation exploiting PWL region structure, with error bounds.

**Method.** Assumed density filtering: after each layer, project the hidden state's law onto a Gaussian by
matching first and second moments. ReLU gives closed-form E and Cov via bivariate-normal orthant formulas.
Heaviside gives P(both on) = 1/4 + arcsin(ρ)/2π.

**Gaps.** (5).

**How to apply.** A single Gaussian for h_t will be badly wrong, because h_t is a piecewise function of order
statistics. Use these formulas as **building blocks inside a mixture/ADF scheme conditioned on the discrete
mechanism state**: one Gaussian per (record-history, gate-pattern) component, with each component's weight
updated by exact bivariate/MVN orthant probabilities at every step.

The final step is P(argmax logits = target | component). That is a 9-constraint MVN orthant probability in
the logit differences, computed by item 8. Krapf et al.'s PLT is the "exact" end of this spectrum: per-region
pushforward.

## 11. Rare-event estimation for neural networks: importance splitting and gradient-informed IS
**Citations.**
- Tit, Furon, Rousset (2021), "Efficient statistical assessment of neural network corruption robustness",
  NeurIPS 2021.
- Tit, Furon, Rousset (2023), "Gradient-informed neural network statistical robustness estimation", AISTATS
  2023, PMLR 206.
- Wu & Hilton (2025), "Estimating the probabilities of rare outputs in language models", ICLR 2025,
  arXiv:2410.13211. Compares importance sampling (ITGIS/MHIS) with activation extrapolation; IS wins.
- Background: Webb, Rainforth, Teh, Kumar (2019), "A statistical approach to assessing neural network
  robustness", ICLR 2019, adaptive multilevel splitting.

**Method.** Adaptive multilevel splitting writes P(score > 0) as a product of conditional probabilities over
increasing levels, with MCMC moves between levels. The gradient-informed version uses ∇score to build better
proposals or a first-order reliability approximation. Wu & Hilton's IS biases inputs toward the rare output
using gradients or logit information.

**Gaps.** (3), (4).

**How to apply.** The error rate of 4.7% is not tiny, so splitting is overkill globally. It becomes useful for
estimating *each stratum's* error, and for the "≤1 positive input" region. There the base probability is
11/1024 and the conditional error may be large. Gradient-informed proposals map onto our setting as
importance-sampling densities that concentrate on |x_(1) − x_(2)| < c and |x_(2) − x_(3)| < c. The
order-statistic gap densities are known in closed form.

## 12. Counting / quantifying unsafe inputs (#DNN-verification) and ProbStar
**Citations.**
- Marzari, Corsi, Cicalese, Farinelli (2023), "The #DNN-Verification problem: counting unsafe inputs for deep
  neural networks", IJCAI 2023. arXiv:2301.07068. Exact count (#P-complete) plus a randomized approximation
  with a confidence bound.
- Tran et al. (2023), "Probabilistic star temporal logic and its application" (HSCC 2023) and the StarV tool.
  ProbStar sets are star sets with Gaussian predicate variables, giving exact or over-approximate violation
  probabilities.

**Gaps.** (2), (5).

**How to apply.** Exact star-set reachability through the unrolled RNN produces explicit polytopes, one per
gate sequence, for the error region. ProbStar then attaches Gaussian probability to each.

Run this only inside high-error rank chambers, where the number of stars stays manageable. It is a
verification-grade cross-check on any mechanistic estimate.

## 13. Distilling tiny RNNs into exact algorithms (mech-interp of algorithmic RNNs)
**Citations.**
- Michaud, Liao, Lad, Liu, Mudide, Loughridge, Guo, Kheirkhah, Vukelić, Tegmark (2024), "Opening the AI black
  box: distilling machine-learned algorithms into code", *Entropy* 26:1046. arXiv:2402.05110 (MIPS).
- Related: Nanda et al. (2023), "Progress measures for grokking via mechanistic interpretability", ICLR 2023.
  Yip, Agrawal, Chan, Gross (2024), "Modular addition without black-boxes: compressing explanations of MLPs
  that compute numerical integration" (arXiv:2412.03773). That paper analyzes ReLU MLP mechanisms
  quantitatively, with error terms, which is in spirit close to what's needed.

**Method.** MIPS normalizes and simplifies a trained RNN, discretizes hidden states with an integer
autoencoder into a finite-state machine, then recovers symbolic update rules by symbolic regression. Yip et
al. show how to turn a ReLU mechanism into an analytic approximation plus bounded error terms.

**Gaps.** (7).

**How to apply.**
- Discretize h_t into its gate pattern plus a few continuous "register" values: the running max, the running
  2nd max, and their gap.
- Check that the gate pattern is a deterministic function of the record history. MIPS-style automaton
  extraction does exactly this.
- The extracted automaton defines the strata for items 1–5, 8 and 10. The continuous registers give the
  low-dimensional linear functionals whose Gaussian laws you integrate exactly.

---

## Suggested combined pipeline (synthesis)
1. **Stratify** by mechanism state. Use the top-3 rank identities plus record history (item 13); each
   stratum's base probability is exact (item 8).
2. **Within a stratum**, the "ideal" algorithm is correct by construction. Express the network's error as
   P(logit margin < 0), which for fixed gates is linear in x. That gives an MVN orthant probability (item 8),
   or a few such probabilities summed over the handful of gate patterns that occur near ties (items 5, 6).
3. **Residual** strata where gates are not determined by ranks: use preintegrated or line-sampled RQMC
   (items 1, 2), with arcs computed exactly along 2-D slices (item 6). Allocate compute by an estimate of
   residual variance.
4. **Evaluate** with ARC's matching-sampling criterion: MSE ≤ ε²·Var at matched FLOPs (item 3), across
   weight perturbations.
5. **Perturbative check**: the derivative of cone measures in the weight perturbation (item 7, or a
   Gaussian-surface-area / boundary-flux formula) predicts how accuracy moves across perturbed copies. This
   can serve as a control variate.

## Sources
- https://arxiv.org/abs/2112.10308 ; https://arxiv.org/pdf/2112.11621 ; https://arxiv.org/pdf/1712.00920
- https://mediatum.ub.tum.de/doc/1612140/c0hrm4jrvecjwf7ti60sgzst0.2021_Papaioannou_CLS.pdf
- https://arxiv.org/abs/2211.06738 ; https://www.alignment.org/blog/competing-with-sampling/ ; https://arxiv.org/abs/2605.05179 ; https://arxiv.org/html/2410.01290v1
- https://proceedings.mlr.press/v108/gessner20a.html ; https://arxiv.org/pdf/1910.09328
- https://proceedings.mlr.press/v267/boetius25a.html ; https://www.arxiv.org/abs/2405.17556
- https://arxiv.org/pdf/2306.07212 ; https://arxiv.org/pdf/2207.07696 ; https://arxiv.org/html/2505.16716v1
- https://arxiv.org/pdf/2304.11102 ; https://uknowledge.uky.edu/math_etds/109
- https://arxiv.org/pdf/2003.11183 ; https://search.r-project.org/CRAN/packages/tlrmvnmvt/tlrmvnmvt.pdf
- https://arxiv.org/pdf/2406.11779
- https://arxiv.org/pdf/2403.16163 ; https://arxiv.org/abs/2601.22307 ; https://ojs.aaai.org/index.php/AAAI/article/view/30029
- https://proceedings.neurips.cc/paper/2021/hash/4d215ab7508a3e089af43fb605dd27d1-Abstract.html ; https://arxiv.org/pdf/2410.13211
- https://www.ijcai.org/proceedings/2023/25 ; https://arxiv.org/pdf/2301.12001
- https://arxiv.org/abs/2402.05110

Note: code URLs for LinConGauss, guarantees-based-mechanistic-interpretability and Boetius' `probspecs` are
from memory or paper pointers and were not individually visited. Verify before relying on them.
