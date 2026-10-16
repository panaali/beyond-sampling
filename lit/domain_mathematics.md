# Methods from across mathematics for computing the accuracy of the 2nd-argmax ReLU RNN

The problem: we need acc = P_{x~N(0,I_10)}[argmax W_o h_10(x) = position of 2nd-largest x_i]. The network is
piecewise linear (PL) and positively homogeneous, so acc is a sum of normalized solid angles of polyhedral cones.
The target is constant on each chamber of the braid arrangement A_9, which has 10! chambers. The 4.7% error mass sits in thin
wedges next to the tie walls x_(1)=x_(2) and x_(2)=x_(3).

Baseline to beat: plain Monte Carlo (MC) with N samples has standard error sqrt(.953*.047/N) ≈ 0.21/sqrt(N).
Every estimator that sends a sample into the indicator pays this Bernoulli variance. The methods below avoid it by
**integrating the discontinuity exactly** in one or a few directions, by **stratifying on the braid chamber**, or by
computing cone measures with deterministic or low-variance cubature.

One fact is used throughout:

**(F) Sorted-Gaussian / chamber factorization.** Write x = σ·y, where y_1 > … > y_10 are the order statistics and
σ ∈ S_10. Then σ is uniform and independent of y, and y has density 10!·∏φ(y_i) on the Weyl chamber. Use gap
coordinates g_k = y_k − y_{k+1} ≥ 0 (k=1..9) and a free location t = y_10. Each chamber is then the orthant
{g ≥ 0} × R, and the label is the constant σ(2). So

acc = (1/10!) Σ_σ P_y[ net(σ·y) = σ(2) ]  =  E_σ E_y [ 1{net(σ·y)=σ(2)} ],

and all error mass sits at small g_1 or small g_2. The integrand is Gaussian in (g,t) with a known
tridiagonal-precision structure, and inside one chamber the gating pattern is nearly constant (our empirical finding).

---

## Ranked list (promise × feasibility)

### 1. Pre-integration (conditional MC/QMC) across the tie direction, done exactly with PL breakpoint walking
**Refs:** Griebel, Kuo, Sloan, "The smoothing effect of integration in R^d and the ANOVA decomposition", *Math. Comp.* 82 (2013)
([pdf](https://ins.uni-bonn.de/media/public/publication-media/GKS09-smoothing.pdf)). Griewank, Kuo, Leövey, Sloan,
"High dimensional integration of kinks and jumps — smoothing by preintegration", *JCAM* 2018 ([arXiv:1712.00920](https://arxiv.org/abs/1712.00920)).
Gilbert, Kuo, Sloan, "Preintegration is not smoothing when monotonicity fails" ([arXiv:2112.11621](https://arxiv.org/abs/2112.11621)).
Z. He, conditional QMC ([arXiv:1708.09512](https://arxiv.org/abs/1708.09512)).

**Idea:** Take an indicator 1{φ(x)>0} with a jump. Integrate out one variable x_j analytically:
P(x_j ∈ {s: φ(x_{-j}, s) > 0}) is a sum of Φ-differences. The resulting function of the other d−1 variables is
smooth (in mixed Sobolev spaces when φ is monotone in x_j). QMC and sparse grids then reach close to O(N^{-1}) instead of
O(N^{-1/2}).

**Recipe:**
- Use (F). Sample (σ, g_2..g_9, t) with randomized QMC: scrambled Sobol for the continuous variables, stratified for σ.
- Integrate g_1 ∈ [0,∞) exactly. Along the ray, the network is PL in g_1. Walk the line: start at g_1=0, compute all
  160 pre-activations as affine functions of g_1, find the next ReLU sign flip, update, and repeat. This gives an exact
  breakpoint list (typically a handful of breakpoints) and the exact set where argmax = σ(2).
- Weight that set by the exact conditional density of g_1 given the rest. In gap coordinates this is a 1-D Gaussian
  restricted to g_1 ≥ 0, so the contribution is a closed-form sum of Φ-differences.
- Optionally pre-integrate a second direction (g_2, or t) the same way. This gives polygons in 2-D, integrated with
  the bivariate normal CDF.

**Cost:** about (1 + #breakpoints) forward passes of 160 affine evaluations per sample. That is roughly 3–10× an MC sample.

**Why it beats MC:** The Bernoulli noise from the thin wedge disappears, because the wedge width is integrated exactly.
What remains is the variance of a smooth conditional probability, which is far smaller, and QMC rates apply. Monotonicity
in g_1 is plausible: growing the top gap should only help. When monotonicity fails, the method stays unbiased and only
loses some of the smoothing.

**Risk:** In rare places the error depends on g_2 or g_3 rather than g_1. Mitigate by pre-integrating along the
direction of the nearest tie wall, or along both g_1 and g_2.

### 2. Exact great-circle / 2-plane slicing (directional sampling, Crofton-style Rao–Blackwellization)
**Refs:** I. Deák, "Three digit accurate multiple normal probabilities", *Numer. Math.* 35 (1980). Spherical MC for MVN
probabilities ([arXiv:1309.3386](https://arxiv.org/abs/1309.3386)). Crofton / spherical kinematic formulas, e.g.
Amelunxen & Lotz, "Intrinsic volumes of polyhedral cones: a combinatorial perspective" ([arXiv:1512.06033](https://arxiv.org/abs/1512.06033)).

**Idea:** The network is homogeneous, so the radial integral is trivial (the indicator depends only on x/|x|).
Directional sampling therefore costs nothing extra but removes the radial variable. One can go a step further: by
rotation invariance, the normalized measure of a cone K equals E_U[ length(K ∩ S^1_U)/2π ] over a uniformly random
2-plane U (a Crofton formula).

**Recipe:**
- Sample orthonormal pairs (u,v) and parametrize x(θ) = cosθ·u + sinθ·v.
- Walk θ ∈ [0,2π) exactly. Each pre-activation is a·cosθ + b·sinθ within a pattern, so the next crossing has a
  closed form. Accumulate the arc length where argmax equals the 2nd-argmax. The 2nd-argmax only changes at the 45
  braid walls, which are also closed-form crossings.
- Variance reduction: choose U to contain a braid-wall normal (e_i − e_j for the current top pair). The circle then
  crosses the dangerous wedge transversally. Or use antithetic / orthogonal frames (Deák's orthonormal-system trick).

**Cost:** about 40–100 crossing solves per circle. Each circle averages over a 1-D continuum of directions.

**Why it beats MC:** It is Rao–Blackwellized, with variance ≤ MC per circle. It is cheap because each crossing is a
1-D closed-form root. It generalizes to k-planes with k=3 (exact spherical-polygon areas via Girard's formula). This
method is the coordinate-free sibling of #1. It does not need a chamber decomposition, so it is robust to weird
perturbed weights.

### 3. Stratification on braid chambers (and their top-k faces) + control variate from an "oracle" chamber map
**Refs:** The chamber combinatorics are classical (Stanley, *An Introduction to Hyperplane Arrangements*). Angle data
for Weyl chambers: Kabluchko, Vysotsky, Zaporozhets, "Convex hulls of random walks, hyperplane arrangements, and Weyl
chambers", *GAFA* 2017 ([doi](https://dx.doi.org/10.1007/s00039-017-0415-x)). Godland & Kabluchko, "Conic intrinsic volumes
of Weyl chambers" ([arXiv:2005.06205](https://arxiv.org/abs/2005.06205)) and "angle sums of Schläfli orthoschemes /
tangent cones of Weyl chambers" ([arXiv:2007.02293](https://arxiv.org/abs/2007.02293)).

**Idea:** All chambers have mass exactly 1/10!. The target is constant on each chamber, and gating is nearly constant
on each chamber, so the natural strata are chambers or coarser unions. The between-stratum variance is killed exactly.

**Recipe:**
- Stratify on the ordered top-3 positions (σ(1),σ(2),σ(3)): 720 strata of mass 1/720 each, which captures where errors
  live. Allocate samples per stratum by Neyman allocation, using a pilot estimate of per-stratum error.
- Inside each stratum, run #1 (pre-integrate g_1, g_2) on QMC points over the remaining coordinates.
- The Weyl-chamber face-angle formulas give exact constants for the leading-order error expansion. The mass within
  ε of the wall x_(1)=x_(2) is ε·(Gaussian surface measure of that facet). The density of the top spacing at 0 has a
  closed form: 10·9·∫φ(y)²Φ(y)^8 dy per unit gap, computed in 1-D.
- Use that expansion only as a **control variate**, with exactly known expectation, not as the answer. This is how
  slowly converging perturbative expansions become useful: they remove most of the variance, and sampling corrects the
  rest without bias.

**Cost:** negligible overhead.

**Why it beats MC:** It never pays for the 0/1 noise of "which chamber am I in", and it gives an exact anchor for a
control variate.

### 4. Exact region enumeration inside each chamber + cone-probability cubature (Genz / Botev minimax tilting)
**Refs:** Serra, Tjandraatmadja, Ramalingam, "Bounding and counting linear regions of deep neural networks", ICML 2018
([arXiv:1711.02114](https://arxiv.org/abs/1711.02114)). Zhang, Naitzat, Lim, "Tropical geometry of deep neural networks",
ICML 2018 ([arXiv:1805.07091](https://arxiv.org/abs/1805.07091)). Star-set / ProbStar exact reachability with Gaussian
inputs (Tran et al., StarV tool; [NSF PAR](https://par.nsf.gov/biblio/10649230-starv-qualitative-andquantitative-verification-tool-forlearning-enabled-systems)).
Botev, "The normal law under linear restrictions: simulation and estimation via minimax tilting", *JRSS-B* 2017
([arXiv:1603.04166](https://arxiv.org/abs/1603.04166); R `TruncatedNormal::mvNcdf`/`mvNqmc`). Genz & Bretz, *Computation of
Multivariate Normal and t Probabilities* (2009).

**Idea:** The RNN unrolls to a depth-10 ReLU net, which is a tropical rational map. The input space splits into a
polyhedral complex indexed by the 160-bit activation pattern, i.e. the symbolic itinerary of the switched linear system.
On each cell the logits are linear, so "correct" is a polyhedral cone. acc is a finite sum of Gaussian cone
probabilities, and each cone probability can be computed to roughly 1e-6–1e-8 relative accuracy with Genz separation of
variables plus Botev's minimax-tilted QMC, which comes with deterministic error bounds.

**Recipe:**
- Restrict the enumeration to the **error set** (4.7% mass) and to the chambers near the top-3 walls. Do a DFS over
  activation patterns with LP feasibility, as in star-set reachability: split a star on each ReLU whose sign is not
  fixed on the current cell.
- For each leaf cone {A x ≤ 0}, compute P via `mvNqmc`. The cone is homogeneous, so P(Ax≤0) = P(z ≤ 0) with z ~ N(0, AAᵀ)
  when A has ≤10 rows. When it has more rows, use the linear-ellipsoidal (LIN-ESS) + Holmes–Diaconis–Ross estimator
  (Gessner, Kanjilal, Hennig, AISTATS 2020, [arXiv:1910.09328](https://arxiv.org/abs/1910.09328)) or minimax tilting with
  redundant-constraint pruning.
- Hybrid version: enumerate exactly only the cells carrying the most error mass, then estimate the remainder with #1.

**Cost:** It depends on the number of cells. If gating is "mostly determined by rank order", expect O(10^3–10^5)
relevant cells after restricting to the top-3 structure. Each LP plus Genz call takes ~ms. Worst case it is exponential.
NP-hardness persists even under parameter noise (Soltanalian 2026, [arXiv:2607.14375](https://arxiv.org/abs/2607.14375)),
so use it as a component, not the whole method.

**Why it beats MC:** Its error is deterministic and tiny per cell. The total error is driven by the truncated tail of
unenumerated cells, which you can bound.

### 5. Ribando–Aomoto hypergeometric series for simplicial-cone solid angles (+ Brion–Vergne decomposition)
**Refs:** J. Ribando, "Measuring solid angles beyond dimension three", *DCG* 36 (2006). Fitisone & Zhou, "Solid angle
measure of polyhedral cones" ([arXiv:2304.11102](https://arxiv.org/abs/2304.11102)) and its SageMath implementation.
Fitisone's thesis ([UKY](https://uknowledge.uky.edu/math_etds/109)).

**Idea:** The normalized solid angle of a simplicial cone with unit generators v_i is a multivariate Taylor series in the
off-diagonal Gram entries ⟨v_i, v_j⟩. It converges when the "associated matrix" is positive definite. Any cone can be
decomposed (Brion–Vergne / tridiagonal decompositions) into pieces that satisfy this.

**Recipe:**
- The chambers of A_9 are simplicial, with generators π(e_1+…+e_k) and Gram matrix min(i,j)-type, which gives a
  tridiagonal precision.
- Error wedges near a wall are thin simplicial cones whose generators are nearly coplanar. Such cones are typically far
  from orthogonal, so the series needs decomposition first.
- Use it to compute *exact* (to series truncation) measures of the thin error cones found in #4, as a deterministic
  alternative to Genz.

**Why/when:** It is deterministic and differentiable in the weights, which helps with robustness across perturbations.
Feasibility in d=10 is moderate, since the series has many terms. Treat it as a secondary tool to cross-check #4.

### 6. Conic intrinsic volumes, kinematic formulas, statistical dimension (Amelunxen–Lotz–McCoy–Tropp)
**Refs:** Amelunxen, Lotz, McCoy, Tropp, "Living on the edge: phase transitions in convex programs with random data",
*Inf. Inference* 2014 ([arXiv:1303.6672](https://arxiv.org/abs/1303.6672)). Amelunxen & Lotz, combinatorial intrinsic
volumes of polyhedral cones ([arXiv:1512.06033](https://arxiv.org/abs/1512.06033)).

**Idea:** Gaussian measures of cones and of thin neighborhoods of their faces are governed by intrinsic volumes
v_k(C). The Gauss–Bonnet relation Σ(−1)^k v_k = 0 and the Steiner formula give exact identities. The kinematic formula
gives the probability that a randomly rotated cone hits another.

**Recipe:**
- Use the spherical Steiner (tube) formula around the braid walls to write the error mass as Σ_k v_k-weighted integrals
  of the wedge width. This gives a principled first-order and second-order "boundary-layer" expansion: error ≈
  ∫_{wall} w(z)φ(z)dz + curvature corrections.
- The Gauss–Bonnet angle-sum identities also provide exact checksums. For example, the per-σ cone measures from #4
  must sum to 1/10!.

**Why:** It gives analytic structure and validation identities, not a stand-alone estimator. It is best used to build
the control variate in #3.

### 7. Sparse grids / randomized QMC on the pre-integrated, chamber-stratified integrand
**Refs:** Griewank et al. (above). Dick, Kuo, Sloan, "High-dimensional integration: the quasi-Monte Carlo way",
*Acta Numerica* 2013.

**Recipe:** After #1 and #3 the integrand on (g_2..g_9, t) is smooth and has low effective dimension, because the top-3
gaps dominate. Map to [0,1]^9 via Φ^{-1}, order the variables by importance (g_2, g_3, t, …), and use scrambled Sobol
with randomized replicates for error bars. Optionally use Smolyak sparse grids in the 2–3 leading variables, with MC in
the rest.

**Expected:** O(N^{-1+ε}) convergence when monotonicity holds, versus O(N^{-1/2}) for MC.

### 8. LIN-ESS + Holmes–Diaconis–Ross nested sampling for small cone probabilities
**Refs:** Gessner, Kanjilal, Hennig, AISTATS 2020 ([arXiv:1910.09328](https://arxiv.org/abs/1910.09328)). BoTorch
`lin_ess`. Wu & Gardner 2024 ([arXiv:2407.10449](https://arxiv.org/abs/2407.10449)).

**Idea:** Elliptical slice sampling restricted to {Ax ≤ 0} is rejection-free, because ellipse–hyperplane intersections
have a closed form. HDR (subset simulation) writes a small probability as a product of moderate conditional
probabilities.

**Use:** Estimate the measure of a single tiny error cell, or of the whole error set when it is defined by a moderate
number of linear constraints after pattern fixing, with relative rather than absolute error. This is useful when
individual error cells are too thin for Genz/Botev to handle well.

### 9. Gaussian comparison (Slepian / Sudakov–Fernique / Gordon) for certified bounds
**Idea:** Inside a fixed gating pattern, the logit differences are linear Gaussian functionals. Within a chamber they are
conditionally Gaussian given the gaps. Slepian-type inequalities compare P(max of a Gaussian vector ≤ 0) across
covariances, which yields rigorous upper and lower bounds on per-pattern correctness from simpler (e.g., equicorrelated)
covariances whose orthant probabilities are 1-D integrals.

**Use:** For cheap certified brackets [L, U] on acc. This is helpful for robustness across random weight perturbations,
but probably not tight enough to beat MC at the 1e-4 level by itself.

### 10. Switched-linear-system / symbolic-dynamics view of the recurrence
**Idea:** h_{t+1} = D_t(W_h h_t + W_i x_t), where D_t is the diagonal 0/1 gate at step t, so h_10 = Σ_t (∏ D W_h) D W_i x_t.
Itineraries (D_1..D_10) are symbolic sequences. Under the order structure, the gate at step t mostly depends on the
running rank of x_t among x_1..x_t, i.e. on records and descents of the permutation. Enumerate itineraries as a
transfer-matrix / automaton over "running-rank" symbols, and attach to each symbol the exact Gaussian conditional
law of the new increment.

**Use:** If gating really is a function of the running-rank sequence (a sequential statistic of σ), acc reduces to a
sum over at most 1·2·…·10 = 10! sequences, but with a Markov/dynamic-programming factorization. The relevant
statistic is often just "rank of x_t among the current top-3", giving ≤4 symbols per step and ~4^10 ≈ 1e6 paths,
with DP merging. Exact contributions per path again come from #4 or #1. This is the most direct exploitation of the
permutation structure. It is speculative and needs checking against the actual gates.

---

## Recommended combination
#3 (stratify on the top-3 chamber face and use the known 1/720 masses) + #1 (exactly pre-integrate the top gap g_1, and
g_2, by PL breakpoint walking) + #7 (scrambled Sobol over the remaining gap coordinates). Add the Steiner/tube expansion
of #6 as a control variate. Use #4 with Botev's `mvNqmc` to compute exactly the few cells holding the most error mass.
#2 is the coordinate-free fallback for weight perturbations where the chamber-based gating assumption breaks.
