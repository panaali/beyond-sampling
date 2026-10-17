# T19: protocol clarification and three bounded research leads

Follow-up: [gate-supported projection formulas and T38 cautions](supported_projection.md). This is a source/formula handoff to the DRI-held T38, not a competing task.

## Question

What does ARC's original challenge require, and which specific numerical ideas remain worth falsifying after E10–E30?

## Method

Read AGENTS, TASKS, INDEX, the existing literature registry, candidate notes and estimator audit; verify primary sources with web search and full-text PDF reading. No experiments or model calls. [source_notes.md](source_notes.md) records URLs, versions and incidental later-source exposure. The applications below are our proposals, not results established by the cited authors.

## Status

Done: source/protocol review. None of the proposed estimators is implemented or validated. DRI should register follow-ups before implementation; EP overlaps T28 and any low-input distribution work belongs to held T1.

## Results

### Protocol: maintain the distinction, correct its attribution

[ARC's January 26, 2026 challenge](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/) asks for deduction from model structure, matching sampling's MSE versus compute. Perturbations must materially change accuracy; training seeds 0–4 should also work. Its broader matching-sampling formulation allows suitable advice. The post distinguishes practical estimation from the harder surprise-accounting goal.

[The linked November 18, 2025 explanation](https://www.alignment.org/blog/competing-with-sampling/) explicitly says it lacks a formal definition of mechanistic. Avoiding random/pseudorandom samples is only a heuristic. Deduction-projection and covariance propagation qualify. Its example says observed point values justify only their own probability mass without structural evidence for extrapolation.

**Our interpretation:** v3's sampling/quadrature/mechanistic classes are a defensible conservative operationalization, not an explicit ARC ban on numerical quadrature. Merely freezing a random seed does not produce structural understanding. Conversely, exact structural reduction followed by numerical integration cannot be rejected solely because it has nodes. Explain which unobserved regions follow from the derived structure and which are inferred from representatives. Adaptive nodes are neither necessary nor sufficient. Keep existing classes for comparability; present ambiguity separately instead of retrospectively declaring a win. Generality, disclosed advice, full implemented cost and honest seed/noise MSE remain essential in our protocol.

### Lead A: probability transport on decision facets

**Primary support:** [Koyama, arXiv:1512.06564v2](https://arxiv.org/abs/1512.06564v2), section 4, equations (17)–(18), develops Gaussian polyhedron derivatives as lower-dimensional face integrals; its holonomic gradient method propagates them through an ODE. General-position assumptions matter. This is not a theorem about efficient RNN integration.

**Our proposed application:** on a fixed rank/gate chamber, write correctness as affine inequalities a_j(s)^T x+b_j(s)>=0 along a parameter path s. Differentiating the Heaviside product and using coarea gives

    dP/ds = sum_j integral_{F_j(s)} phi_d(x)
                      (a'_j(s)^T x+b'_j(s))/||a_j(s)|| dS(x).

Here F_j also satisfies every other inequality. After summing adjacent gate chambers, artificial interior interfaces cancel when the correctness indicator agrees on their two sides. Only decision facets remain, though their geometry can still require many patches. Integrate this derivative from a reference with known accuracy. One possible reference uses task-derived logits g_i(x)=-|x_i-second_largest(x)|, giving accuracy one almost everywhere, and l_i(s)=(1-s)g_i+s f_i. This avoids assuming a width-16 perfect RNN exists; it introduces additional rank pieces and endpoint degeneracies.

**Cheap feasibility gate, not performed:** first use exactly enumerable T=2,3 DEV systems, or a finite set of already constructed cones. Compare analytic flux against centered finite differences of independently integrated cone masses; then count surviving decision patches after interior-face cancellation. Cap at 1000 patches and 10 CPU-minutes. Advance only if patch count falls substantially and probability increments agree to 1e-6 away from degeneracies. Count all enumeration, linear algebra, ODE and face-integration operations.

**Two alternatives:** (i) interior cancellation is algebraically valid but leaves essentially all error-cone complexity—patch counts falsify compression; (ii) topology changes or singular faces invalidate a smooth path—step-halving and independent endpoint masses distinguish this from numerical ODE error. E14's huge cone population is evidence against cheap global enumeration; HGM alone does not repair it. This is lower priority unless an existing candidate yields a genuinely small face complex.

### Lead B: adaptive functional tensor train after exact preintegration

**Primary support:** [Gorodetsky, Karaman and Marzouk, arXiv:1510.09088v3](https://arxiv.org/abs/1510.09088v3), sections 3–5, constructs adaptive univariate fibers and rank adaptation. The representation is a product of matrix-valued functions, allowing integration through products of integrated cores. Its successful discontinuous examples do not establish small rank for this RNN.

**Our proposed application:** for one fixed tie-position configuration, transform the exact face measure to independent uniforms u and use the existing exact segment walker to obtain f(u), its error-mass integrand. Approximate

    f(u_1,...,u_d) ~= G_1(u_1)...G_d(u_d),
    integral f du ~= [integral G_1 du_1]...[integral G_d du_d].

Unlike the failed additive surrogate, bounded tensor rank can retain interactions between all lower inputs. Use adaptive piecewise fibers near band changes, not one global polynomial. Do not train a model-specific table off-budget.

**Cheap feasibility gate, not performed:** predeclare four DEV configurations using position indices only; use deterministic cross pivots, ranks 2/4/8 and at most 1024 exact face evaluations per configuration. Hold out a separately generated deterministic validation grid and compare its weighted residual, not just training interpolation error. Compare the estimated integral at matched cost to fixed Sobol and its refinement. Stop if increasing rank does not reduce held-out weighted error or misses tail mass; do not extrapolate a good four-configuration result to the full model. Account for pivots, all failed adaptive fits, validation and matrix decompositions.

**Two alternatives:** (i) low rank exists but rare thin peaks evade pivot selection—test a second predeclared shifted validation grid and deliberate near-tie extremes; (ii) rank is intrinsically large because gate histories entangle inputs—inspect singular values of small deterministic unfoldings before full FT implementation. Both are compatible with E23's poor compact regression. Classify this as quadrature until a structural low-rank derivation supports a different DRI ruling.

### Lead C: rank-respecting EP as a diagnostic of joint probability closure

**Primary support:** [Cunningham, Hennig and Lacoste-Julien, arXiv:1111.6832v2](https://arxiv.org/abs/1111.6832v2), sections 2.1.1, 3.2 and 7, gives EP for Gaussian halfspace products and documents severe representation sensitivity. Equation (57) uses a rank-one covariance update. Near-redundant constraints can remain pathological even after duplicate removal.

**Our proposed application:** retain the true Gaussian input and explicit rank/gate inequalities of a chosen affine surrogate, instead of replacing the rank-conditioned input with a full-support Gaussian and summing marginal failures. Approximate a chamber's joint correctness probability

    Z = integral phi(x) product_i 1[l_i < c_i^T x < u_i] dx,
    q(x) proportional to phi(x) product_i exp(-tau_i (c_i^T x)^2/2
                                                       + nu_i c_i^T x).

For cavity projection z~N(m,v), the exact one-factor normalizer is Phi((u-m)/sqrt(v))-Phi((l-m)/sqrt(v)); its first two moments update the site. The covariance update is

    Sigma_new = Sigma - dtau/(1+dtau*c^T Sigma c) (Sigma c)(Sigma c)^T.

**Cheap feasibility gate, not performed:** this is a T28 extension, not another full T1 distribution-propagation attempt. On 12 predeclared DEV surrogate chambers compare EP, marginal union and a tightly converged low-dimensional reference. Repeat after positive constraint rescaling, reordering and insertion/removal of an exact duplicate. Stop if equivalent representations move the answer by more than the target accuracy, or if the surrogate remains wrong when its probability is integrated accurately. At most 20 sweeps/site; charge O(m d^2) updates, determinants, preprocessing and reference integration. No claim that EP fixes frozen-gate bias.

**Two alternatives:** (i) marginal overlap/chamber leakage caused E30's error—accurate joint integration helps the unchanged surrogate; (ii) surrogate gate geometry caused it—EP/reference agreement with persistent network disagreement refutes that repair. Existing T1 diagnostics can supply the latter comparison without duplicating their runs.

### Reproduction of this review

No numerical performance result is reported. From the repository root, the inventory check was:

```sh
rg -n -i 'tensor.train|holonomic|epmgp|expectation.propagation' m16 lit
```

It found prior EP discussion in literature, not an EP implementation; no tensor-train or holonomic estimator was found. This is an inventory observation, not proof nobody tried one elsewhere. Open the versioned URLs and sections in source_notes.md to reproduce source checks.

## Compute accounting

0 model forward passes; 0 generated model inputs; 0 LP solves; 0 estimator benchmarks. Work was source reading, repository text search and writing. Proposed budgets above are caps, not measured costs. No new performance result qualifies for the leaderboard.

## Claims

- [measured] Primary-source text and cited equations were read; later-source exposure is logged separately.
- [inferred] The DRI's current classification is conservative and stronger/more specific than ARC's informal wording.
- [speculative] Boundary cancellation or tensor rank might compress error geometry; neither is evidence of a competitive estimator.
- [inferred] Generic EP is a poor unconditional recommendation given its documented polyhedral pathologies; use only the falsifiable diagnostic above.

## Alternatives not ruled out

Existing exact preintegration may already capture most accessible structural gain. None of these papers establishes tractability for M_16,10. Low-dimensional successes can fail at T=5 or T=10. A model-specific explanation/advice scheme could be acceptable under ARC's broader formulation while failing our stricter transfer protocol.

## Failure modes

Huge face complexes, singular homotopies, rare peaks missed by adaptive grids, high tensor rank, EP representation dependence and approximation bias exceeding sampling's shrinking error. Source wording cannot resolve eligibility of a particular unpublished estimator; maintain transparent separate classes and avoid claiming an ARC ruling.
