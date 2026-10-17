# T19 analytical handoff: reference geometry plus output-error transport

## Question
Can a tractable algorithmic reference retain the non-Gaussian rank geometry while a cheaper calculation handles only the trained model's departure from it?

## Method
Analytical derivation and repository review only. This memo proposes a concrete version of T37 for the DRI to reserve and assess; it does not claim T37 or implement experiments. It incorporates the distinction in `lit/web_arc/README.md`, S4, between propagating an exact reference plus an error and approximating the entire network distribution. T35, T36, T38 and the parent's T33/T34 remain separate work.

## Status
**Correction: the proposed top-two normalization has an infinite second moment on the original checkpoint. Its Gaussian covariance-matching version is withdrawn.** See the exact obstruction below. The endpoint identity remains conditional on its assumed surrogate law, but true residual covariance cannot be supplied for this scale.

Ready for DRI handoff. Zero experiments or model calls. Recommendation: do a small oracle independence falsifier before investing in recurrent covariance propagation. The reference-error route has a tractable endpoint calculation, but its independence and residual-shape assumptions are substantial and unverified.

## Results: exact reference anchor

Reading the ACTUAL assignments in `alg_zoo/handcrafted.py`, `handcrafted_2nd_argmax_10`, gives reference logits

    g_i(x) = x_i^+ + 2 max(0, x_j : j != i).

The nearby source comment says minus; the implemented output weights give PLUS. The distinction matters. Let M=x_a and S=x_b be the largest and second-largest inputs. Then

    g_b - g_a = M^+ - S^+,
    g_b - g_j = S^+ - x_j^+   (j != a,b).

Consequently the reference is correct whenever at least two inputs are positive. If zero are positive, all scores tie and success probability is 1/T. If exactly one is positive, all nonmaximum positions tie and success is 1/(T-1). For iid continuous, sign-symmetric inputs, including the Gaussian law,

    A_ref = 1 - 2^(-T) [ (T-1)/T + T(T-2)/(T-1) ].

At T=10 this is **0.9904405381944444**. This is an algebraic deduction from the implemented reference, not a Monte Carlo measurement. It supplies the reference-only value requested in T37. It does not approximate the trained model's accuracy.

The 22-neuron reference stores delayed positive inputs, not just a two-dimensional max state. Its simple final accuracy therefore does NOT establish that the entire joint law of reference hidden state and trained-minus-reference error can be propagated cheaply.

## Algorithm A: exact order-statistic endpoint plus a low-factor residual

### 1. Define an error with appropriate invariances

Remove the arbitrary common logit shift using P=I-11'/T. Choose a positive, charged/model-derived alignment scale beta. Set

    s(M,S) = sqrt(M*M + S*S),
    eta(x) = P [ f(x)/beta - g(x) ] / s(M,S).

The zero denominator has probability zero. The scale s is positively homogeneous and depends only on the two top values. Using an UNscaled additive Gaussian output error would break the network's positive homogeneity and is a weaker proposal.

The approximation is a law for eta that is independent of the reference inputs:

    eta ~= mu + A G + diag(sigma) xi,
    G ~ N(0,I_q), xi ~ N(0,I_T), mutually independent and independent of X.

Because a common shift never affects argmax, centering need not be maintained in the factor representation as long as its pairwise margins match. Full covariance corresponds to large q and may be intractable; q=0,1,2 is a hypothesis to falsify, not an established feature.

This is NOT a Gaussian approximation to the entire trained margin conditional on a top-three skeleton. It retains the reference's individual clipped low inputs, their truncation, and the exact winner event. That distinction is the possible escape from T1. If instead the reference is fixed given the SAME skeleton and one Gaussianizes the residual conditional on that skeleton, it is exactly the same translated-Gaussian family as Gaussianizing the full logits. That version cannot evade T1's oracle ceiling merely by renaming the random variable.

### 2. Integrate the joint decision event without a union bound

Condition on ordered top positions (a,b), their values (M,S), latent G=g, and the normalized second-position error eta_b=z. Put mu_j(g)=mu_j+A_j g. Conditional on these quantities, the residuals for other outputs are independent, and each low input X_j is independently N(0,1) truncated below S. Therefore the approximate conditional success probability is exactly

    Q_ab(M,S,g,z)
      = Phi((z + (M^+-S^+)/s - mu_a(g))/sigma_a)
        * product_{j != a,b} J_j(S,s,g,z),

    J_j = (1/Phi(S)) int_{-infinity}^S phi(x)
          Phi((z + (S^+-x^+)/s - mu_j(g))/sigma_j) dx.

Here f/beta was used, so no beta remains in the reference gaps. Equivalently one can avoid dividing f and carry beta in every gap. z has law N(mu_b(g),sigma_b^2). Zero sigma is handled by deterministic inequalities with explicit argmax tie conventions.

For S<=0, J_j=Phi((z-mu_j(g))/sigma_j). For S>0, the x<=0 contribution is

    [Phi(0)/Phi(S)] Phi((z+S/s-mu_j(g))/sigma_j),

and the integral from 0 to S is a difference of bivariate-normal CDFs after rescaling. Thus there is no residual numerical integration over seven/eight individual low inputs. Crucially, this product computes the JOINT winning event; it does not sum marginal failure probabilities.

The density of the largest and second-largest values, ignoring their positions, is

    T(T-1) phi(M) phi(S) Phi(S)^(T-2),  S<M.

Average Q_ab over this density, the Gaussian z and G, and the T(T-1) equally likely ordered position pairs. The resulting calculation has **3+q continuous integration variables per pair**, regardless of T. The all-negative and one-positive sectors are included; they are not given a fitted blind-region constant.

An arbitrary scalar-conditioned residual s(M,S) with the same homogeneity works identically. More flexible functions must be charged and predeclared, not selected after test errors.

### 3. Cost and error bottleneck

At q=0 and n=8 nodes in each of M,S,z, the naive rule has 90*8^3=46,080 nodes and roughly eight bivariate-CDF factors per node. At q=1 it has 368,640 nodes; at q=2 about 2.95 million. These are NOT FPE: special-function costs, Gaussian transforms, products, recurrence moment propagation, and alignment all count. Even q=0 can consume tens of millions of arithmetic operations. Adaptivity may reduce this but is not assumed.

The main unresolved error is the residual law, not the endpoint geometry. A cheaper endpoint does not help if input dependence, nonlinear shape or a high factor rank creates >0.1 pp bias. For a budget B the relevant accuracy tolerance is sqrt(p(1-p)/B), rather than a universal fixed pp target. A low-budget method can tolerate more bias than a high-budget one.

### 4. Cheapest falsifier, to be claimed by DRI before execution

Use only predeclared DEV copies and fixed train/held-out input sets. This is a diagnostic ceiling, not the estimator. No fit is shared with scored test models.

1. Obtain normalized residual eta(x) exactly from trained/reference logits. If beta is fitted for this ceiling, fit it only on the training set and charge/report it; this is optimistic evidence, not a weight-derived algorithm.
2. **Independence alternative:** reconstruct held-out logits g(x)+s(M,S)*eta(x') using residuals from independently generated x'. This preserves the empirical non-Gaussian joint residual law but breaks its dependence on the reference. Compare accuracy and paired error families with the true model. Multiple independent residual assignments quantify Monte Carlo uncertainty. If the oracle independent residual already fails the intended bias tolerance, stop Algorithm A.
3. **Shape alternative:** replace the independent empirical residual by a Gaussian with its oracle training mean/covariance. This isolates Gaussian shape from independence. T1 makes failure plausible, but does not itself settle this richer-reference test.
4. **Factor-rank alternative:** compare q=0/1/2 factor-plus-diagonal approximations with the full-covariance Gaussian on untouched inputs. Only if the endpoint law remains accurate at affordable q should the deterministic order-statistic integral be built.

These tests do not require computing ground truth to high precision: paired/common-random-number comparisons measure each approximation's bias directly. Small observed bias is still not proof; residual uncertainty and independent replication are required.

## Algorithm B: propagate reference error with an explicit gate-crossing remainder

This is the constructive route to the residual moments IF Algorithm A's oracle tests survive. It also specifies why a vague 'small error covariance' argument is insufficient.

Let r_{t+1}=rho(A r_t+b x_t) be an exact reference state. Choose a linear alignment B from the reference state to the trained hidden coordinates, hbar_t=B r_t; B can be rectangular. Define e_t=h_t-hbar_t and

    z_t = W hbar_t+w x_t,
    D_t = diag(1[z_t>0]),
    delta_t = rho(z_t)-B rho(A r_t+b x_t).

Then the exact residual recurrence is

    e_{t+1} = delta_t + D_t W e_t + kappa_t,
    kappa_t = rho(z_t+W e_t)-rho(z_t)-D_t W e_t.

Componentwise,

    kappa_t >= 0,
    kappa_t <= |W e_t| * 1[|z_t| <= |W e_t|].

The remainder vanishes unless the residual can cross a reference gate. This is a pathwise identity, not an independence approximation. The output error is W_o B r_T minus the chosen reference logits, plus W_o e_T; readout mismatch must be included rather than silently setting W_o B equal to the reference readout.

Under a conditional Gaussian approximation v_i=(W e_t)_i ~ N(mu_i,sigma_i^2) with z_i FIXED by the reference state, the gate-crossing mean has the exact formula

    E[kappa_i] = (z_i+mu_i) Phi(a_i) + sigma_i phi(a_i)
                 - rho(z_i) - 1[z_i>0] mu_i,
    a_i = (z_i+mu_i)/sigma_i.

Stein's identity also gives

    Cov(v_i,kappa_i) = sigma_i^2 [Phi(a_i)-1[z_i>0]].

Cross-neuron terms require their joint Gaussian law, not independent gates. These formulas prevent dropping the mean gate-crossing correction just because E[e] is small. They do NOT justify the conditional Gaussian approximation or dropping covariance between delta_t, reference state, and e_t.

Ignoring kappa produces the first-order recurrence e_lin. Its exact terminal difference is

    e_T-e_lin_T = sum_t (D_{T-1}W)...(D_{t+1}W) kappa_t.

Multiplying by output DIFFERENCE rows gives decision-specific remainder contributions. Large hidden-state RMS can be harmless after readout, while small hidden RMS can reverse a tiny margin. This identity therefore supports a cheap, goal-oriented diagnostic before constructing a moment closure.

### Distinct alternatives and falsification

- **Small state error but important gate crossings:** evaluate the exact weighted-kappa decomposition on DEV trajectories. Compare the first-order model's labels/accuracy with the trained model. If the output-relevant remainder is not small, linear error propagation is ruled out; analytic gate-crossing correction or explicit mixture structure is necessary.
- **Gate remainder is small but residual/reference dependence dominates:** Algorithm A's residual-shuffle test fails even though first-order recurrence tracks the original. Then covariance propagation with independence is the wrong closure; improving last-step quadrature cannot repair it.
- **Reference alignment is the bottleneck:** a 16-vs22 hidden-width difference prevents literal weight subtraction. B and its fitting/derivation cost must be explicit. High hidden R-squared or a known reference accuracy is not sufficient. The known separable-state reference in FINDINGS has already poor decision fidelity and should not be advertised as a small-error baseline without this test.

Dense conditional covariance propagation costs O(T H^3) per reference state node, plus reference transport and nonzero-mean bivariate ReLU moments. This can greatly exceed a forward pass. Compressing covariance is another falsifiable assumption, not free accounting. Exact reference trajectories can still be high-dimensional even though their terminal ordering is simple.

## Recommendation

Recommend the **oracle residual-independence / shape / factor-rank diagnostic for Algorithm A**, under a DRI claim for T37, before any full recurrent propagation. It directly tests whether an analytically tractable endpoint exists and separates three failure modes with inexpensive paired comparisons. If it survives, Algorithm B gives an exact starting identity for deriving rather than empirically observing its residual moments. If it fails, preserve the exact reference anchor and do not build a Gaussian-error estimator merely because its covariance recurrence is available.

Backward probability transport was also considered: augment hidden state with the true running top-two values/positions and apply the Bellman operator V_t(s)=int phi(x)V_{t+1}(F(s,x))dx. The final step is analytically integrable by scalar breakpoints, but representing V_t globally introduces min/max affine CDF expressions with rapidly proliferating pieces. Without an additional proved or falsified compression property this repackages exact preintegration/region growth, so it is lower priority than the diagnostic above. No new implementation is proposed for it.

## Claims

- [inferred] The implemented handcrafted reference has the exact Gaussian accuracy derived above.
- [inferred] Under the stated low-factor independent residual law, the joint accuracy reduces exactly to the stated 3+q dimensional order-statistic integrals.
- [inferred] The residual recurrence and nonnegative gate-crossing remainder are exact pathwise identities.
- [speculative] An affordable residual factorization and closure exists for the trained M16,T10 family. No result here supports that empirical premise.

## Alternatives not ruled out

Dependence on top positions, reference scale, low-input order statistics, or gate history may be essential. Conditioning on increasingly rich reference variables can remove closure bias while making integration too expensive, as T1's fidelity/leverage tradeoff illustrates. A non-Gaussian low-factor law could survive where the Gaussian law fails. A per-family explanation may be allowed by ARC while failing the repository's stricter no-advice transfer rule; neither is silently assumed.

## Failure modes

Residual/reference dependence; non-Gaussian multimodality; large factor rank; costly bivariate CDF quadrature; conditional covariance transport requiring the full reference history; tiny-margin amplification of gate-remainder errors; uncharged fitted alignment or residual statistics; calling a diagnostic fit mechanistic without deriving it.

## Compute accounting and source provenance

Zero model evaluations, generated inputs, LP solves, fitting operations or new experiments. Work is file reading, algebra and documentation. Source implementation read: `/workspace/dependencies/alg-zoo/alg_zoo/handcrafted.py` (existing checkout). Prior results reviewed: T1/stoch_margin, T29/tensor_mass, T32/tensor_cross, `lit/web_methods/README.md`, `lit/web_arc/README.md`, FINDINGS, and the existing theory probes. The T32 report is treated as provisional while the parent's stale-prefix walker audit is pending; none of its numerical results is used in these derivations.

No new external URL was opened. The ARC reference/error motivation comes through the DRI's local web_arc summary of Hilton's December21,2024 notes (S4: https://drive.google.com/file/d/1t2rXhDQqWrOXRwjqstbrBPWiCmYlMts_/view). The local summary also exposed post-January2026 sources already logged by the DRI in `lit/web_log.md`; this memo depends on their methodological context, not their unverified performance claims. It does not derive from an unpublished M16,T10 solution.

## Correction: top-two normalization is singular (independently flagged by parent)

The assertion that s=sqrt(M^2+S^2) merely has a probability-zero zero denominator
missed a second-moment divergence. At the deterministic anchor
x=(0,0,-1,-1,-1,-1,-1,-1,-1,-1), the reference scores are all zero, while exact
rational evaluation of the original checkpoint gives f_0-f_1=-0.6230636190194814
(the decimal is rounded; the rational is printed by `normalization_witness.py`).
The centered-logit norm is approximately54.7154. Thus the numerator is nonzero.

By continuity there is an open box of negative low inputs near -1, and a small
neighborhood of (M,S)=(0,0), on which |f_0-f_1| has a positive lower bound. In that
neighborhood reference-score differences are O(s), so for every fixed finite
positive beta the normalized residual difference has magnitude at least c/s on
a sufficiently small subneighborhood. The Gaussian density is bounded below.
The ordering constraint M>S occupies a nonzero angular sector. Integrating its
squared magnitude gives a positive constant times

    integral_0^epsilon r dr / r^2 = infinity.

Therefore at least one residual coordinate has infinite second moment. Finite
sample covariance can look well behaved while drifting as rare nearzero top-two
values are encountered. Covariance matching is not a globally defined oracle
for this scale. The same obstruction affects any comparable continuous,
degree-one scale depending only on (M,S); changing its norm does not fix it.

Two possible NEW assumptions are a fixed positive scale (finite Gaussian-input
moments but loss of positive homogeneity) or a scale proportional to ||X||
(bounded residual on the sphere, but the low-input product endpoint no longer
factorizes as written). A sector cutoff must explicitly include its excluded
probability/error contribution. None of these repairs was tested or selected in
this memo, and no finite empirical covariance silently repairs the old formula.

`normalization_witness.py` reproduces the anchor using Fraction arithmetic on the
binary floating-point checkpoint weights. Checkpoint SHA256:
498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb.
This correction performed ONE deterministic forward-weight evaluation, with
exact rational arithmetic, not zero model calls. No random dataset or estimator
experiment was run. The original memo's zero-call accounting applies only to its
initial derivation, before this correction.
