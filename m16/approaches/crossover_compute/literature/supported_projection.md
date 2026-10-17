# T19 follow-up: gate-supported projection for T37/T38

## Question

Can the harmonic deduction-projection idea become a genuinely cheap, different primitive for reference-model error integration, without repeating T35's cumulant implementation?

## Method

Read the new `lit/web_methods` and `lit/web_arc` scans, then the primary halfspace paper and the full cumulant paper. The recommendation is **an auxiliary polytope integrator, not yet an RNN accuracy estimator**: keep one hemisphere constraint exactly while projecting its residual density onto linear functions. This changes the projection family in T38. It is potentially useful only after T37 supplies a small reference geometry.

## Source evidence

[Hilton's halfspace paper](https://www.alignment.org/content/files/2026/05/deductive_estimation_for_random_halfspace_intersections.pdf), dated May 2, 2025 and posted May 2026, equation (1), uses `g_i = P_D(H_i g_{i-1})` on the sphere. Section 3 chooses projections to preserve inner products with future factors, with an average-case statement for random halfspaces. It permits signed approximations. Section 5 computes updates by rotations and spherical monomial moments.

**Our extension, not a result claimed in that paper:** replace the fixed global polynomial space with `H_i` times linear polynomials, using the inner product weighted by `H_i`. This preserves the newest constraint exactly and closes each update with two-hemisphere moments. It does not require ARC's private implementation.

Two rejected duplications: Wu's [2025 thesis](https://gabrieldwu.com/assets/thesis.pdf), section 6.4, uses Gaussian halfspace conditioning and component pruning, substantially the E13 family. The [May 2026 cumulant paper](https://arxiv.org/pdf/2605.05179v2), section S.4.3, factorizes the same truncated estimator; factorization alone cannot fix its bias. Table 1's displayed K=3 factorized terms at n=16,L=10 sum to 13,885,440 FLOPs, about 2314 FPE at 6000 FLOPs/FPE, before omitted terms and an accuracy endpoint. This is illustrative arithmetic, not an RNN FLOP bound; T35 already owns this direction.

A public [Team Puffi write-up](https://discourse.aicrowd.com/uploads/short-url/yQsl71y2a3zDgb4SpYQFnt9EpoE.pdf), August 2026, equation (2), instead approximates fourth cumulants through a few covariance-response matrices. Its coupling tables and drift corrector are fitted offline, and its own deepest-layer comparison still loses to sampling. That is a distinct closure but not a ready weights-only solution for trained RNNs. It reinforces the value of testing projection families; no coefficients or results from it are imported here.

## Exact formulas for the proposed primitive

Let x be uniform on S^(n-1), n>=2; unit v defines H_v(x)=1[v·x>=0]. Write

    c_n = Gamma(n/2)/(2 sqrt(pi) Gamma((n+1)/2)),
    psi(x) = (1,x),
    q(x) = H_v(x) (a+b·x).

The hemisphere Gram matrix is exactly

    G_v = E[H_v psi psi^T] = [[1/2, c_n v^T],
                              [c_n v, I/(2n)]].

Given old support H_u and coefficients beta=(a,b), the next coefficients are

    beta_new = G_v^(-1) M_uv beta,
    M_uv = E[H_u H_v psi psi^T].

This is the weighted least-squares projection of `H_u(a+b·x)` onto linear polynomials under measure `H_v dmu`. Thus `H_v` stays exact even when the polynomial is signed. The scalar mass is `a_new/2+c_n v·b_new`. Because the basis contains the constant, it equals the incoming approximate density integrated against H_v; projection does not itself alter that step's mass.

All of M_uv is analytic. Put theta=acos(u·v), L=pi-theta, p=L/(2pi). Away from coincident/opposite normals, set

    w=(u+v)/(2 cos(theta/2)), z=(u-v)/(2 sin(theta/2)),
    m = (c_n/2)(u+v),
    S = (p/n) I + sin(L)/(2pi n) (w w^T-z z^T),
    M_uv = [[p,m^T],[m,S]].

**Derivation:** the first two coordinates in the span of u,v have uniform polar angle; their surviving angular interval is `[-L/2,L/2]` about w. Their squared radius has expectation `2/n`, and their radius has expectation `pi*c_n`. Integrating 1, cos(phi), cos²(phi), sin²(phi) gives the displayed moments. Remaining coordinates contribute `p/n` each. At theta=0 or pi use the continuous limits; do not normalize a zero vector.

The solve is explicit. If r=M_uv beta, then

    a_new = (r_0-2n c_n v·r_x)/(1/2-2n c_n²),
    b_new = 2n(r_x-c_n v a_new).

Use the rank-two form of S, avoiding dense inverses. Each update requires O(n) arithmetic plus acos/sin and vector normalizations. If implemented with dense matrices it still costs O(n²), but should be charged as implemented. Arbitrary nonzero means or affine offsets are not covered by these hemisphere formulas.

## Why this differs from the global degree-two rule

For the global projection, starting from g=1, the first truncated hemisphere is `1/2+n*c_n*v·x`; the degree-two term vanishes by parity. Applying the same hemisphere a second time gives `1/4+n*c_n²`, while its opposite gives `1/4-n*c_n²`. At n=10 these are 0.417301212 and 0.082698788 instead of 0.5 and 0. This is a direct correlated-constraint obstruction to naively importing the random-halfspace guarantee.

The supported rule gets *every pair* exactly, including duplicates/opposites. It does not preserve all previous supports. Non-adjacent repeated constraints and different constraint orders can still produce bias. Exact duplicate/contradiction removal is necessary preprocessing, not evidence of general accuracy.

## Results and reproduction

Synthetic coplanar halfspaces in R^10 have exact intersection mass `(pi-angular_span)/(2pi)` when the span is at most pi. No trained model or sampled sphere points are used.

| Ordered normal angles (degrees) | Supported estimate | Exact | Bias |
|---|---:|---:|---:|
| 0, 0 | 0.500000000 | 0.5 | 0 |
| 0, 90 | 0.250000000 | 0.25 | 0 |
| 0, 180 | 0.000000000 | 0 | 0 |
| 0, 45, 90 | 0.250271421 | 0.25 | +0.000271421 |
| 0, 90, 45 | 0.246649911 | 0.25 | -0.003350089 |
| 0, 30, 60, 90 | 0.252677959 | 0.25 | +0.002677959 |
| 0, 90, 0 | 0.208650606 | 0.25 | -0.041349394 |

```sh
python3 m16/approaches/crossover_compute/literature/hemisphere_check.py
```

The script also checks n=3 and rotation invariance. These tests establish a tiny algebra implementation and exhibit its failures; they do not validate an estimator on AlgZoo.

## Recommendation and cheapest next test

T38 is now held by the Claude DRI (`harmonic_sphere`). Hand these formulas and warnings to that owner; do not start a competing experiment. The following is a suggested bounded check for their consideration, dependent on T37 producing homogeneous reference constraints or on an existing small cone set. Compare the supported rule with global D=1/2, ADF and exact 2D/3D spherical volumes on 100 *predeclared deterministic* cone geometries, including thin wedges and near-redundant constraints. Check forward/reverse and fixed geometric orderings. Stop if ordering spread or actual bias is too large for the intended outer sum. Charge preprocessing and every order tried. No noisy-copy or test-truth tuning.

**Alternative 1:** most error comes from unsupported mass across the newest gate. Pair exactness tests and comparison against global projection can falsify the proposed repair.

**Alternative 2:** past constraints, not the current one, dominate the bias. Non-adjacent duplicate tests already demonstrate this failure mode; retaining two exact supports is a possible extension, but its next update requires three-hemisphere polynomial moments and must be separately costed.

A single cone with at most 200 constraints is plausibly cheap enough for <1000 FPE (6M FLOPs). This is a generous target, not a measured claim. The principal obstacle is **how many cones the RNN needs**, not this primitive's arithmetic. Each gate normal depends on the previous gate history through h=A_history x. Summing projected input densities across histories without retaining their A matrices does not define the next gate. Therefore T38 cannot simply apply 160 fixed halfspace updates and claim to integrate the entire network. A reference mechanism, state reduction, or another explicit closure is still required. Likewise, a noisy-weight ensemble is not an independent ensemble of future factors in Hilton's theorem; constructing its optimal projection basis is a new computational problem.

## Status

Done: formula derivation and synthetic check only. The check was completed before the parent relayed the DRI’s new T38 reservation; no further algorithm experiments were run. This is an auxiliary memo for that task, not a claim to an independent workstream. No solution claim.

## Compute accounting

0 model forward passes; 0 random model inputs; 0 LP solves; 0 training. Eighteen tiny synthetic cone calculations including rotation checks, plus arithmetic and assertions, under one second of shell execution. Web reading and derivations are research overhead. No experiment uses model truths.

## Claims

- [measured] The exact-pair and rotation checks pass; synthetic failures are retained above.
- [inferred] Global degree-two spherical projection is inadequate for arbitrary strongly correlated constraints.
- [speculative] Retaining selected exact supports can outperform global harmonic projection on the particular small reference complexes needed by T37.
- [inferred] Neither the full cumulant paper nor the public contest write-up supplies a proven cheap trained-RNN estimator.

## Alternatives not ruled out / failure modes

More exact supports may help but erase the cost advantage. Signed densities may give negative probabilities or values above one; clipping hides rather than fixes this and is not used. Reordering can flatter estimates by chance. Affine residual-Gaussian models require different moments. Low-cost single-cone integration does not address the exponential set of recurrent gate histories.
