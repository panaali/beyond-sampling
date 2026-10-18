# T266 independent time-reversal residual contract review

## Question
What does input-law-preserving reversal prove about residual pairing, and what remains necessary for a cheaper accuracy calculation?

## Method
Written review of source1cdbc9be/shared e827a54b under delegation9a5fe201. Check six text snapshots against exact original Git bytes, frozen documentation hashes, algebra, domain/tie conditions and cost interpretation. No source imports or scientific data/evaluation calls.

## Status
Written source/domain contract passes; execution remains false. A fixed deployable surrogate with justified integral and a certified paired partition/integral remain concrete missing prerequisites.

## Results
[inferred] Reversal is an orthogonal involution preserving N(0,I), independently of network weights. With distinct input values, second-rank labels reverse; first-index ties require actual recomputation and are not equivariant in general. Output ties remain relevant and cannot be dropped. No neural equivariance follows.

[inferred] For bounded measurable r=Y−S, the symmetrized p=(r+r∘R)/2 is invariant and has the same expectation. Thus pairing cannot change the signed bias from dropping the residual. The pointwise triangle gap is min(|r|,|r∘R|) on opposite signs and zero otherwise; integrating yields exactly the stated absolute-budget reduction. Binary residuals reduce it by the probability of nonzero opposite signs. S=0 or1 produces one-sided residuals and no such gain. Any approximate surrogate integral contributes its own error.

[inferred] D={x0<x_(T−1)} and RD partition the law off a null hyperplane. Changing variables in the RD integral gives E[r]=∫D(r+r∘R)dmu. Since p is invariant and mu(D)=1/2, this also equals E[p|D]; the same normalization applies to |p|. Both function evaluations and any intersected classification complex remain required. On an invariant retained set K, the omitted absolute-integral bound plus retained integration errors is valid. Conditioning on one non-invariant prefix or rank chamber does not inherit the unconditional identity.

[inferred] Equal marginal variances give Var(p)=sigma²(1+rho)/2. At2n independent-single versus n paired evaluations with exactly doubled cost, mean-variance ratio is1+rho. At deterministic budgets it is(m/n)(1+rho)/2 using the two floor counts. Both formulas require independent sampled pairs and their stated fixed-cost assumptions; setup, adaptive stopping or variable runtimes need separate treatment. Zero variance leaves the ratio undefined. These sampling identities imply no deterministic quadrature gain.

[inferred] The bounded source inventory accurately identifies missing paired complexes and deployable-surrogate input. The per-input map obtains gates via run(x); the old cone writer retains aggregate checks/rules, not complete maps; captured recipe-prefix cones disclaim true-domain completeness. Rank-chart and historical control-variate results are not revalidated or silently promoted to current cost/geometry certificates. The inventory is not a whole-repository absence theorem.

## Compute accounting
Metadata hashing and Git-byte comparison only. Zero models, arrays, CDFs, geometry, mass, RNG, fits or author-module imports. No historical score/cost contraction. No runtime benefit was measured.

## Claims
[inferred] The conditional requirements and symbolic identities are correct. Smaller absolute residual bounds may be useful, but actual integrated signed error is unchanged by pairing alone. A favorable covariance is neither a deployable cheap surrogate nor a deterministic proof.

## Alternatives not ruled out
Other fixed surrogates, certified paired strata or direct integrals could pay for their construction. Antisymmetric, symmetric, one-sided and tied-input symbolic controls would test specific algebraic implementations under separately frozen scope. No full family is ruled out.

## Failure modes
Incorrect label reversal on tied fixtures, using a conditional non-invariant law, failing to charge both evaluations/refinement, selecting pairings after scoring, or treating old source claims as renewed empirical evidence would invalidate an application. No numerical continuation is authorized by this written review.
