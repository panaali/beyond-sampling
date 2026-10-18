# T185: five-dimensional normalization and saved evidence

Frozen source SHA a88f62c41a94af636399675e92ed0ce88f109808f66eff23fdbddb9a87ef2bad.
This is a source and saved-data review, with no new mass, model, or random evaluations.

For five independent cone rays, radial integration gives
`3 |det R|/(8 pi^2) integral_simplex (u^T R R^T u)^(-5/2) du`.
The implemented four-dimensional Duffy map has Jacobian
`(1-a)^3 (1-b)^2 (1-c)` and the correct normalization. A small
determinant alone does not certify the fixed six-point tensor rule: nearly
opposite rays can have a small determinant and substantial cone mass.

For exact unit-diagonal positive-definite correlations, Plackett's derivative
for a pair is `r/(2 pi sqrt(1-t^2 r^2))` times a conditional three-dimensional
orthant probability. Its constant term is 1/8 and its three angle terms have
coefficient 1/(4 pi). Integration yields exactly the source's base
`1/32 + sum asin(r)/(16 pi)` and residual factor `1/(8 pi^2)`.
Each conditional pair is correctly obtained from its four-coordinate block,
conditioning on the selected two coordinates. The relevant Jacobi identity is
`cof_kk cof_ll - cof_kl^2 = det(Q) det(Q_without_k_l)`; the latter determinant
is `1-t^2 r_ij^2` under the exact unit-diagonal premise. The block implementation
has this interpretation; the introductory full-matrix identity must not be
used in its place. Floating diagonal drift, determinant floors, tiny-denominator
floors and cofactor cancellation are outside this exact identity review.

For the ideal residual path, truncation after `t0=1-exp(-70)` has absolute tail
at most `3/(16 pi) sum_pairs[asin(|r|)-asin(t0 |r|)]`, hence at most
`15/(8 pi) acos(t0)`, approximately 5.32e-16 per cone. This says nothing about
interior fixed-rule error or accumulated floating error.

The independent saved audit checks all 14 file hashes and 60 validation rows.
The nominal one-model ratio is 0.009706984995, but inversion of the documented
10-million-draw binomial tails gives a ratio interval from zero to about
106.648. The hit count is reconstructed from the saved scalar and source
documentation, not a retained draw ledger. Independent binomial-tail inversion
agrees with the author's beta inversion within 4.75e-14 in probability;
neither is an outward-rounded certificate. Five reconstructed log64 values
are negative. Agreement of thin6 with thin10 on these rows is not an error
certificate against exact masses.

Saved timing includes the leaf diagnostic mass work whose proxy operations
are discarded. It is not a pure recipe runtime comparison. Observed T4-to-T5
growth does not imply a T10 lower bound: the quoted 13.5^5 continuation is an
assumption. The earlier unchanged-source factorial-chart bound is a separate
valid scope and must not be confused with this extrapolation. The known
simultaneous-freeze and vertex-only certificate limitations also remain.

The separate Hopf precision reference source gate accepts the original
`mc_acc` body with only its final hit-count return changed. Completed six
deterministic controls account for 318 extra evaluations. Model identity,
completed-control hashes, RNG states and no-overwrite guards are required.
Only the frozen first-million pilot was cleared before the independent
throughput/resource gate; no accuracy-dependent stopping is authorized.
