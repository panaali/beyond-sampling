# T121 first bounded common-unit source audit

The pinned slow R4 source executes dense matrix products. The AST wrapper
retains operands, ordering and a@b evaluation; its 18 sites include the
actual rows_at gate-mask products. For an output dot of length k the
separate scalar count k multiplications plus k-1 additions is correct.
The rows_at historical structural charges match their explicit ops.add
calls. Replacing those charges avoids double counting; other sites use
dense dimensions already present in the old counter. Composite charges
such as 11 around three length-three dots retain their two divisions.

Independently, one H4T3 dense forward with known initial zero state has
56 multiplications, 41 additions and 15 comparisons: 12 ReLUs, two
predicted-argmax comparisons and one correctness comparison. Its true
label is supplied free under rule5. In general the total is
4*T*H + 2*(T-1)*H^2. Thus112 is consistent with the scalar convention.

check_cost_rotation_review.py independently sums every saved site and
checks dot dimensions for all ten frozen cases, recomputes the partial
replacement and rational budget ratio, and verifies saved parity fields.
It does not rerun the estimator or certify the unaudited operations.
A zero rows_at alone executes168 multiplications,123 additions and36
mask products; its retained x20 makes347 named scalar operations,
versus historical n24+x20. This is a concrete missing dense-work issue.

On the eight trained cases, partial corrected sampling-equivalent cost
is .78185 to .82641 times the historical hybrid/72 value: both numerator
and denominator change. The two analytic fixtures instead increase.
These are partial inventories. Remaining scalar/vector operations,
comparisons, special functions and model-dependent setup require their
own audit before a complete common-unit number or leaderboard claim.
Rule4 model-free declared preprocessing stays exempt.
