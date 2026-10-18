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

## Native-expression phase and matched sampler

The second AST wrapper retains arithmetic operands, comparison-chain
short-circuiting, in-place updates and actual sorting comparisons. It
omits historical counter calls themselves. Source inspection finds the
pinned reductions/special functions covered, including _Sint. Direct
inventory confirms nonzero _Sint20 operations versus the inherited17;
zero case2. These counts assign one per special call and n-1 per Boolean
or extremum reduction; they are abstract source operations, not cycles.

Review found that floating-only totals exclude integer class/true-label
comparisons although sampler112 includes correctness. The matched sampler
now runs through the same AST and yields111 floating plus1 integer. The
inclusive inventory is consistent under its explicitly code-dependent
convention: count every explicit numeric expression, including explicit
index arithmetic; implicit loop/branch machinery, memory/selection/cache
lookups and interpreter work are free on both sides. It is not a complete
implementation/hardware count or a compiler-independent metric.

Independent saved-site aggregation and rational ratio recomputation pass
all ten cases. Inclusive trained ratios versus old hybrid/72 are.805806
through.858590; analytic fixtures1.977350 and1.265606. No new MSE verdict
follows from this finite audit. The first raw checker used absolute4e-16
agreement and rejected zero's two-stage floating division difference
4.440892098500626e-16. This was a reviewer tolerance issue, not a count
mismatch; exact rational count unchanged, validation now allows two ulps.
