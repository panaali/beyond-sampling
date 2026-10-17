# T67 penultimate cache: source and finite outcome review

Read-only review of `penultimate_cycle/cache.py`, `driver.py`, inherited
`target_last_cycle/terminal_cache.py:values`, and pilot 4da6686.

The chronological substitution is correct. From fixed h_(T-2), the cache
first inserts target u, then integrates the final independent scalar x.
First-stage gate roots partition u; left-closed/right-open ownership
assigns each internal boundary once. ReLU continuity gives the same
hidden state there. Second-stage active gates are strict, inactive gates
weak; identically zero gates are inactive. Readout inequalities are strict
against earlier-index competitors and weak against the later label. This
preserves positive-volume first-index ties.

The resulting original affine halfspaces are verticalized in x. Their
line intersections, support limits, u-only constraints and first-u roots
give the complete topology in exact arithmetic. The inherited query
routine depends only on these pieces and below/above support denominators;
it has no hard-coded target-last chronology. The preceding backward
recursion correctly uses m=T-t-1 ordinary inputs and the mixture
`(m-1)/m * below_need1 + above_need0/m`. Only the penultimate label takes
the new cache. Other label contributions are unchanged at matching nodes.

`check_recent_aggregates.py` independently contracts all four saved rows
with 60-digit Decimal arithmetic. GL4 baseline is 0.30772324883398353;
cached is 0.3042118647759948. The penultimate component error shrinks from
about +0.003055119 to -0.000456265, but the other components retain their
errors, so total error worsens from -7.49419e-6 to -0.00351887825. This
agrees with the earlier independent per-label reference and demonstrates
loss of accidental cancellation, not an arithmetic inconsistency.

The cached GL2/4 difference 1.77167e-7 fails the predeclared 1e-7 gate.
The finite source and direct-scalar controls therefore establish a useful
component repair, not full numerical convergence or a total-accuracy win.
Floating root ordering, CDF subtraction, spatial interpolation and
portable full cost remain uncertified. No larger sweep is implied. A
future combination must be predeclared rather than selected label by
label using the reference truth.
