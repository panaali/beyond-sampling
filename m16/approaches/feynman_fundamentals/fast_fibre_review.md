# T128 fast fibre primitive source review

For a prescribed gate pattern, flipping gate(t,k) changes later maps by
rank-one propagation: initial coefficient +/-Wh[:,k], then Wh*(g*c),
and the readout Wo*(g*c). Sorting multiple flips by time ensures later
flips use already updated preactivation rows, retaining cross terms.
Same-step preactivation rows do not depend on those step's gate choices.
Thus inherited cached ranges remain valid through the earliest flipped
step; recomputation beginning at(tmin+1)*H is correct. Projections obey
the same linear updates. The finite64-subset dyadic controls check this
algebra; they do not certify tolerance-based child-group topology.

_range evaluates endpoints plus stationary maximum/minimum angles.
This is the correct real-arithmetic range on the pinned arcs shorter than
pi. A gate whose range misses(0,maxclamp) cannot cross any valid fibre;
nonoverlapping range intervals make an ordering sign constant. Thus R1
and R2 pruning are conservative in exact arithmetic. Floating root tests,
1e-12 same-crossing grouping, nearzero cutoffs and finite tracing behavior
remain separately approximate. No global slow/fast equality is inferred
from these primitive controls or the source's historical docstring.

The native inventory extensions count outer products, min/max, hypot,
array equality and explicit integer division under the previously stated
source-expression convention. It is still code-dependent and excludes
implicit control/memory costs. The currently imported fibre helper adds
X charges directly to n; therefore its n is not automatically the old
T95/T99 reported counter, even when fibre_fast itself is unchanged.
Source-era count reconciliation is pending before historical ratios.
