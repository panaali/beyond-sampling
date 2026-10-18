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

## Counter provenance and fixed-panel arithmetic review

Independent source AST comparison confirms imported rows_at, _roots_vec,
Cons and trace_arc_rows differ from the original-author helpers only by
pure counter X calls. Saved current n=original n+recorded X holds for all12
cases, including seed0 original211218. Independent aggregation of every
site ledger, rational cost ratios and conditional squared-bias comparisons
passes. Under the matched explicit-expression/112 convention, trained
sample-equivalent budgets are1.872486 to1.887998 times original-author/72,
versus1.108565 to1.151513 times the newer helper-adjusted counter/72.
These source eras must not be interchanged.

Using the separately reviewed T114 polygon reference on the same fixed12,
the largest conditional squared-bias/MC-variance ratio is.19237642029
(seed1 clean), followed by.08484327908 (k24seed4 clean). This compares a
fixed deterministic estimate with fractional equivalent sampling budget;
it is not a noisy-panel ARC pass, complete hardware cost or numerical
truth enclosure. Constant first-index output controls retain1/3. Native
instrumentation preserves every saved output/trace/statistic on this
panel. Previously known approximation failures remain part of the record.

The no-change cap iterator yields exactly0..63. It flags exhaustion only
when the loop requests the next value after64 iterations, so a break or
return during the final allowed iteration is not falsely labeled a cap.
The AST check replaces exactly forward_trace and reverse_walk. Fresh
module/value/counter parity plus no exhaustions on the fixed12 (maxima9
and4) exclude those particular64-step caps as an explanation there;
this is not a universal termination or topology proof. Tiny-angle and
nearzero approximations remain separately disclosed.

## T134 frozen471-instance common-unit audit

Pinned loader order installs instrumented fiber3/fibre definitions before
consumers import them and restores registry entries in finally. Current
helper-X versus old helper AST equivalence and per-model original counter
recovery remain intact. The same explicit-source sampler convention gives
4H^2+12H operations forT3:112,160,216 atH4/5/6. Group ratios compare equal
empirical mean squared bias with equal mean per-model matched-budget
sampling variance; they are neither mean individual ratios nor one common
fixed budget. Model-free declared preprocessing is separately free.

`check_fibre_suite_review.py` independently checks25 pinned manifest files,
all471 identities/prior estimates/old counters,942 site-ledger aggregations,
and every individual and21 group ratio using rational arithmetic on saved
binary64 values. All groups remain below1; largest individual ratios by
H4/5/6 are.19237642029,.003886719292,8.850394381e-9. Review CPU.2088,
no estimator rerun or new model/noise/truth generation. This preserves
finite existing-suite evidence under the explicit source-expression cost
convention, not full implementation/hardware cost. Very small H6 errors
are against saved floating truths without interval certificates; no new
broad ARC or all-noise-population success follows.
