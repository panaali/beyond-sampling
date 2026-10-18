# Source accounting contract for a later instrumented check

Let M be kept prefix rows, H hidden width (16 on this leaderboard), T=10,
K=H+3 ordered interval positions, and A_k=1 if any row satisfies hi>lo at
position k. Let J=sum A_k. Source work depends on M and J, not only a typical
single prefix's number of nonempty intervals. Sorting and library internals
remain separately unresolved.

The dense dot-product skeleton is exact as a call-shape statement:

- Nine calls (M,H)@(H,H), one identical-shape last-step-offset call.
- Two calls (M,H)@(H,T) for each of the J active batch interval positions.
- No rank-one update kernel occurs on this path.

Under the explicitly declared dense convention2 FLOPs per multiply-add, the
skeleton is `20 M H² + 4 J M H T`. Under the mathematical dot-expression
convention H products and H−1 additions, it is
`10 M H(2H−1) + 2 J M T(2H−1)`. These are different conventions, not two timings.
At H16,T10,J19 the first gives17280M and the second16740M scalar units, BEFORE
all ReLUs, input terms, breakpoint arithmetic, inequalities, CDFs, reductions,
sorting, sampling/stratum work or setup. Neither is a complete multiplier.

The source-return equation is C_raw=ceil(float(M*c)) with binary c=1.6+0.1.
The existing harness charges `(3.3/1.6)*C_raw`. An analytical nominal comparison
using exact decimal coefficients is `(33/16)*ceil(17M/10)` versus
`ceil(34M/10)` if only Q1.6 is replaced by3.3 while draw0.1 is retained. Without
ceilings their difference is17M*33/160−34M/10=17M/160, approximately0.10625FPE
per kept row. Real runtime operations must retain their binary expressions and
separate rounded ceilings; this identity is not a replacement implementation.

Current leaderboard rule5 exempts one generated input per evaluation and
charges extra rejection candidates at0.04FPE each. If M kept prefixes count as
those evaluations, its literal extra-candidate allowance is0.04(D−M), not0.1M.
With one50000-candidate batch and minimum M28, these are1998.88 versus2.8FPE.
This is a symbolic small-budget control, not a measured historical violation;
actual D, allocation, reused setup and the DRI interpretation must be reported.
The raw candidate count is not present in old returned costs.

A future tracer must distinguish: float arithmetic; comparisons/Boolean
reductions; dense dot shapes; sorting/search operations; scalar special-function
invocations; RNG extra-candidate convention; allocation/weighting/ceiling;
model-free cached tables and original-model advice; indexing/copy/allocation and
native internals. Track branch masks and actual batch dimensions without
substituting a new algorithm. Complete cost is still unknown until an agreed
accounting scope covers these categories. No native-kernel chain is reopened by
this source inventory.
