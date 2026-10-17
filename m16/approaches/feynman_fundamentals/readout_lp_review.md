# T116 exact readout certificates and their scope

Independent `check_readout_lp_review.py` imports no author LP solver.
It reconstructs every constraint directly from the saved exact weights
and verifies20 trained/synthetic rows using Fraction arithmetic.

Normalize nonnegative hidden h by sum(h)=1. For candidate label j>0,
earlier competitors require a common positive margin epsilon; later
competitors permit weak inequalities. A feasible epsilon>0 is equivalent
to that label being selected somewhere in the nonnegative hidden cone.
For j0, the origin always selects0, so its optional positive-margin
optimization is diagnostic rather than a necessary reachability test.

Writing constraints A*z>=0 and equality e*z=1, saved nonnegative dual
multipliers satisfy target=bound*e-mu*A exactly. Therefore epsilon is
at most bound. The saved primal achieves that bound, certifying the
reported optimum. All20 rows pass normalization, nonnegative slacks,
dual sign/stationarity and matching objectives; positive trained
witnesses also reproduce their stated first-index argmax. Review CPU
is .063 seconds, including reading the compressed source artifact.

The first review script attempt passed JSON string numerator/denominator
values directly to Fraction(a,b), which rejected them with TypeError.
Explicit integer conversion repaired this input parsing issue before
any certificate result. This was a reviewer implementation defect, not
evidence against the LP or the method.

All four trained labels are possible in the nonnegative hidden superset.
This rules out only a structural readout-cone impossibility. The actual
recurrence may reach a smaller subset, and no input witness or reachable
class mass follows from these LP witnesses. The source can also raise
when normalized weak-later constraints are infeasible; the saved cases
do not test a fully general infeasibility-classification interface.
