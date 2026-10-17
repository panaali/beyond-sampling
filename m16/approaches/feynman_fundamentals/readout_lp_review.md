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

## Gate-history attribution source and arithmetic

The attribution source uses one common dyadic weight denominator D.
Its integer hidden state and affine map at step t represent D^t times
their real counterparts; the input injection carries the matching
power of D. Direct recurrence and mapped recurrence therefore agree
without rounding. Exact gate signs are checked on every simplex ray,
which establishes the fixed gate history throughout each cone. Correct
simplices additionally check direct second-largest/prediction agreement
at strict input-order interiors. This reviews the classification logic;
the existing geometry and mass engine are reused, not re-enumerated.

Independent60-digit sums in `check_training_bulk_review.py` recover
final-all-off error .0223244065579057694, earlier-all-off error
.0003777396708759991, exact contrast-column-erasure-without-reset error0,
and remaining error .50028472777709983. Total error is
.5229868740058815985; remaining accounts for95.6591365%. Gate mass sums
to .9999999999999993022 and correct mass .4770131259941177037.

The contrast test means that a given input column is exactly equal
across every affine logit map in that gate cell. It is stronger than
output-label insensitivity. A label can remain constant while nonzero
contrast coefficients vary, so these results do not exclude broader
output forgetting or approximate-memory explanations. The proposed
constant-label gate-cone rescue should use all-ray nonnegative margins
and at least one positive ray against earlier-index competitors for
strictness in the cone interior. Requiring all boundary rays strictly
positive would incorrectly reject valid full-measure interior labels.

## Independent direct recurrence and correctness certificates

`check_constant_cones_review.py` evaluates the network by direct integer
ReLU recurrence on4252 ray inputs from all450 saved histories, without
importing the author's final-map or certificate code. It reconstructs
every constant-label decision:183 constant interiors and267 varying
ones. Summing each cone's rays gives a strict interior input witness;
all four output labels have such witnesses. Actual recurrence
reachability is therefore established for these four labels, beyond
the earlier nonnegative-superset LP result. Geometry itself is still
the independently checked pre-existing artifact.

The extended conservative correctness rule also checks independently
on all450 histories. A competitor dominating every ray excludes a
candidate output, with earlier-index ties handled weakly and later
competitors requiring strict interior advantage. An input cannot be
second largest if two others are always above it, or if no other input
can ever be above it. These produce supersets of possible labels.
Disjoint output/target sets prove wrong classification; matching
singleton sets prove correctness. The review reproduces48 wrong,
39 correct and363 unresolved histories. The reported resolved mass
is only about .005750413, leaving .994249587 unresolved before numerical
mass qualifications. This weak shortcut is insufficient on the saved
model, not an impossibility proof for tighter joint-label constraints.
