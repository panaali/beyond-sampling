# First-mismatch certificate for a frozen rank-chamber gate history

This is an unimplemented alternative motivated by T28's independently verified
large frozen-gate bias. It keeps the exact rank-conditioned input law. It is
not a rescue result or a claim that the resulting bound is useful.

Fix largest/second positions, a second-quantile stratum, and a deterministic
gate template D_t. Propagate symbolic preactivations through this template:
Z_ti=a_ti^T X (add a constant term if the model has biases). Let s_ti=+1 for
an active template gate, -1 otherwise. Before the first actual/template gate
mismatch, actual and template hidden states agree. Therefore the first
mismatch must satisfy s_ti Z_ti<=0 for one template preactivation. We may use
the weak inequality conservatively; identically zero inactive forms are safe
and should be excluded rather than charged probability1.

Consequently actual/frozen prediction disagreement, and the absolute
conditional accuracy bias, are bounded by

    min(1, sum_ti P(s_ti Z_ti <= 0 | fixed chamber/stratum)).

This bound does not assume later actual trajectories follow the frozen map.
It uses the earliest mismatch, where the affine preactivation is still valid.
It can be very loose: many individual violations can overlap, and a gate
mismatch need not change the prediction.

If the template is defined by the conditional mean input, signed margin
m=E[sZ]>=0 and variance v=a^T Cov(X) a. Cantelli gives v/(v+m²) when m>0;
use1 for zero margin with nonzero variance, and handle v=0 deterministically.
The moments need rigorous error control for a numerical certificate. A
floating moment table gives only a conditional real-arithmetic diagnostic.
Computing this bound from existing template Jacobians is cheap and may be
vacuous; it would be a falsifier of certificate usefulness, not the method.

A sharper alternative conditions on b=second input. Write q=Phi(b). Given
b and the two positions, the maximum is normal truncated above b, the eight
labeled lower inputs are independent normals truncated below b, and the
second input is fixed b. Their moment-generating functions are

    M_lower(t)=exp(t²/2) Phi(b-t)/Phi(b),
    M_upper(t)=exp(t²/2) Phi(t-b)/Phi(-b),
    M_second(t)=exp(tb).

For Z=sum c_j X_j, multiply these at arguments t*c_j. Chernoff bounds the
adverse event by min(1, inf_{theta>=0} E exp(-theta*s*Z | b)). This is a
one-dimensional convex optimization in log-MGF; a finite theta already
provides a valid real-arithmetic upper bound, so optimization failure must
not fabricate an underestimate. Evaluate normal log-CDFs in the tails.
Integrate this positive bound against the Beta(9,2) density for q over the
fixed stratum, divided by its probability. Equal-probability half strata
have normalization factor2. Sum across gates, then cap at1, or integrate the
pointwise capped sum for a tighter union bound. No Gaussian approximation to
the rank law occurs. Conditional mean margin can have either sign; the
optimization includes theta0 and thus returns at most1, never a negative
probability. Degenerate deterministic forms need explicit cases.

Necessary controls before implementation acceptance: constant/zero gates,
no-recurrence exact template, single-coordinate thresholds with known
truncated-normal probability, strict upper-bound checks against those exact
cases, numerical log-MGF convexity and theta0=0, positive quadrature error,
and full Jacobian/optimization/cell costs. Do not compare a heuristic
integral of upper bounds to truth as if it were rigorously enclosed.

Bounded usefulness test (Cycle39): the cheap Cantelli union was evaluated on
all180 fixed rank/quantile cells for DEV .0025 and .02 copies, reusing the
pinned model-free moment table. Every cell saturates at1. Raw union minima
24.0671/25.6363, means66.6878/68.3453. Thus this certificate is entirely
vacuous on both models; the average individual gate bounds .41680/.42716
must not be mistaken for prediction-disagreement probabilities. Cost.571954
CPU after imports, zero model input samples. Conditional Chernoff, first-hit
and output-sensitive alternatives remain untested. Numerical moment/error
qualification remains as stated above. See check_gate_certificate.py and
its raw gate_certificate_results.json.
