# Independent T76 review

Scope: the zero-prefix, final-two-input surrogate, not the original network.
Reviewed suffix.py SHA256 ce16872b58fa8112fcccb6a1f89d7ca5ed13fbba99f6b3b35373231604b19ffe.

## Separate numerical falsifier

`check_suffix_review.py` constructs all readout-pair boundaries under all four
final gate masks, rather than using the author's upper envelope. It evaluates
the midpoint label by a direct two-step recurrence. At each independent GL24
angle it integrates the original Rayleigh density r exp(-r²/2) on [0,12],
using normal CDF rank probabilities rather than any closed radial formula.
The omitted radial tail is <=exp(-72); the adaptive and angular quadratures
are numerical checks, not directed enclosures.

| case | independent surrogate | difference from author | difference from full reference |
|---|---:|---:|---:|
| T4 RMS .02 copy0 | .2949371023857567 | -7.93e-12 | +.0312691904942534 |
| T4 RMS .10 copy0 | .2506648752749391 | +3.46e-11 | +.0029253038303131 |
| T5 seed2 | .2935052654904501 | -2.00e-12 | -.0112701932916673 |

The .10 copy0 case is a passing transfer control, not a failed case. Constant
readout and a zero-recurrence sign predictor reproduce exact .25 and .3125
within1.12e-16. Total324072 radial integrand evaluations,2352 angular nodes,
1.071823 CPU seconds. No random model inputs, training or geometry of the
full model. Reproduce with the review lock and single-thread Python, as for
other scripts in this folder. Full output is `suffix_review_results.json`.

[inferred] The .02 copy0 and T5 seed2 failures are genuine approximation bias
from replacing earlier state by zero, subject to the independent full-model
reference qualifications. They are not explained by the author's radial
formula or missed envelope sectors at the observed precision. This verifies
two particular negatives; it does not reject boundary methods generally.

## Algebra, tie and cost inspection

The all-positive-radius homogeneity separates angle, label and rank weights.
The rank formulas and Rayleigh closed forms in boundary_history_certificate.md
match the implementation. Both first-input signs, final gate roots and the
suffix order planes are split. Nonzero axis readout labels and the origin's
first-index tie are retained; full-plane pair boundaries outside active gates
would only add redundant cuts.

The portable cost uses scalar arithmetic-circuit accounting, not instruction
or CPU counts. The inspected charges cover envelope pair construction/sort,
first-step affine maps, roots, sector readouts, elementary radial functions,
rank accumulation, history setup and all margin/activation checks. In
particular radial n2/n3 charges90/125 per node exceed a direct count roughly
73/101 with the TRANS20 convention; angular80 includes sin/cos40, and the
history charge100+24T covers its fixed-width per-competitor loop. gauss_rule
is the separately counted existing rule. No opaque QR/SVD/kernel appears in
calculate; diagnostic inventory.describe's spectral norm is not called by
it. This is a source upper-count review, not an independent benchmark of every
reported equal-cost ratio or a certification of elementary arithmetic.

The history error bound is algebraically sufficient, but its angular integral
is explicitly only a numerical diagnostic. Its near-zero agreement lower
estimate means that this certificate is weak, not that the true predictors
almost always disagree. A rigorous lower angular sum would still be needed
for certification. Large absolute recurrent gains erase cancellation and can
make a box/Lipschitz bound vacuous even when average approximation is useful.
The documented conservative angle-cell minima provide a distinct possible
certificate refinement, though they cannot repair a truly large bound.

## Alternatives and next finite falsifier

Closed-form integration and partition bugs were tested separately above.
Other full-model reference errors remain subject to their own certificates.
More accurate quadrature cannot remove the independently reproduced surrogate
bias. Two promising changes have not been rejected: retain three suffix
inputs, or replace zero history by a weight-derived state approximation.
The former is now owned by T76; this reviewer does not implement its engine.

For a three-input radius R~chi3 and J_n(a)=integral_0^infinity phi(r)Phi(ar)^n dr,
integration by parts gives F_n=E Phi(aR)^n=2(J_n+a J_n'). Therefore

    F1 = 1/2 + [atan(a)+a/(1+a²)]/pi,
    F2 = 1/4 + [2atan(a)+asin(a²/(1+a²))+2a/(1+a²)
                +2a²/((1+a²)sqrt(1+2a²))]/(2pi).

These formulas hold for negative a too. Use the largest and SECOND-largest
retained inputs in the rank weights; a predicted third-largest suffix input
cannot be second-largest overall. Fixed three-step gate/readout/order planes
on S² and positive two-dimensional triangle quadrature are an economical
candidate. All geometry/nodes must be charged; no favorable cancellation or
exact full-model claim follows merely from one extra input.

A separate cheap control is the last-input-only predictor. Let j+ and j- be
its first-index output labels on positive/negative last input. Its accuracy is

    1/T + (1[j+=last]-1[j-=last])
              * (1/2-(T+1)/2^T)/(T-1).

This follows by integrating the last input's rank density on each half-line;
all earlier labels are exchangeable. The tested T4 toy has j+=last,j-=0 and
accuracy.3125. This value is not the trained T4 baseline: its last label is
unreachable, so that model's corresponding control is.25.

## Independent three-input rescue check

Reviewed suffix3.py's fixed octant/gate/readout/order splitting. Successive
linear gate maps, adjacent upper-envelope planes on the nonnegative quadrant,
and all three suffix order planes suffice in exact arithmetic. Positive
spherical density is |detV|/(4pi||Vy||³) times the triangle coordinate measure;
the Duffy factor1-u is present. The chi3 rank formulas above are used correctly.
Exactly identical readouts retain first-index ties. Floating thin triangles
and readout/gate rounding remain qualified; the author's small midpoint
mismatch masses must not be silently discarded.

`check_suffix3_review.py` independently uses a tensor S² rule and direct
three-step recurrence, without any author clipping or boundary helper. On the
same failed T4 RMS.02 copy0 model:

| sphere rule | surrogate accuracy | full-reference difference |
|---|---:|---:|
|128x256|.2893959027359555|+.0257279908444522|
|256x512|.2894181217090186|+.0257502098175153|

The author clipped GL8 value is about.28945814. This check does not establish
high-precision equality: its remaining difference is about4e-5. It does
independently reproduce a2.57pp failure, hundreds of times larger than the
observed change/difference. Constant-readout rank controls err3.63e-7 and
1.31e-7, respectively. Work163840 deterministic three-step quadrature forwards,
.032894CPU, zero random samples. This independently exercises the strongest
implemented rescue; it does not exhaust nonzero deterministic history states
or all possible boundary approximations. A weight-derived nonzero mean state
would break radial homogeneity and require new affine/radial integration and
cost accounting; it remains open rather than implicitly ruled out.
