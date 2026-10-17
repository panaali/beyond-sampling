# Independent nonzero-history suffix check

The T76 exact-two-step mean formula is valid, but deterministic substitution
for the prefix hidden state is not exact conditional history. Rank correlation
and hidden-state variation remain even with the exact unconditional mean.

Read-only review of nonzero_history/affine.py found the scalar rank integrals
consistent: below y, first-suffix truth is m(1-Phi(y))*Phi(y)^(m-1), while
last-suffix truth is Phi(x)^m; above y these become Phi(y)^m and
m(1-Phi(x))*Phi(x)^(m-1). Every omitted label gets the remaining probability
mass divided by m. Splitting accepted intervals at x=y is necessary. Its
power differences and survival-tail CDF differences improve cancellation.
An identically zero final gate is assigned inactive, and identical readout
rows use first-index argmax. Such positive-measure ties cannot be discarded
as generic boundaries. Floating plane/root classification remains qualified.

Independent check_mean_suffix_review.py imports only checkpoint loading,
computes the exact mean independently, and evaluates direct two-step ReLU
forwards. It uses no affine region construction or author probability kernel.
The fixed T4 RMS .02 copy0 reference is .2636679118915033.

|Rule|Zero prefix|Exact mean prefix|
|---|---:|---:|
|Gaussian128x129|.296692296770|.298922765675|
|Gaussian256x257|.295762016586|.296353178778|
|CDF midpoint1024x1025|.294990304387|.299914157667|
|CDF midpoint2048x2049|.294964863318|.299901995199|

The first rules have substantial discontinuous-label quadrature drift; they
are retained as a failed precision check. The distinct fixed CDF midpoint
rules match the author's refined surrogate estimates within2.8e-5 (zero) and
1.9e-5 (exact mean), and reproduce both the large history bias and worsening
under the exact-mean substitution. This is independent numerical corroboration,
not a rigorous quadrature enclosure or proof that other history models fail.
Constant older-label controls converge to .25 with errors7.94e-8 and1.99e-8.

The combined saved run evaluates10,656,512 deterministic two-step quadrature
forwards in1.569579CPU; initial Gaussian-only attempt added.889774CPU and165k
forwards. Every array chunk has at most65536 rows. All cost is diagnostic,
not an estimator efficiency claim. Conditional/stochastic history alternatives
remain open. No further precision refinement is justified for this failure.
