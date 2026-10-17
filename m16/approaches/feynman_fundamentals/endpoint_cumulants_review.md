# Independent trained T3 cumulant algebra review

Read-only review of endpoint_cumulant_review/trained_t3.py confirms the
conditional moment measure. For triangle rays R and barycentric b,
dOmega=|detR|/||bR||^3 db. The Duffy Jacobian is1-u; Gaussian direction
normalization is1/(4*pi). The independent chi3 radial moment is
2^(k/2)*Gamma((3+k)/2)/Gamma(3/2). Homogeneous gate/rank conditions preserve
radius independence. Joint moments are normalized by each target-label mass.
This is an exact real-arithmetic identity evaluated by numerical quadrature,
not a quadrature or floating-topology enclosure.

The author's cumulant routine expands central moments and subtracts the
three covariance pairings at order4. Independent
check_endpoint_cumulants_review.py instead enumerates set partitions:
kappa(X1,...,Xk)=sum_pi (-1)^(|pi|-1)(|pi|-1)! product_B E[product_(i in B) Xi].
All tensor entries through order4, for three target laws before and after
atom separation, agree to maximum scaled difference2.97e-16 using60-digit
Decimal arithmetic on the saved raw moments. Work took .02345 CPU seconds.
This checks the conversion independently, not the numerical moment source.

The h=0 atom contributes zero to every positive-degree raw logit moment.
Removing it therefore leaves those joint tensors unchanged and changes only
the conditioning mass. The code divides by jointmass-atom and adds the atom
to accuracy only for target0, consistently with first-index argmax. It does
not claim to remove all possible positive-volume readout ties from nonzero
hidden states. Analytic atom mass and numerical joint mass differ at finite
quadrature precision; the accepted mass checks remain relevant.

The reported endpoint bias is thus not explained by an obvious centering or
atom-renormalization bug. Large raw values outside probability bounds remain
visible alongside separately labeled simplex projections. Finite cumulant
truncation, support information, and moment sensitivity remain distinct
limitations. Gate/rank-conditioned mixtures with exact one-dimensional
readout support are a plausible untested rescue; this review does not retire
the endpoint family or certify the trained numerical moments.
