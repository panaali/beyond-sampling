# Positive-diagonal reparameterization invariance

This is a proof-only corollary to PROOF.md, with identical finite-domain,
fresh-input and no-numerical-certificate qualifications. It changes the model's
parameterization while preserving its function; it is NOT a fixed-weight
coordinate automorphism of the kind excluded by T229.

Let D be a positive diagonal matrix and reparameterize

    h'=D h, Wi'=D Wi, Wh'=D Wh D^-1, Wo'=Wo D^-1.

Positive homogeneity of coordinatewise ReLU gives h'_t=D h_t at every step
with identical inputs and zero initial states. At selected coordinates i,j,
a'=D_i a and b'=D_j b. Both slopes and intercepts scale by the same positive
factors, so u'=u, v'=v and the anchor's gate is unchanged. Thus r'=D_j r,
and the conditional disagreement probability is invariant.

For the margin envelope of PROOF.md section4, define

    K=|Wo_y−Wo_k| |Wh|^s e_j.

Because D is positive diagonal, absolute values commute with its multiplication:

    |Wh'|^s=D |Wh|^s D^-1,
    |Wo'_y−Wo'_k|=|Wo_y−Wo_k| D^-1,
    K'=K/D_j.

Consequently K' r'=K r, K' E[r'|h']=K E[r|h], and the bound
K |beta| (u−v)^2/(2 sqrt(2pi)) is unchanged. The raw-state-moment expression
also remains unchanged after H'=D H: ell'=ell D^-1, so
ell' E[H'H'^T] ell'^T=ell E[HH^T] ell^T (row-vector notation here).

This makes the specified disagreement/margin diagnostic invariant under these
positive diagonal gauges, unlike an unnormalized augmented-row cosine.
It does not prove that the diagnostic is useful, finite-cost, small, numerically
stable, or informative about classification without margin information.
