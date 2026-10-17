# Scope of the inherited seed-overlap defect

The corrected experiment uses distinct per-kind namespaces. This note explains
why the original shared seeds invalidate independent-replicate uncertainty,
without claiming that this defect explains the large collapse in point gains.

Assume independent identically distributed scrambles U_s across distinct seeds,
and write one replicate as E_r=A(U_r)+B(U_(r+7)), including normalizer factors
and signs in A,B. For R consecutive rule seeds, let
v=Var(A)+Var(B), c=Cov(A(U),B(U)). Within each replicate A and B are independent,
so Var(E_r)=v. Between replicates only a separation of7 shares a scramble,
giving Cov(E_r,E_(r+7))=c, with R-7 such pairs when R>7.

The usual unbiased sample variance about the sample mean therefore obeys

    E[s²] = v - 2(R-7)c / [R(R-1)],
    Var(mean(E)) = v/R + 2(R-7)c/R².

By Cauchy-Schwarz and arithmetic-geometric mean, |c|<=v/2. Consequently the
relative bias in expected sample variance is bounded by
(R-7)/[R(R-1)]:3.75% atR16 and25/992≈2.5202% atR32. The variance-of-mean ratio
to the independent-replicate expression v/R can differ by as much as
(R-7)/R:56.25% atR16 and78.125% atR32. The sign is not predetermined.

These are bounds on population moments under the iid-scramble idealization,
not confidence bounds on one realized reciprocal-variance gain. A reciprocal
is nonlinear, so the sample-variance bound is not a strict gain-ratio bound.
The overlap alone cannot create an orders-of-magnitude change in the expected
sample variance, but it changes the dependence assumed by the ordinary iid
bootstrap and bias tests. Real finite seeds and numerical Sobol generation
also require their usual pseudorandomness assumptions.

The same +7 schedule occurs in historical ht_seeds/est.py. This note records
the issue for DRI review; no historical or shared source is silently rewritten.

T55 independently checked both identities and the Cauchy-Schwarz bound. The
review explicitly retains the fixed-function/iid-stream assumptions and the
difference between expected variance and realized reciprocal-gain behavior.
