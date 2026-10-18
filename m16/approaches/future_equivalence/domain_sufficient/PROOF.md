# Sufficient equality on two represented positive hulls

Let R be a finite nonempty list of exact rational vectors and C=pos(R), including zero. A linear form b satisfies b(x)>=0 throughout C if b(r)>=0 on every generator; the nonpositive analogue also holds. Thus ReLU(bx) may be replaced by bx on C in the first case, or zero in the second. If all generator dots are zero, use zero. Mixed generator signs retain the original hinge. No claim about completeness of an H/V representation is needed for this implication on the represented positive hull itself.

Apply this independently to the difference between two vector-valued homogeneous ReLU margin functions. Retain the difference of their explicit linear parts. The reviewed rational canonicalizer preserves the resulting global function, using ReLU(-u)=ReLU(u)-u. If the simplified difference has no nonzero hinge and zero linear part, it vanishes on C. Repeat on both supplied hulls; two certificates establish equality on their union without constructing or claiming a convex union.

This is sufficient, not necessary. In particular a nonzero ambient linear form can vanish on a lower-dimensional hull yet this implementation returns unknown. A mixed-sign hinge may still cancel only on the domain in ways not captured by this procedure. Neither failed sign tests nor nonzero canonical keys imply functional inequality on the hull. Unknown is preserved. Empty/ragged/nonfinite input is rejected, not silently interpreted as a certificate.

Equality of all output margins relative to output0 preserves pairwise scores and first-index ties. This is stronger than label equality; a noncertificate does not refute label equality, projective equality, or other merge methods. Stored binary rays become exact dyadic generators for this statement. It does not establish those rays cover a real neural-history region under floating clipping, nor that a merged algorithm saves cost or mass computations.

The canonicalizer was independently reviewed in earlier T188 controls. New source gate and saved independent controls remain required. No trained application or necessity claim is authorized by this proof alone.
