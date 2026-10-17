# Independent T82 stencil and geometry review

The R23 window is min(sqrt2(o1-s)/3,sqrt2(s-o2),8.5). Its variable-upper-input
branches meet at o1=4s-3o2, and the first branch reaches the cap at
s+25.5/sqrt2. Mapping through the upper conditional normal CDF gives the
stated knots. For R12 the largest lower input switches identity at the largest
other lower quantile, and sqrt2(s-o2)/3=8.5 gives
v=Phi(s-25.5/sqrt2)/Phi(s). Some candidates can be inactive because another
branch dominates; extra cuts are safe but are not themselves evidence of an
actual kink. The author's active-switch diagnostic distinguishes this.

Strict selection of left AND right stencil endpoints in[.05,.95] ensures the
middle point also lies there; midpoint-only selection does not. Independent
script check_surplus_review.py reads the first frozen NPZ(T5,.0025,R23,axis2)
and computes all levels4..8 without importing any author summary code.
Largest RMS/L1 discrepancy is5.422e-20. Independently fitted RMS slopes:

| selection | independent | author |
|---|---:|---:|
|strict whole stencil|1.0792201812339077|1.0792201812339086|
|geometry panels|1.0889792272260600|1.0889792272260608|
|same-count uniform panels|1.1183889026744416|1.1183889026744418|

The same-panel-count control is essential. For f(x)=x², midpoint surplus on
widthH is−H²/4. Panels of relative lengths p_j therefore improve width-weighted
RMS by sqrt(sum_j p_j^5) at a fixed per-panel level even without aligning any
feature. Equal K panels give improvement K^-2 merely through extra resolution;
the decay exponent stays2. The saved panel widths obey this identity to
1.108e-12 in the independent raw-data calculation. A derivative hinge normally
has RMS exponent3/2 but L1 exponent2 until exact splitting removes the kink;
these metrics should not be conflated.

Cost.022998CPU, zero new face/model evaluations. The face oracle is shared
corrected walker code, and the raw-array review is independent statistics,
not an independent neural integration method. Existing backend agreement
supports implementation consistency but cannot exclude a common formula bug.

The result verifies that neither complete-stencil filtering nor these known
geometry knots repairs the first tested negative; the owner's four-case
summary broadens that finite observation. It does not prove asymptotic
regularity, rule out actual network-boundary splitting or certify sparse-grid
failure. Finite slopes, rare-line concentration, other axes/models, mixed
regularity, adaptive weights and different parameterizations remain relevant
alternatives. No accuracy truth or fitted knot was used in this review.
