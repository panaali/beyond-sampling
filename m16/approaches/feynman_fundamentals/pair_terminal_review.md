# T117 fixed-state two-input terminal primitive

Source review of `gaussian_pair_terminal/terminal.py` e2b38dd. From fixed
absolute h, the first fresh input x has affine preactivation Wh*h+Wi*x.
Its exact root intervals determine the first mask. On each interval the
second preactivation is affine in(x,y), with coefficients
Wh*(Wi*mask), Wi and Wh*((Wh*h)*mask). The source uses these offsets;
it does not silently replace the problem by homogeneous zero-state
geometry. Enumerating second masks and readout inequalities yields the
correct affine planar regions. Active gates are strict, inactive weak,
and exact-zero forms are OFF. Earlier-index readout competitors are
strict, including identically zero margins.

Vertical lower/upper envelopes are split at all line intersections and
vertical bounds. Rank cuts add x=u and the intersections of y=u with
each envelope. Integration of phi(x)[Phi(upper)-Phi(lower)] is correct
on each resulting interval. Exchanging axes and transposing the final
rank matrix preserves the original meaning of the two input positions.

Let low=Phi(u)-Phi(-limit), high=Phi(limit)-Phi(u). Raw P00 divided by
low² is the need0 probability. The average of raw P01 and P10 divided
by low*high is the need1 probability: conditional on exactly one of two
remaining ordinary inputs exceeding the target, the two positions are
equally likely. This assumes the target was already inserted; the
production label contract j<=T-3 enforces that chronology. Crossed
fixed h,u fixtures need not be jointly reachable under an earlier prefix,
so they validate the extension operator, not its eventual weighted error.

## Retained domain failure

An independent probe accepted limit40,u39 with Wi=(1,-1), Wh=0,
Wo=0, h=0,label0. The primitive returned [1,NaN], with invalid-division
warnings at raw/denominators and error normalization. The Gaussian high
side had underflowed to zero; the unit-interval comparisons did not
reject NaN. This was observed on e2b38dd, outside the frozen limit6
panel, and reported immediately. An explicit finite-positive side-mass/
denominator check and finite-result check is the local remedy; stable
extreme-tail arithmetic remains a separate potential extension.

This API-domain failure does not refute the fixed-limit method or its
trained panel. Adaptive quadrature estimates and swapped-axis agreement
are useful controls but do not prove all numerical error bounded. Actual
setup/query cost and full-recursion integration remain unverified here.

The owner preserved this failure and added guards in4b8ef51. Source
inspection confirms that nonpositive/nonfinite normalization products
now raise before division, nonfinite affine maps/readouts and quadrature
values/errors raise, and final normalized values/errors are checked.
This repairs the observed domain failure without claiming stable
extreme-tail integration. The fixed36-control suite was retained by
the owner; no duplicate full suite was run by this reviewer.
