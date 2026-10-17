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

## Rank support and full-cycle topology review

The interpolation comparison uses low/high first-input transition
operators followed by below/above terminal probabilities. Therefore
need0=low[:,0] and need1=(low[:,1]+high[:,0])/2 are correct. Matching
u to a chosen prefix input alone does not make both need flags
reachable: the other observed prefix input fixes whether the one
allowed exceeding input has already occurred. The owner retained the
full extension table and added the four untied, rank-consistent probes;
one still has local discrepancy about-.01637. Tied input fixtures are
explicit diagnostics, not positive-probability population events.

For a fixed polygon, moving cuts x=u,y=u change clipping topology only
at vertex x/y coordinates or when their intersection(u,u) crosses a
polygon edge. A diagonal-coincident edge needs its endpoint events,
not an extra isolated crossing. The new complete union includes these,
earlier target/interpolation switches and the common penultimate cache.
This establishes real-arithmetic geometric branch completeness, not
rounding or adaptive-quadrature certification.

The few-ulp fallback uses separate endpoint extrema of the lower/upper
affine bounds, not endpoint values of their CDF difference alone.
Monotonicity then bounds the gap, while Gaussian density extrema occur
at endpoints or zero. Its floating padding is explicitly heuristic.
The wider-panel rescue maps x=a+(b-a)*t to[0,1], divides absolute
tolerance by width, and multiplies the returned integral/error by width;
these factors are correct. Both original QUADPACK failures remain
visible and are not method-family negatives.

Full h3 recursion source keeps the two-input terminal at stage2, then
propagates steps1 and0. The remaining ordinary-input count
T-t-int(label>=t) and target-insertion chronology are correct. A source
branch initially routed every nonpenultimate label to the new cache
although only labels0/1 were constructed. This is harmless on the
frozen model where label3 is excluded, but was reported for a local
`paired and label<2` guard before general interpretation. The full
comparison remains bounded, with no h9 or accuracy claim implied.

## Completed fixed h3 comparison

`check_pair_full_review.py` independently contracts all four raw rows
with60-digit Decimal arithmetic. At GL4 the new estimate is
.30171233058385778 versus baseline .25658923510284402 and reference
.30773074302392744. Absolute-error improvement is8.4975080, component
L2 improvement6.7295328, and the descriptive error-squared-times-CPU
improvement15.7013446. The penultimate-label contribution is exactly
unchanged between arms, as required by the shared cache.

GL2 toGL4 changes remain7.66825534042e-6 and7.20229103228e-6, both
failing the frozen stability threshold. This is a scoped representation
gain on one small-grid model, not convergence, a sampling comparison,
or a certified portable-cost claim. Label1 error worsens locally while
label0 improves strongly; unresolved prefix interpolation and outer
integration remain open. No larger-grid or noisy sweep is implied.

## T127 middle-target chronology and independent nontrivial control

The new affine construction correctly propagates h -> x -> fixed u -> y.
First-step gate roots partition x; middle target insertion creates further
x-only roots; final gates/readouts are affine in(x,y) and use the reviewed
polygon integrator. Stable zero-OFF and earlier-index readout ties remain
as before. Conditional need0=P00/L^2 and need1=(P01+P10)/(2LU) follow from
two exchangeable fresh truncated-Gaussian inputs. For actual preceding x0,
need0 iff x0>u, otherwise need1, excluding separately annotated ties.

Independent check_middle_target_review.py uses Wh=I,Wi=(1,-1), with default
penultimate label2 winning iff h1>h2. After x and fixed u, the two hidden
coordinates are a=max(max(x,0)+u,0),b=max(max(-x,0)-u,0). Final label2 wins
iff y>(b-a)/2. A separately implemented scalar Gaussian integral splits
these explicit affine boundaries and rank thresholds; at u=-1,0,1 it
matches the polygon primitive within1.12e-16, .0074CPU. This tests a
nontrivial default target, not a dominated zero-probability class.

The old comparator transports actual stage1 states through StagePieces
into the stage2 grid, then applies the penultimate cache. Its low[:,0]
and(low[:,1]+high[:,0])/2 use the same rank mixture. Source/full-image
support checks are valid because each ReLU coordinate achieves its maximum
on an interval endpoint. These are fixed-state primitive/comparator checks;
trained probe results and full outer behavior remain separate.

The first T127 trained panel has six untied rank-consistent queries.
Only x0=0,u=1,need1 shows the large-.1356406015 old-minus-pair difference;
the other five agree at floating precision. This fixed-state result at
an exactzero prefix alone cannot establish a population-weighted error.
Other table branches include impossible rank requirements or input ties.
A separately frozen x0=+/-1/4,u=1 continuation is pending to test whether
the discrepancy persists away from zero; no full outer run is implied.
