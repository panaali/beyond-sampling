# T240 separated-rank correct-mass contract

This is a prospective exact-real contract, not an experiment or a statement that the easy mass has been measured. T189's reviewed affine/interval recurrence and all-rival first-index predicates are inherited; no claim that every enclosure was independently recomputed is added.

## 1. Event and geometry
For T>=3 iid standard Gaussian coordinates, let X_(1)>X_(2)>X_(3) denote descending order statistics outside their null tie set. Fix absolute rational g>=0 and

    E_g = { X_(1)-X_(2)>=g and X_(2)-X_(3)>=g }.

The initial proposed reporting thresholds are g in {0,1/4,1/2,1}, chosen before any new contraction. These are absolute input-unit gaps. For g>0 the labeled pieces are affine polyhedra, NOT homogeneous cones. The claim at g0 ignores input ties only under the continuous Gaussian law; network readout ties can have positive probability and retain first-index ownership.

For ordered distinct labels i,j, define E_ij(g) by x_i>=x_j+g and x_j>=x_r+g for every r not in{i,j}. Its true second label is j except on null input ties. The pieces partition E_g almost surely. No ordering among the remaining labels is imposed. A normalized-gap event such as gaps>=gamma*||x|| is scale-invariant but is a different event with a different mass law; it will not silently replace this definition. Using infinity norm could yield polyhedral subpieces after fixing its active signed coordinate, but that is a distinct implementation alternative.

Conditional on x_j=t, the remaining labels are independent. Therefore the exact probability is

    P(E_g)=T(T-1) integral phi(t) [1-Phi(t+g)] Phi(t-g)^(T-2) dt.

At g0, substituting q=Phi(t) gives T(T-1) integral_0^1 (1-q)q^(T-2)dq=1. This identity provides a separate denominator route, not a preexisting directed quadrature or free setup. It requires no Gaussian hidden-state assumption.

## 2. Conservative Cartesian-domain ownership
Let a closed box B have finite rational endpoints l_r<=u_r. Whole-box containment in one labeled gap piece follows from

    l_i-u_j>=g, and l_j-max_(r not i,j) u_r>=g.

With positive-width bins and continuous input, shared endpoints are null. This sufficient inner test is exact for containment in that particular fixed labeled polyhedron; a box spanning label changes is not certified by choosing its center. Do not infer that failure means B misses E_g.

For an outer cover, B intersects the closed E_ij(g) if and only if

    max(l_j, max_(r not i,j) l_r + g) <= min(u_j, u_i-g).

Necessity follows by bounding x_j. For sufficiency choose x_j in that interval, other coordinates at their lower endpoints, and x_i=max(l_i,x_j+g), which stays below u_i. Equality may describe only a boundary intersection, so keeping it in an outer cover is conservative. Any box passing at least one pair is counted ONCE, not once per feasible pair. Different inner pairs cannot own positive-volume overlap for g>=0 under strict input ranking. Reject malformed boxes rather than dropping them.

The closed containment and intersection predicates preserve g0 ties conservatively. In a disjoint half-open Cartesian partition of [-A,A]^T, an inner-box mass lower bound and an outer-box mass upper bound plus the full outside-cube tail enclose P(E_g). Product Gaussian bin intervals can reuse the reviewed rational CDF method. The outside tail is not assumed to fail E_g. Evaluating the displayed one-dimensional integral would be a separate source/resource-gated alternative, not needed for these conservative bounds.

## 3. Network correctness and mass
For a whole-box label j, use inherited exact affine plus interval hidden enclosures, with every residual dependence retained conservatively. For each rival o, certify correctness if the lower endpoint of logit_j-logit_o is >0 for o<j and >=0 for o>j. A negative certificate uses the converse upper predicates <=0 for an earlier rival or <0 for a later rival. Otherwise mark unknown. Never drop a candidate because the bound failed.

The first conservative route uses only boxes wholly inside E_ij(g), with certified class j. Their disjoint masses give C_g lower/upper enclosures for the algorithm's certified-correct event. Analogous wrong boxes give W_g. All other E_g inputs remain unresolved, including gap-boundary boxes and outside tails. The exchangeable 1/m rule from T189 is NOT transferred after conditioning on E_g: that event changes the within-bin conditional law. A future subdivision or exact conditional integral may recover these boxes, but requires its own proof and accounting.

Let [p_lo,p_hi] enclose P(E_g), and [c_lo,c_hi],[w_lo,w_hi] the disjoint certificate event masses. Then absolute correctness in E_g is at least c_lo; if p_lo>0 a conservative conditional accuracy bracket is [c_lo/p_hi,1-w_lo/p_hi]. The unresolved event mass is bracketed by max(0,p_lo-c_hi-w_hi) and min(1,p_hi-c_lo-w_lo), with interval arithmetic and consistency checks. Numerical inconsistency is a failure, not permission to clip a contradictory interval. These are certificate-event bounds, not actual error/MSE bounds. Report p, C, W, unresolved absolute mass and the conditional fraction separately; a large conditional fraction on a tiny E_g is not most of the full 95%.

## 4. Stronger certificates and monotone inheritance
A saved-only filter of T189 certificates is a baseline. Its TRUE certified-correct event mass cannot exceed the original whole-domain certified-event TRUE mass, because it is a subset. A reported original LOWER bound is not a numerical upper ceiling and cannot be used as one. This phase has not read the original exact upper endpoint; any numerical ceiling requires that endpoint and its saved provenance in the separately gated contraction. Filtering adds localization, not a larger certified event, and does not itself establish large correct-mass coverage or cheap cost. It can locate where that particular certificate succeeds without re-running the recurrence.

To improve, a later bounded candidate may restrict affine ranges to the exact gap/rank polytope B intersect E_ij(g), or subdivide a fixed box and inherit its parent certificate. On a fixed bounded rational polytope, exact affine extrema occur at vertices; exact vertex completeness or an independent valid LP dual bound is required. Evaluating only sampled vertices or one representative is not a certificate. ReLU remainders must be recomputed against valid restricted ranges, with all kinks retained through the inherited enclosure. This is a possible stronger primitive, not an implemented or cheap method.

Under refinement, retain previously valid parent certificates so certified coverage cannot be lost merely because a newly selected local affine approximation is looser. Charge every attempted child, failed bound, exact rational/LP construction and all setup. Adaptive priority rules must be frozen before outcomes; no reporting only cheap successes.

## 5. Prefix-level target limitation
T219 supplies exact target prefix histories/maps, not final correctness. A k<T prefix event contains arbitrary future inputs. Current leader/gaps or a stable prefix gate word do not determine the final second-largest input or final network output. Prefix diagnostics may report certified hidden/gate enclosures, prefix-gap event mass and costs, explicitly labeled non-accuracy information. They may not be added to C_g for the full target.

A genuine target correctness certificate must bound both future rank ownership and the remaining network computation on a specified full-input event, then integrate that event. Unbounded future coordinates are not silently set to zero or to their conditional mean. A suffix tail bound or further finite partition is additional charged work. No Cartesian10-dimensional exhaustive grid is proposed by this phase.

## 6. Scope of any outcome
The owner hypothesis concerns a cheap large correct set, not a particular box representation. Low C_g may mean coarse domain boundaries, gate-remainder dependency loss, genuinely wrong inputs, or insufficient compute. High C_g demonstrates only the specific verified event and cost convention. Neither outcome alone establishes random-like errors, an architecture capacity ceiling or impossibility of aggregation. Complete cost must distinguish probability setup, model binding, certificate arithmetic, classification, data handling and review, with integer bit lengths and wall/process CPU. Existing selected-expression ledgers are partial; do not equate one arbitrary-precision operation with one hardware FLOP.
