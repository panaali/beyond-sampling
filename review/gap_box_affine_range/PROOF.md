# Exact affine range on a labeled separated-gap box

## 1. Domain, coordinates and target of the proof
Let T>=3, distinct labels i,j, finite exact physical-coordinate endpoints l_k<=u_k, and physical gap g>=0. The closed box is B=product[l_k,u_k]. Define

D_ij(B,g)={x in B: x_i-x_j>=g and x_j-x_r>=g for all r not i,j}.

We seek the exact minimum and maximum of f(x)=c+sum a_k x_k on this closed polytope. This is a range problem, not Gaussian integration or a correctness certificate. Reversed endpoints, invalid labels, nonfinite/nonexact inputs and negative g are invalid inputs. Equal endpoints and lower-dimensional feasible domains are allowed and retained, not discarded. A nonempty closed domain may have Gaussian probability zero.

All symbols in this proof are in physical x units. In inherited T189 data v=2x. Two mathematically identical adapters are possible but MUST be explicit: (A) transform l_x=l_v/2, u_x=u_v/2 and a_x=2*a_v for objective c+a_v dot v, keeping g physical; or (B) retain v endpoints and objective but replace gap by2g. Constants and additive remainder intervals do not scale. Neither adapter is automatic in the proposed primitive; its core interface takes physical x. A future caller must prove that its affine denominator/coefficient representation is converted consistently, not merely halve a displayed gap. Probability CDF arguments remain physical x. T240's original unit failure is a regression premise, not erased.

## 2. Exact one-coordinate feasibility
Set t=x_j. Then the coordinate intervals are

x_i in [max(l_i,t+g),u_i],
x_r in [l_r,min(u_r,t-g)] for r not i,j,
x_j=t.

They are simultaneously nonempty exactly when

L=max(l_j, max_{r not i,j}(l_r+g)) <= t <= U=min(u_j,u_i-g).

If L>U, D_ij is empty. If L<=U, every t in [L,U] yields a nonempty independent Cartesian product for the other coordinates; selecting any permitted endpoints constructs a feasible point. In particular L=U is retained as a feasible section, although positive-volume claims do not follow. The top-label lower endpoint l_i and remaining upper endpoints u_r cause clipping changes, not additional feasibility constraints, because the original intervals are themselves valid.

## 3. Conditional extrema and finite candidate set
For each t, define p_k(t),q_k(t) as the lower/upper endpoints above. Set x_j=t. The conditional maximum chooses x_k=q_k(t) when a_k>=0 and p_k(t) when a_k<0. The conditional minimum uses the opposite endpoints. At a_k=0 either endpoint gives the same objective; fix the lower endpoint for both for deterministic witnesses.

Call these attained objectives F_max(t) and F_min(t). Each is continuous and piecewise affine. The only potential clipping breakpoints are l_i-g and u_r+g for r not i,j. Coefficient signs may make some irrelevant; retaining all of them is conservative and avoids case-dependent omission. Define K to contain L,U and every such breakpoint lying in [L,U], deduplicated exactly. If L=U, K has one element.

On each closed subinterval between consecutive elements of K, both objectives are affine, so each maximum and minimum on that subinterval occurs at an endpoint (or everywhere if constant). Therefore

max_{D_ij} f = max_{t in K} F_max(t),
min_{D_ij} f = min_{t in K} F_min(t).

A maximizing/minimizing witness is the corresponding t and sign-selected endpoint completion. Directly checking that witness satisfies all box/gap constraints and the reported exact objective provides an attainability check. This is an exact finite reduction, not endpoint-only checking of an unresolved ReLU function: it applies solely to a SINGLE affine objective on one fixed labeled domain, and includes every clipping kink.

There are at most T+1 distinct candidates (two endpoints, one leader breakpoint, T-2 remaining-coordinate breakpoints). Each conditional endpoint completion and objective evaluation uses O(T) exact scalar expressions, so both extrema cost O(T^2) expressions per ordered pair, plus input validation/deduplication and bit-complexity. One may scan without sorting; a sorting implementation must charge it. Rational numerator/denominator growth, conversion, comparisons, storage and runtime are not a complete FLOP count merely because this expression bound is polynomial. Testing all T(T-1) label pairs introduces that separate factor; it is not free.

## 4. Rank and boundary ownership
For g>0, i and j are uniquely the largest and second-largest coordinates on any feasible point; the remaining coordinates need not be ordered. At g=0, the closed inequalities admit ties. The geometric range includes them conservatively, but j need not be the first-index second-order label under a deterministic tie rule on those boundary points. Continuous iid inputs give null probability to coordinate ties; a positive-mass probability argument may use this only after its distribution is stated. A pointwise certificate with g=0 must separately require strict rank or explicitly refine tie ownership. Do not infer rank identity from the closed LP result alone.

Likewise a final network margin equality follows first-index argmax ownership, not strict positivity for every rival. That later predicate is not changed by this primitive. Zero-volume cells must not be assigned positive Gaussian box mass because a feasible witness exists.

## 5. Safe use in an inherited affine/ReLU enclosure
Suppose an inherited real quantity is enclosed as z(x)=c+a dot x+e(x), with e(x) in [eL,eU] for every x in a parent box. Restricting to D_ij subset B preserves that remainder premise. The exact affine range above yields a valid enclosure [min_D f+eL,max_D f+eU]. It may be intersected with another independently valid range on D. Empty intersection indicates a premise/implementation failure; do not floor or silently clip it.

This improves only the affine-input part for fixed c,a,eL,eU. Downstream ReLU propagation can use that valid interval with a proven remainder construction. However, choosing a different local affine slope after restriction can change later errors; it does not prove the whole final certificate monotonically tightens. Preserve the old certificate as an independently valid fallback on the subset, and count all work. A proof of exact affine range is not a proof of exact ReLU-network extrema.

Partial-box probability remains T252's separate problem. A domain certificate cannot be multiplied by the full unconditioned box mass. Under identical-bin exchangeability, label1/m can survive conditioning on the symmetric gap event, but the required mass is P(B intersect E_g), not P(B). No actual mass, trained result, cost advantage, target-prefix accuracy or solved-rung conclusion is part of this proof.

## 6. Preconditions for any implementation or experiment
This written phase contains no executable primitive or numerical call. A later source/control scope must freeze exact fixtures, output witnesses, physical-unit adapters, empty/degenerate policy, operation categories, durable failure handling and limits before execution. Independent references should enumerate vertices or solve rational constraints in a distinct way on small bounded controls, not repeat the same t-breakpoint algorithm alone. Source and saved reviews precede any trained application. No automatic target grid or domain enumeration follows.
