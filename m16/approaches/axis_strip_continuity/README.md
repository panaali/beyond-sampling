# T155 support-axis strip continuity with exact word feasibility

## Question
How much of T147's remaining bound is sensitivity transverse to the actual prefix support, or gate words impossible on a fixed neighborhood? Claim9719cd4 preceded implementation.

## Method
Freeze the two original T127 prefix neighborhoods x0 in[-3/8,-1/8] and[1/8,3/8], target u=1 and next input x in[-6,6]. The same H2T4 checkpoint has Wi0<0<Wi1: these prefixes put h1 on axis0 or axis1. For the active coordinate s, the exact interval is obtained by multiplying the prefix endpoints by its input weight.

Reuse all pinned T147 affine boundary forms. Restrict hidden motion to that axis, replacing ||C||1 by |C_axis|; deduplicate the reduced(x,y,s,u) form exactly. Report this all-word baseline before pruning. For each of16 first/middle gate words, clip the compact(s,x) rectangle by its four exact closed gate inequalities. Discard a word only when its closed feasible set is empty. Keep zero-area line and point sets, including strict-active boundary over-approximations. Recount the directional strip union using the feasible words. Compare the two stages separately; translate to prefix-coordinate sensitivity by multiplying by |Wi_axis|.

Use the actual need1 rank denominator2LU, and separately the elementary certified bound(81/2)*K*|Wi_axis|. Both fixed prefix intervals lie below u, so need1 is physically consistent. This regularizes the direct exact pair value versus x0; it does not certify the old stage2 interpolation into later mixed-support states. No new input probes or grid changes.

Freeze controls before weights: exact unit-square clipping to half-square, empty set, line, point, constant true/false inequalities, and duplicate planes. For Wi=(1,1), Wh=I, u=1 and axis state s in[1,2], exactly three first/middle words are feasible on each axis: middle always both active; first excludes the opposite-only support. Independent trained feasibility review must use a different method, such as enumerating all pairwise boundary intersections against every inequality, rather than trusting polygon area.

Competing explanations: (1) large sensitivity remains on the actual axis; directional baseline may stay large. (2) impossible gate words dominate; exact feasibility pruning may help. (3) pruning drops degenerate feasible states unsafely; line/point and independent intersection controls must detect this. No finite bound improvement proves a practical integral or interpolation recipe.

## Status
Protocol frozen; exact primitives next.

## Results
Pending.

## Compute accounting
Reuse saved boundary forms, one checkpoint load,32 trained word feasibility problems in two variables and exact finite coefficient sums. Zero new trained forward evaluations or random inputs. Count synthetic geometry controls and timing. One thread,2GB, review lock <=20CPU.

## Claims
[inferred] Directional strip bounds and conservative closed-word feasibility can sharpen the prior sufficient bound on these two fixed intervals. [speculative] The trained constants become practically useful.

## Alternatives not ruled out
Final-support/rank-facet restriction, exact Gaussian line densities, other prefix intervals, direct outer integration and local higher-order regularity. No bound outside the frozen intervals is implied. Transverse state variation and later mixed-support interpolation remain separate problems.

## Failure modes
Using polygon area to discard feasible lines/points; treating strict ReLU ownership as a closed infeasibility certificate; forgetting prefix-to-hidden scaling; silently applying local word exclusions globally; or confusing direct-pair continuity with the old interpolated estimator's error.
