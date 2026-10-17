# T82 actual network-switch partitions on fixed face slices

## Question
Does splitting at actual network branch topology, rather than only model-free window switches, remove a specific source of face-integrand quadrature error? Claim TASKS-only8fdc3c1 was published before implementation via the authenticated GitHub connector after CLI authentication failed. The prepared local82247c8 is equivalent and must not be pushed independently.

## Method
Fix the face configuration, tie value and all but one nontied input. Let that input be z and the tie displacement be t. The RNN input is affine in(z,t). Clip the bounded valid face window to a convex polygon; propagate exact-form affine hidden coefficients and split polygons at ReLU gates. Split t sign and clip each target's winning readout inequalities with first-index ownership for identically tied logits. The correctness polygons have disjoint interiors. Their projected z-vertices contain every change of vertical-section endpoints. Within each slab the Gaussian t-mass is a finite sum of Phi(affine z) differences, hence smooth. Convert z vertices to the original truncated-normal quantile q before comparing quadrature rules.

T55 independently checked this exact-arithmetic topology argument: include all window/sign/exclusion boundaries, avoid duplicating identically-zero gate branches, and distinguish floating completeness from the symbolic theorem. This is two-dimensional slice enumeration, not a full input-space accuracy estimator or a scalability claim.

## Frozen protocol and alternatives
First primitive controls: clipping square/triangle and narrow positive strips, exact polygon area partition, affine recurrence agreement with direct evaluation at cell centroids, all-zero and duplicate readout first-index ties, analytic vertical Gaussian mass against the separately implemented corrected Python segment walker. No discarded thin cells by a user-tuned tolerance; report floating degeneracy and use a finite cell cap that raises explicitly.

Freeze eight trained slices: H2T4 and H4T4 public seed0; R23/R12; first and last configuration, all remaining quantile anchors .5; vary face coordinate q2 on[.05,.95]. Construct all topology before looking at reference errors. For each line compare composite GL4 on network vertex panels, GL4 on the same number of equal-width quantile panels, and GL4 on model-free geometry-vertex panels. All network-aware discovery cost is charged, so same-node comparisons isolate accuracy only. Use network panels GL16 and GL32 as a consistency reference, with independent segment-walker values at17 fixed q points. Retain numerical disagreement rather than calling GL32 exact. No new MC, noise/model selection, wider H/T or grid sweep in this cycle.

Alternative1: network switching causes the observed irregularity; actual projected topology improves quadrature over equal-node uniform panels. Alternative2: smooth but steep inverse-CDF variation or cancellation dominates, so topology discovery gives little gain despite more construction. Alternative3: common clipping/walker defects or missing thin cells mimic improvement; independent pointwise, tie, area and direct-state controls must pass first. Explicit cell cap5000, <=110CPU cycle, <=2GB, one thread/shared lock. Primitive tests <=20CPU may use review lock.

## Compute accounting
Count every polygon clip, tested vertex, intersection, retained/split cell, affine coefficient operation, readout inequality, CDF/quantile evaluation, panel and quadrature node. Save cold CPU and source/checkpoint hashes. Python allocation/sorting and numerical-library internals remain named overhead; operation proxies are not a complete portable certificate. Independent walker calls and GL16/32 reference work are diagnostic, separately charged. A failed cap is an implementation-budget limit, not proof of mathematical failure.

## Status
Protocol frozen; implementation pending.

## Results
Pending; all failures and changes will be retained.

## Claims
- [inferred, independently reviewed] In exact arithmetic projected polygon vertices suffice for the specified bounded two-dimensional slice topology.
- [speculative] These topology knots may improve deterministic integration on the frozen slices.
- No full-model accuracy estimate, compute advantage or solved claim.

## Failure modes and untested alternatives
Thin polygons, nearparallel clipping, duplicate/identical readout ownership, floating vertex ordering, combinatorial slice growth, inherited face-table error and endpoint cutoffs. Results cover only the truncated quantile interval, not the omitted endpoints or every anchor. Sparse mixed-dimensional integration, local selective switching and larger models remain untested; no broad sparse-quadrature negative follows.

## First bounded result
All primitive controls pass, including an approximately1e-12 positive strip, zero/duplicate readout first-index ownership, affine states at cell centroids and gate-area partitions. On eight trained slices, the independently implemented corrected segment walker agrees with polygon vertical mass to6.67e-16. The GL16/GL32 network-panel reference disagreement is at most3.73e-10; this is numerical consistency, not a certified error bound.

Network-aware GL4 beats the same-node equal-width quantile control in4/8 slices. The H2 R12 errors are approximately1.23e-11/2.08e-11 versus uniform8.51e-5/2.85e-5. H4 R12 first configuration similarly improves to6.77e-12 versus2.23e-5; its last configuration is already smooth and uniform does better. R23 outcomes are mostly negative: H2 network errors8.85e-5/1.76e-4 exceed uniform1.02e-5/4.50e-6. H4 R23 first configuration is worse(4.82e-6 versus6.90e-7), last is better(1.11e-7 versus1.05e-6). Thus exact topology resolves some kinks but does not remove all quadrature difficulty or automatically give efficient allocation.

`results.json` retains all knots, counts, work, cold construction and integration times, pointwise reference cost and signed errors. Reproduce with `/workspace/clean-neurons/.venv/bin/python -m m16.approaches.network_switch_slices.check_slices` under the shared lock/thread/memory wrapper. No random model inputs or accuracy truths enter the construction.

## Frozen second cycle: natural Gaussian coordinate
The observed R23 negative is consistent with the already-listed smooth inverse-CDF explanation. Test one coordinate change on all eight cases with unchanged network polygons/panels and4nodes per panel: integrate in z with density phi(z)/(1−Phi(tie)) for R23 or phi(z)/Phi(tie) for R12. This is exactly dq/dz for the varying truncated-normal input; it does not reweight by a fitted model statistic. Use the same network-panel endpoints transformed from q, and retain original q GL4 and equal-count q controls. Compare z GL16 and z GL32 with the saved q GL32 reference. Charge all geometry reconstruction and every transformed node, Gaussian exponential, CDF and normalization. No panel/order/model selection or new network partitions.

Alternative1: inverse-CDF curvature caused the R23 error, so natural-coordinate integration helps even when topology alone did not. Alternative2: steep Gaussian-CDF boundaries or small panels still dominate; transformed integration remains mixed. This tests a coordinate formulation, not blind grid escalation. Same <=110CPU/shared lock/2GB limits. All eight outcomes retained.

## Natural-coordinate result: R23 repair, R12 tradeoff
All four R23 GL4 errors improve with the same network panels and node counts: H2 first/last become+1.0495e-7/+2.3590e-7 (from−8.8489e-5/−1.7642e-4); H4 first/last become+4.5119e-8/−6.3744e-10 (from−4.8188e-6/−1.1059e-7). This supports the smooth inverse-CDF mechanism for those negatives. All four R12 cases worsen relative to their near-exact q-rule values, to5.6080e-10/8.3364e-8/9.2053e-8/9.2055e-8. No family-based or model-specific rule selection is made.

The natural-coordinate GL16/GL32 disagreement is<=2.23e-16 and normalized Gaussian weights reproduce the .9-wide quantile interval to floating accuracy. The normalizer is the original upper/lower truncation probability; there is no extra division by.9. Independent T55 read-only review checks this Jacobian and the window/target/first-index polygon construction. `gaussian_results.json` saves all rules, construction cost and Gaussian/CDF work. Reproduce with `python -m m16.approaches.network_switch_slices.check_gaussian` under the same wrapper. The smaller maximum error is a bounded slice diagnostic, not full accuracy or a compute win.

## Frozen independent attribution control
The improvement could come from the natural coordinate alone. T55 is requested to use the same eight slices and four nodes per panel with (a) the same number of equal-width z panels as network panels and (b) geometry-only panels in z. Compare to the frozen topology-z rule and retain every case. This changes neither model, anchor, degree nor discovered topology. It is an independent bounded attribution check, not another refinement sweep; reviewer code/results remain in T55's folder. Network construction cost still counts even when the accuracy control uses equal nodes. Irrelevant topology vertices, thin floating panels and endpoint exclusion remain limitations.

## Independent attribution result and frozen endpoint-completeness phase
T55's independent coordinate-rule assembly on the same polygon masses finds network knots beat same-count equal-width z panels in6/8 cases, not all. Both H2R23 slices prefer equal-width z (errors−7.13e-9/−4.34e-10), so coordinate choice alone accounts for those gains. Geometry-only z misses substantive topology on H2R12 first configuration(error−.002885) and H4R23 last(+.001282). Several other slices do not need full network construction. GL4 Gaussian-weight defects up to9.37e-6 are retained without renormalization. This review independently checks rule assembly/attribution, not a second polygon engine; final review hash will be added on handoff.

The .05..95 interval is not endpoint-complete enough for an accuracy estimator. Before any full face contraction, freeze the same eight slices extended to q in[delta,1−delta], delta=1e-6. The omitted nonnegative face mass is at most2delta per slice because segment error mass lies in[0,1]; across both complete face families the corresponding accuracy uncertainty is at most2delta*(Z23+Z12). This does not bound later outer-coordinate quadrature error.

Use natural-coordinate GL4 as the retained cheap control, GL16 as the predeclared candidate, GL32 as consistency reference; retain every model/slice. Check full Gaussian probability weights against1−2delta, partition into old interior and two tails, and compare the independent segment walker at17 fixed log-tail/interior q values. Require GL16/32 differences below1e-8 on these controls before a frozen H2/H4T4 full contraction at outer orders3/5 is attempted. Failure pauses that dependent step; do not silently increase degree or cut endpoints more aggressively. Alternatives: endpoint curvature may remain tame in z, or large slabs/nearzero-width geometry may invalidate the cheap rule. This is domain completeness with an explicit omitted-mass bound, not a new model/grid sweep. <=110CPU/shared lock/one thread/2GB; no new MC.

## Endpoint result and frozen full-face contraction
All eight endpoint-complete controls pass: independent tail/interior segment values agree within5.56e-16, GL16/32 within1.67e-16, and lower/interior/upper partition sums within1.12e-16. Gaussian weight normalization passes without renormalization. This validates these finite controls and allows the predeclared dependent full-face experiment; it is not an all-anchor error certificate. `endpoint_results.json` retains every check and explicit omitted-mass bound.

Freeze the full H2T4/H4T4 seed0 face contraction at remaining-coordinate Gauss orders3 and5, with every discrete configuration included. For each configuration and pair of outer nodes (tie-value quantile and the other nontied input), build the same two-dimensional partition for q2 in[1e-6,1−1e-6], and integrate its natural-coordinate panels with GL16. Use GL32 on the same constructed polygons as a separately charged inner-rule diagnostic, not a per-slice adaptive replacement. Sum both exact face-law factors. Compare all four outputs against prior complete tensor-grid values and the independent T4 accuracy references; separate inner GL16/32 discrepancy, explicit q2 endpoint omission and unresolved outer quadrature error.

Alternative1: one accurately integrated face coordinate materially reduces full-grid error at manageable construction cost. Alternative2: the remaining coordinates dominate or full slice discovery is too expensive, despite excellent one-line controls. Do not infer scalable accuracy from smooth slices. Record every actual polygon/clip/state/readout/CDF operation proxy, all slices, cold CPU and failures; no cached values or model truths enter construction. Original order3 is the fixed interface recipe, order5 a retained diagnostic, no model-dependent winner. Same <=110CPU/2GB/one-thread/shared-lock limits; no new models, seeds, random inputs or further order sweep automatically follows.
