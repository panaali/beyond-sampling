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
