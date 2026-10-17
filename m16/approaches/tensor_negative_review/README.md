# Independent review of the TT-cross negative (T81)

## Question
Does the reported T32 failure come from incorrect tensor algebra, coarse face quadrature, or the particular cross interpolation and variable ordering? Parent pushed claim f61f3bc before implementation. This review owns only this folder and does not modify the author estimator or shared walker.

## Method
Freeze before experiments: independently implement weighted TT-SVD, full tensor reconstruction and direct product-weight contraction. The author's TensorCross is the object under review, never described as our independent implementation. Check rank-two product sums, constants, zero tensors and a localized rank-one spike on fully enumerable small arrays. Compare every reconstructed entry and the integral with independent direct enumeration; report rank deficiency rather than silently treating a failed pivot as a method impossibility.

For the neural diagnostic, freeze seed0 H2T4 and H4T4, Gauss orders3 and5, both complete R23/R12 configuration sets, and ranks2/4. Each order5 pair has2,250 exact face-grid entries, making full-grid reference feasible. Build the grid independently from the pinned corrected sci_tao_v2 Python walker and its shared face-geometry functions. These are common oracle dependencies, not independently implemented network integration. Record hashes, checkpoint hashes, all face costs and failures. Existing independent T4 accuracy references are .30773074302315223 and .47701312599404416; they remain qualified floating references, not exact truth.

Separate full-grid quadrature error from the cross error on that same grid. Compare weighted TT-SVD as an oracle approximation using every entry; never present its full-data result as a cheap estimator. Freeze one rescue: move configuration from first to last, retaining the quantile order, ranks, two sweeps and all other author parameters. No tuning from held-out entries, extra models, ranks or orders after looking at results. A full residual-enrichment estimator is an untested alternative, not implied by the oracle SVD diagnostic.

## Compute accounting
Each cycle is at most110CPU seconds under the primary shared lock, one numerical thread and2GB RSS. Primitive checks below20CPU may use the review lock. Full-grid oracle calls are diagnostic cost; cross construction reports every unique queried grid entry with its saved walker cost plus the author's declared algebra proxy. Oracle tensor SVD/QR and direct array contractions are named numerical kernels, not a portable scalar-operation certificate. Report CPU, entry counts, ranks and kernel sizes. No new Monte Carlo runs.

## Claims
- [inferred] The prior eight fixed settings cannot establish impossibility of all tensor representations.
- [speculative] Coarse nodes, cross pivot selection and variable order can each contribute independently to the reported bias.
- [measured] No new experiment has run at this protocol commit.

## Alternatives not ruled out
1. A contraction/indexing error could mimic a negative result. Falsifier: direct enumeration and independent TT-SVD on exact low-rank tensors, including weighted contraction.
2. The discrete tensor might be approximated well while the grid misses mass. Falsifier: compare full-grid integral with the separate T4 reference, and cross reconstruction with the exact same grid.
3. Configuration-first ordering may hide narrow high-mass fibers. Falsifier: predeclared configuration-last order with identical ranks and sweeps; weighted SVD supplies an oracle rank benchmark, not a deployable rescue.
4. Larger ranks, residual-driven enrichment, local/continuous coordinates or a different grid could differ. These remain untested; this bounded review does not close the whole tensor family.

## Failure modes
Rank-deficient pivots, ill-conditioned interpolation, shared walker or geometry errors, underestimated algebra cost, full-grid validation not representative of H16T10, and cancellation between grid and compression bias. Computational failure is classified separately from approximation error and from mathematical impossibility. Only the DRI classifies claims after independent review.

## Algebra cycle
[measured] Independent weighted TT-SVD reconstruction, explicit scalar contraction, direct tensor enumeration and an independently selected maximum-area CUR identity pass the frozen low-rank controls. Largest reconstruction error is3.38e-14 and CUR error7.11e-15. Author cross contraction agrees with independent enumeration/scalar recursion within1.78e-15, including rank-deficient constant/zero cases. No contraction/indexing failure was found. CPU0.565 seconds; no neural model evaluations.

[measured] The predeclared localized rank-one spike is missed entirely by the author's deterministic initialization/sweeps:45 queried entries are zero, the returned integral is0, and the true weighted integral is0.005291005291005291. Independent weighted TT-SVD reconstructs it within4.44e-16. This is a concrete pivot-discovery failure despite exact low rank, not a contraction bug or a proof against low-rank tensor representations. Neural full-grid bias/compression separation is the next frozen cycle.

## Full-grid cycle
[measured] All frozen cases completed without an exception in1.805CPU seconds including checkpoint loading after imports. The shared corrected walker evaluated5,472 grid faces, charged90,922.09 FPE proxies. No compiled walker or stochastic input sampling was used. Source hashes and complete face/fitting statistics are in `grid_results.json`. Cross fits replay these complete arrays as oracle values; their separate unique-query costs sum the recorded per-entry walker charges and author algebra proxy. They are counterfactual construction costs, not additional model evaluations or a newly validated portable operation bound. Full-data TT-SVD is diagnostic only.

The following errors are percentage points. Cross/SVD errors are relative to the same fully enumerated grid; grid bias is relative to the separate T4 reference.

| H | Grid order | Grid bias | Rank4 cross, configuration first | Rank4 cross, configuration last | Rank4 weighted SVD, configuration first |
|---|---|---|---|---|---|
|2|3|-.88222|+.33815|+.22037|+.12109|
|2|5|-.30001|-1.55527|+6.73267|+.05907|
|4|3|+1.63427|+1.24947|+5.52422|+.14065|
|4|5|+.35563|+6.99049|+9.34715|+.11256|

[measured] The configuration-last rescue reduces absolute cross integral error in4 of8 frozen rank/model/order settings and worsens4. It is not a reliable repair. Rank2 cross errors relative to the grid range from1.263 to27.331 percentage points in the baseline order; oracle weighted SVD is substantially closer on these same grids. The configuration-first rank4 SVD integral errors are0.059–0.141 percentage points. These observations distinguish cross support/pivot selection from low-rank capacity, while full-grid bias0.300–1.634 percentage points shows that coarse quadrature remains a separate obstacle. Cancellation can make a bad cross fit look closer to network accuracy; no setting is selected by that cancellation.

The evidence verifies failure of these specific frozen cross/settings on additional small neural tensors. It does not reproduce a complete H16T10 tensor, establish that all neural tensors have low rank, or certify an improved estimator. Residual-driven enrichment, local decomposition, higher ranks, adaptive grids and neural-specific support constraints remain open. No runtime or precision failure was used as a mathematical negative.

## Support-discovery obstruction and its scope
Root independently supplied the following limited argument. Let a deterministic algorithm access an arbitrary nonnegative tensor only through point queries, and suppose it leaves an unqueried index after receiving zero at every query. A tensor supported only at that index is a rank-one product of coordinate indicators. It produces exactly the same response history as the zero tensor, so the algorithm returns the same estimate for both. With strictly positive quadrature weights the two true integrals differ. Therefore rank one alone cannot guarantee exact integral recovery with fewer than every grid point in this worst-case class. For spike height at most1, at least one of the two absolute errors is at least half the product weight of the unqueried index.

This is an indistinguishability proof for arbitrary point-query tensors, not a proof against the neural tensor family, structural access to weights, or approximate average-case recovery. Known support, incoherence, probabilistic support coverage or additional structure can change the task. In the special strictly positive rank-one case, one nonzero base entry and its coordinate fibers already determine the entire tensor by products of ratios; positivity alone does not supply the analogous guarantee at higher rank. Weighted residual enrichment is plausible empirically but cannot defeat the arbitrary zero-response worst case without additional assumptions.

## Exhaustion checklist and handoff
- Checked independent contraction, reconstruction and CUR algebra; no defect found.
- Checked exact weighted rank2/constant/zero fixtures and an adversarial rank1 support fixture; pivot discovery can fail despite low rank.
- Fully enumerated both families on two frozen small neural models at two grid orders; grid and compression errors separated.
- Tested one predeclared variable-order rescue with unchanged ranks/sweeps; mixed outcome, not a general repair.
- Did not test residual enrichment, larger ranks, localized/continuous grids, neural support assumptions or full H16T10 enumeration; these are open, not exhausted or impossible.
- Shared face/walker source remains a common dependency with qualified floating accuracy. Root/DRI independent review and classification remain separate; this review does not mark its own conclusion confirmed.
