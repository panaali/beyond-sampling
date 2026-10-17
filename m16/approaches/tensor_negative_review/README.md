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
