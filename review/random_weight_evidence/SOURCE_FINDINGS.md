# T248 frozen source and evidence limits

Author a5c2a0b5, preregistration1d6f4db9; claimc5da29ed. This is text/Git provenance review. No author imports, array decoding, model generation, numeric mass integration or result contraction has occurred.

## Implemented comparison

`rc.py` implements target; five Frobenius-matched Gaussian draws (100..104); five row-matched draws (200..204); one recurrent-row permutation (300); one elementwise recurrent sign flip (301). Row-matched scalar input weights retain trained magnitudes with independent signs. Readout matrices are randomized too for the random ensembles. They affect diagnostic accuracy but are not used in exact prefix gate-history enumeration. The two random ensembles have different laws and should be summarized separately as well as in the preregistered pooled ten-draw median. Permutation/sign-flip controls are outside that median.

`prefix.run` starts from a simplex fan and the two first-input signs, recursively passes exact=True, and counts both terminal pieces (`cells`) and distinct packed gate histories (`regions`). The exact branch resolves source-detected cuts; floating tolerance1e-10, clipping, rank and inverse calculations remain. This does not independently certify every new random model's geometry. T219's earlier target recount is useful corroboration, not a count certificate transferable to new weights. Readout/correctness is not used to subdivide these prefix histories, so these are not homogeneous final correct/error cells.

## Mass evidence is aggregate-only

At k4 the driver sums numerical `estT.mass` values into history-keyed `w`, then sorts values and normalizes by their sum. It stores only N_delta, inverse Simpson and total-minus-one. Neither `w`, per-piece masses, rays nor histories are serialized by this driver. Thus a later saved review can reconcile the 39 per-run JSON records with summary/log/report, but cannot independently recompute these 39 coverage thresholds, per-history positivity, minimality, inverse-Simpson values, mass ordering or original geometry. Near-unit total is not a certificate of individual mass accuracy.

The reached mass source uses a 10-cubed positive Duffy rule for small determinants and a finite log-transformed128-point Plackett rule otherwise. Floating floors and determinant clipping occur internally. These are numerical integration approximations, not exact probability arithmetic, despite the author's phrase exact leaf masses. There is no saved individual mass error bound. Source `rc.py` does not reject a negative/nonfinite history mass before normalization; whether that happened cannot be determined from aggregate records alone. This is an observability limitation, not an assertion of a failed run.

## Diagnostic accuracy and provenance

Each k2 process generates one million float64 normal10-input rows from default_rng(7); thus source uses common inputs across models if runtime generation is identical. These are not independent per-model Monte Carlo streams. Prediction uses first-index argmax and truth uses argsort second-largest. Only mean correctness and final-state positive fraction are saved. No per-input outputs, integer hit field, RNG state, realized model bytes, target checkpoint hash or runtime package/binary identities are bound in the original records. An exactly recoverable integer from a saved mean would still be a reported statistic, not independent forward validation. All reconstructed runtime semantics remain source-qualified.

The frozen source imports local m43 (including an unrelated NPZ at import) and externally installed alg_zoo to load the target. Local import-time dependency bytes are frozen without decoding; external runtime/checkpoint identity is unbound. Git proves preregistration ancestry and separate later source/result publication, not wall-clock execution ordering or lack of unrecorded exploratory runs. The summary builder is not in the frozen folder; saved arithmetic can still test its output consistency.

## Selection, uncertainty and interpretation

The script runs all13 models at k2,3,4 and skips existing output files; it has no k5 loop or stale-output source-hash check. README explains no k5 run via extrapolated resource cost, while preregistration mentions k4 runtime under roughly one minute and a ten-minute k5 budget. Preserve that omission and ambiguity rather than claiming a measured k5 comparison. `drive.sh` uses one thread,600s per invocation and2500000KiB virtual-memory setting; this is historical author accounting, not our current20CPU/2GiB review contract. Timing in prefix/rc is process CPU, with target/model creation before the rc timer; mass counter work is not added to saved ops_geometry. There is no complete estimator-cost comparison.

The preregistered phrase no training effect is a finite descriptive median category (within30% at k4). It is not a formal equality/equivalence test, universal training-effect null, architecture ceiling or generic higher-depth growth theorem. Random ensembles and trained target differ in sparsity/gain/spectrum and one trained target is not a sample of training seeds. T236's exact rescaling and generic-depth-two statements remain distinct from these finite measurements. Coarse mass concentration, rare-tail coverage and full-function simplicity are separate. A median advantage at a chosen delta does not bound every model, every delta<=1e-4 or a necessary accuracy-estimation cost.

## Useful next check and alternatives

After a separate published source/resource gate, reconcile all39 per-run scalar records, the13-model summary, per-ensemble/pooled exact count ratios, descriptive thresholds, rounded text and diagnostic/timing consistency. Preserve all discrepancies without source replay. Missing per-history masses and realized model/checkpoint identities can only be supplied by existing author artifacts or a separately scoped capture; this review does not authorize recapture. Direct functional aggregation, different ensembles, higher depths and error-conditioned complexity remain open.
