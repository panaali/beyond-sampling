# T69 structured prefix mixture

## Question
Does retaining the exact two-input gate-sector label improve Gaussian moment propagation beyond collapsing those exact prefix moments into one Gaussian? T80 found that exact h2 moments improve but do not fix the historical H16T4 seed3 closure. This is the first bounded discriminator before the original random-width T69 question, not completion of that broader question.

## Method
Claim187346b7 freezes every h2 gate sector as one component. Its probability is angular width/(2pi); divide exact-real sector integrals by that probability to obtain conditional mean/covariance. Propagate the unchanged inherited Gaussian closure separately from each component at steps3,4, retaining weights; recombine with both within- and between-component covariance only for reporting. Pin sources and reuse the same T80 model and eight-block reference. No new input draws or final-accuracy evaluation.

Two competing explanations: most loss comes from merging distinct prefix activation patterns, in which case preserving them should reduce error; or within-sector non-Gaussian/support structure dominates, in which case this repair can still fail. Six frozen zero/duplicate/opposite/diagonal/mixed controls test weights/recombination before trained use. Component Gaussian support is not the original sector support.

## Status
Single-model source/aggregate independently reviewed23c5c972. The separately frozen24-cell random-width phase is complete; independent full-panel audit pending. The broad structured-propagation family remains open.

## Results
See the fixed single-model result below. Prior reference reused unchanged; no old model calculation repeated.

## Compute accounting
One thread2GB; review cycles<=20CPU, primary<=110CPU. Existing reference construction costs remain charged in T80, not repeated or presented as candidate work. Conditional moment construction, one Gaussian closure per component per remaining step, recombination and model-dependent setup are candidate costs; report source structure and CPU, not full scalar/hardware cost without an audit.

## Claims
[speculative] Keeping prefix sector identity may reduce structural mixing error. No success, impossibility or competitive-cost claim.

## Alternatives not ruled out
Within-sector radial/truncation effects, finer or later conditioning, enriched moments and support-aware endpoints. Original random-width/K question remains separate.

## Failure modes
Ordinary floating angles/moments; narrow-sector cancellation; tiny covariance eigenvalues; inherited Gaussian quadrature/clipping; retained labels describe only the first two inputs. Recombination may hide individual component defects, so mass and covariance diagnostics are retained.

## Frozen single-model result
Source gate passed independent T55 review before trained evaluation. All six controls preserve unit total mass and the original aggregate exact-prefix moments (max discrepancy8.88e−16). The trained H16T4 seed3 has34 sectors; weights sum1, minimum7.0981e−5. Tiny component covariance eigenvalue−1.72e−14 is retained without projection.

| Step | Collapsed exact-prefix mean relative RMS | Sector mixture | Collapsed covariance relative RMS | Sector mixture |
|---|---:|---:|---:|---:|
|2|.002602|.002602|.004649|.004649|
|3|1.065742|.092078|6.878440|.307598|
|4|2.955124|.706845|20.974102|3.192038|

The additional conditioning substantially improves this fixed model, but final hidden moments still have large errors. Against the original unmodified Gaussian closure, step4 mean error falls5.663295→.706845 and covariance56.822947→3.192038. No output-accuracy implication follows directly. Step4 paired eight-block reductions relative to the collapsed reset are2.24825±.00293 descriptive SE for mean and17.78150±.04168 for covariance; no rigorous interval or selection-adjusted claim.

Reproduce in this folder using `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python run.py controls` then `trained`, under codex-review.lock and20CPU/2GB. Controls used.001400CPU; trained process.806435CPU including imports/model load, candidate continuation.023999CPU. Candidate performs68 inherited Gaussian layer calls plus component setup/recombination; no full-cost assertion. `trained.json` retains every conditional moment at every step, so aggregate improvement does not conceal individual states. Reference generation is the already charged T80 work, with no new model input draws.

## Frozen random-width diagnostic
Scopea15a7b01/protocol26c8cbee addresses the original T69 random-network question with a finite panel: T=4, H=4/8/16/32, three He-initialized model seeds0/1/2, scalar or H-dimensional fresh inputs. Weight order/scales match the original `diag_width.py`; references use separately frozen streams and131072 samples per cell,8blocks16384. All24 cells and72 candidate summaries completed. The diagnostic is output-mean squared error divided by empirical output variance, averaged over output coordinates and the three model seeds. The nominal reference-noise floor is1/N=7.6294e−6; saved floor-subtracted values are not clipped. Estimated denominators and finite blocks prevent treating that correction as an exact confidence statement.

| H | Scalar Gaussian | Scalar arc2 | Collapsed exact h2 | Sector mixture | Vector-input Gaussian |
|---|---:|---:|---:|---:|---:|
|4|.00563218|.00331216|.00512906|.00078565|.00771914|
|8|.00539993|.00485613|.00404790|.00213586|.00038705|
|16|.00402555|.00620176|.00308976|.00096481|.00017610|
|32|.00422308|.00555799|.00279867|.00094790|.00012252|

The mixture improves on the ordinary Gaussian closure in all12 scalar-model cells. It is worse than the collapsed exact-prefix reset on H8 seed1 (.004078 versus .003615), retained as a failure. Mean error is nonmonotone over width and remains about1e−3 at H16/32, above the vector-input Gaussian comparison in those cells. These observations do not establish a width-scaling cure, an asymptotic lower bound, failure of all structured encodings, or final classification accuracy. Only three model seeds were used; no test panel was enlarged or tuned after results.

Mixture component counts are10/18/34/66 at the four widths, requiring20/36/68/132 inherited Gaussian closure calls per model. Candidate cost increases with these counts, so lower bias does not itself imply better equal-compute estimation. Some very short per-method CPU readings are0.0 at clock resolution, not free work. Four process chunks used3.190287CPU seconds (cell work3.132682, reference work2.326073); references generated100663296 scalar normals. `random_H*_d*_s*.npz` retains weights, output reference blocks and final component arrays; JSONs retain every estimate/error/count/time and archive hash. Main reference arithmetic categories are explicit; RNG internals, summaries and runtime remain unpriced. Candidate timings and call counts are descriptive, not a complete scalar cost audit.

Reproduce each width using `random_width.py H` with the same thread/interpreter environment under compute.lock/110CPU/2GB, then `report_random.py` for saved contraction. Existing completed cells are hash-pinned and skipped. `random_summary.json` includes all per-seed values and the H8 seed1 loss. Independent raw/source review is requested; no additional numerical run is warranted without a distinct preregistered question.
