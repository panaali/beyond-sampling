# T69 structured prefix mixture

## Question
Does retaining the exact two-input gate-sector label improve Gaussian moment propagation beyond collapsing those exact prefix moments into one Gaussian? T80 found that exact h2 moments improve but do not fix the historical H16T4 seed3 closure. This is the first bounded discriminator before the original random-width T69 question, not completion of that broader question.

## Method
Claim187346b7 freezes every h2 gate sector as one component. Its probability is angular width/(2pi); divide exact-real sector integrals by that probability to obtain conditional mean/covariance. Propagate the unchanged inherited Gaussian closure separately from each component at steps3,4, retaining weights; recombine with both within- and between-component covariance only for reporting. Pin sources and reuse the same T80 model and eight-block reference. No new input draws or final-accuracy evaluation.

Two competing explanations: most loss comes from merging distinct prefix activation patterns, in which case preserving them should reduce error; or within-sector non-Gaussian/support structure dominates, in which case this repair can still fail. Six frozen zero/duplicate/opposite/diagonal/mixed controls test weights/recombination before trained use. Component Gaussian support is not the original sector support.

## Status
Protocol frozen; implementation and controls next.

## Results
None yet. No old result is rerun.

## Compute accounting
One thread2GB; review cycles<=20CPU, primary<=110CPU. Existing reference construction costs remain charged in T80, not repeated or presented as candidate work. Conditional moment construction, one Gaussian closure per component per remaining step, recombination and model-dependent setup are candidate costs; report source structure and CPU, not full scalar/hardware cost without an audit.

## Claims
[speculative] Keeping prefix sector identity may reduce structural mixing error. No success, impossibility or competitive-cost claim.

## Alternatives not ruled out
Within-sector radial/truncation effects, finer or later conditioning, enriched moments and support-aware endpoints. Original random-width/K question remains separate.

## Failure modes
Ordinary floating angles/moments; narrow-sector cancellation; tiny covariance eigenvalues; inherited Gaussian quadrature/clipping; retained labels describe only the first two inputs. Recombination may hide individual component defects, so mass and covariance diagnostics are retained.
