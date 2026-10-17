# T11586 exact edge-split discriminator

## Question
Can one certified positive-edge split remove the shape failure of the frozen5D product controls, and what accuracy/cost remains unresolved?

## Method
Claim65930317. Reuse all11 original T11583 stored binary rays; one fixed split on rays0/1, no adaptive edge/order choice. Construct v=(r0+r1)/norm(r0+r1) in ordinary binary64, but certify the ACTUAL returned v by exact rational coordinates in the original ray basis. Strictly positive coefficients only on rays0/1 certify a disjoint-interior two-cone partition. Record rounding from intended positive sum separately; unsupported geometry remains unknown. Independent geometry/source gate precedes mass evaluation. Then unchanged source forcedDuffy6/10/16 on22children, plus reviewed exact ranges/Directed50 ideal-truncation bounds; no tuned order or trained model.

## Status
Bounded execution complete at57531a1b. Independent geometry saved audit and numerical source gate passed before66 fixed child evaluations. Final numerical saved audit pending. T11583 radial saved review c39c57fa passes.

## Results
All eleven actual-binary edge splits accepted and all seven rejection controls passed. Exact coefficients, determinants, common-section identities and rounding residuals are saved in geometry.json. Geometry run: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/edge_split5/geometry.py` (immutable output, do not rerun in place). At order16 all11 groups pass absolute1e-10 and relative1e-6, worst absolute error5.130794205e-12. Orders6/10 pass3/5 absolute cases. Only2/11 ideal bounds qualify at each order; the remaining certification gap is retained. See HANDOFF.md and summary.json. Numerical command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-compute.lock /workspace/clean-neurons/.venv/bin/python review/edge_split5/numerical.py`; this output is also immutable. Inputs/pins are in protocol.json.

## Compute accounting
One thread,2GB, primary<=110CPU and review<=20CPU. Fixed22child×3order rule budget; retain node counts and inherited source Ops as proxy, exact rational operation counts/bit widths, directed categories and normalization arithmetic separately. Geometry research cost 0.027403172 CPU seconds, peak34MB. Eleven successful normalizations carry220 named arithmetic units; exact-operation totals {'multiply': 4147, 'divide': 1067, 'subtract': 3742, 'add': 418} with maximum133-bit intermediates. Rejected-path exact counts and verification/control/import costs are not fully priced. Numerical research cost0.248005002CPU, peak50MB. All66 source-proxy evaluations total67,620,410 units, plus33 assembly additions and separate setup. Child radial exact intermediates reach137bits; all counts are retained in summary.json. No full native or hardware cost claim.

## Claims
[inferred] Positive edge subdivision preserves Gaussian mass because it partitions the cone up to a codimension-one boundary. [measured] The fixed order16 split evaluations pass both diagnostic tolerances on all11 synthetic fixtures; lower orders retain failures. [measured] The reviewed ideal bounds remain too loose on9/11 fixtures at every order. [speculative] Sharper bounds may certify this numerical rescue cheaply.

## Alternatives not ruled out
An apparent rescue may come from an altered floating cone; exact certification of actual child rays separates that. Numerical agreement can coexist with a loose truncation bound; both are reported separately. Better coordinates, analytic integration and adaptive rules remain open.

## Failure modes
Zero/nonfinite normalization, singular parent, non-edge rounding or nonpositive coefficients are rejected/preserved. Child costs can exceed benefit. No universal efficient subdivision or trainedT5 accuracy claim follows; ideal-rule bounds exclude nodes/evaluation/determinant floating errors. All source failures remain preserved.
