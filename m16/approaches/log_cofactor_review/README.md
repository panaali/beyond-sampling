Latest source139a7b9e has a separately reviewed continuation in [latest/HANDOFF.md](latest/HANDOFF.md); all35098 actual cases pass its frozen diagnostic tolerance, while four generic thin-shape controls fail. Older-source results below remain version-specific.

# T179 log/cofactor rule audit

## Question
Does the unchanged DRI ac321d97 mass rule pass fixed independent analytic and saved positive-reference checks, without relying on aggregate cancellation?

## Method
Frozen protocol.json pins source and references. First analytic orthant/equicorrelation controls and original negative fixture672; then256 evenly spaced original error cones after independent gate. Observe every raw cofactor product/partial correlation and floor/clipping hit, retain source cutoff and individual errors. No geometry rerun or new model inputs.

## Status
Pilot independently reviewed882efc72. Full35098 finite phase complete and fails one frozen absolute-tolerance case; see FULL_HANDOFF.md. Full raw review ba401fec independently confirms the scoped failure.

## Results
`python controls.py` from this folder reproduces18 controls. All observed source values match unobserved source bitwise; worst analytic/fixture absolute discrepancy9.283159210226312e-16 at fixture672 (new1.4731889319552494e-11 versus positive1.473096100363147e-11). This greatly improves the old negative result, without certifying all floating errors. Raw nodes in control_nodes.npz; third cofactor array stores the unsigned complementary determinant before source parity sign. Diagnostic thresholds are frozen tolerances, not rigorous bounds.

## Compute accounting
One thread2GB,20CPU review/110CPU primary. Source inherited counts and explicit native call multiplicities are separate from complete cost; diagnostics/reference reuse are not free candidate work. No full-cost claim. Each noncut source cone uses18 batched3x3 determinant calls (2304 scalar determinants), one4x4 inverse and one4x4 determinant, plus all array arithmetic and elementary functions. Source inherited counter41628 is partial. Each observation is a separate full replay charged as review; rule construction and hashing are setup/review work. Control process .330412888CPU including imports/setup and replay.

## Claims
[inferred] Log substitution has a tiny true omitted tail, but this does not bound interior quadrature or floating cancellation. [speculative] Fixed source may remain accurate on thin reference cones despite cofactor floors.

## Alternatives not ruled out
Interior quadrature error versus covariance/cofactor rounding: individual saved positive references distinguish total error, raw-node diagnostics localize potential instability. Signed cancellation can hide errors: report maximum and L1 as well as signed aggregate. Higher precision or other representations remain untested, not retired.

## Failure modes
Near singular covariance, cancellation, principal-minor floor, clipping, unbounded fixed quadrature error, inherited determinant cutoff, conditional reference bounds and incomplete native cost. No every-scale or full-reference guarantee from a finite pass.

Fixed panel: `python panel.py` after source gate. Max absolute error3.86817e-14 versus old1.31271e-6; L1 error2.20596e-13. No floors, clipping, cutoffs or negative values. No full corpus or floating certificate.

Full corpus source accuracy .9761608074917736 matches DRI; positive .9761608075466027. Largest individual error2.06288e-10 exceeds frozen1e-10 diagnostic tolerance; one tiny negative mass,9clipped cones and73cutoffs preserved. Full result supersedes pilot-only positivity/no-clipping statements.
