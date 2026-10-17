# T179 log/cofactor rule audit

## Question
Does the unchanged DRI ac321d97 mass rule pass fixed independent analytic and saved positive-reference checks, without relying on aggregate cancellation?

## Method
Frozen protocol.json pins source and references. First analytic orthant/equicorrelation controls and original negative fixture672; then256 evenly spaced original error cones after independent gate. Observe every raw cofactor product/partial correlation and floor/clipping hit, retain source cutoff and individual errors. No geometry rerun or new model inputs.

## Status
Protocol frozen; implementation pending.

## Results
None yet.

## Compute accounting
One thread2GB,20CPU review/110CPU primary. Source inherited counts and explicit native call multiplicities are separate from complete cost; diagnostics/reference reuse are not free candidate work. No full-cost claim.

## Claims
[inferred] Log substitution has a tiny true omitted tail, but this does not bound interior quadrature or floating cancellation. [speculative] Fixed source may remain accurate on thin reference cones despite cofactor floors.

## Alternatives not ruled out
Interior quadrature error versus covariance/cofactor rounding: individual saved positive references distinguish total error, raw-node diagnostics localize potential instability. Signed cancellation can hide errors: report maximum and L1 as well as signed aggregate. Higher precision or other representations remain untested, not retired.

## Failure modes
Near singular covariance, cancellation, principal-minor floor, clipping, unbounded fixed quadrature error, inherited determinant cutoff, conditional reference bounds and incomplete native cost. No every-scale or full-reference guarantee from a finite pass.
