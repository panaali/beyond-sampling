# Fixed Q source-expression controls

Claim1bf720bd follows reviewed source prerequisitebfad6901/8840bc29. Freeze exactly
six explicit H16/T10 model/prefix fixtures in fixtures.json: zero readout, zero
input weights, zero recurrence, repeated breakpoints, heterogeneous intervals,
and slopes straddling the unchanged1e-12 cutoff. No random input or stored model.
All original semantics, including tie defect/warnings, are preserved.

AST extraction executes only the pinned h9/Q function definitions with supplied
NumPy/SciPy/T10 globals. No author module, precompute, allocation, draw, basis or
control-variate fit is imported. Observer preserves operator trees and order,
logs actual outputs/call shapes and uses original in-place addition. Indexing,
copying, selection, sorting and special/native library internals remain unpriced.
Diagnostic logging/shape queries are research overhead, not estimator work.

For each fixture run one original/observed batch pair and one pair per row. Save
bit encodings and warning(category,message) multisets before assertions. Check
array real arithmetic and both dense conventions against separately derived
formulas. Scalar/control, Boolean/comparison and function-call records remain
separate. Batch-vs-single values are diagnostics, not an assumed BLAS invariant.
The all-zero-readout source defect must remain zero; no shared repair occurs.

One process under review lock, external20CPU/2GB and internal16CPU, singlethread.
Source gate and all pins published before execution. Refuse existing results or
progress, preserve original output before observed run and all partial failures.
No retries, parameter tuning, new trained case or full multiplier inference.
Independent saved review is required. cv allocation/draw controls are deferred.
