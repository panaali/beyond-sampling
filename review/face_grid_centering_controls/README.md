# T247 actual analytic grid centering controls

## Question
Does the previously proved formal right-endpoint centering bias survive the
actual stored T2 source grid, and does one local trapezoidal alternative restore
centering within fixed diagnostics?

## Method
`control.py` extracts unchanged source AST from `frozen/core.py`, records its
single T2 construction, and compares a separately implemented trapezoidal CDF.
`PROTOCOL.md` freezes three tiny controls, the actual analytic grid, exact stored
array contractions, finite interpolation probes and all scope/resource limits.
`protocol.json` binds source/runtime and completed predecessor evidence.

## Status
Sole gated four-case phase complete in `6d2db5f4`; all fixed controls passed.
h4 source review `006ca7d5` is complete; T55 independent saved review is pending.
No trained phase authorized.

## Results
See [RESULTS.md](RESULTS.md) and immutable `output/` for all four controls.
Historical sole execution command (do not rerun completed work):

```sh
set -e
set -o noclobber
ulimit -v 2097152
ulimit -t 20
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/face_grid_centering_controls/control.py > review/face_grid_centering_controls/run1.log 2>&1
```

## Compute accounting
Planned one original T2 table and four local alternatives, three tiny original
cumsums,18 finite interpolation probes and eight exact stored-law contractions.
All source construction, diagnostic integer arithmetic, copies/hash/I/O and
library setup are research work. Model, face_points and RNG calls zero.
No cost or speed advantage claimed.

## Claims
[inferred] T238 establishes bias only under its formal symmetric-real premises.
[measured, author; independent saved review pending] The original stored-law
central gap is2.82095e-5 and the local alternative meets frozen centering checks.
Separate normalization and finite-input-law limitations remain.

## Alternatives not ruled out
Actual grid/ordinate/cumulative rounding; tail plateaus; finite latent endpoint
semantics; asymmetric higher-T densities; Gaussian-law and Z integration error;
trained integrand sensitivity; better quadrature or conditional representations.

## Failure modes
A centered stored-grid law can still have substantial non-centering integration
error. Finite np.interp endpoints are not the continuous latent law. A change to
CDF does not automatically repair Z. Library/source results are not outward
certificates of ideal Gaussian integrals. Resource/diagnostic failures remain
visible and do not declare the tie-slicing family exhausted.
