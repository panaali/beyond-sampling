# T11580 strict compiled fast-enclosure gate

Claimf058ce6f precedes all implementation. This is a port of the already independently reviewed stored3x3 determinant enclosure, not a new mass rule or a point-accuracy guarantee.

## Frozen contract

The C entry receives nine binary64 inputs and a five-double output. Status0 writes point, lower, upper, positive permanent and error allowance. Status1 requests the existing charged Python exact fallback; status2 reports nonfinite unknown; status3 reports unsupported floating environment; status4 reports internal zero-error inconsistency. It does not implement a free hidden fallback. All error statuses leave output ownership with the caller.

The14-operation determinant and14-operation permanent trees match primitive.py statement order. Error is2^-50 times permanent; outward subtraction/addition and nextafter match exactly. Zero allowance yields positive-zero endpoints, while preserving the point sign. Finite/range guards remain explicit.

Require binary64 double, x86_64/SSE arithmetic, FE_TONEAREST, FTZ/DAZ disabled and SSE exceptions masked. Read runtime fenv/MXCSR before work; do not silently reset caller state. Nextafter and fenv implementations remain named external calls until separately priced. Check ordinary round-to-nearest again after each controlled call; do not claim support if later asynchronous caller changes the environment.

## Build gate

Pin compiler path/version/binary SHA and exact command. GCC14.2 flags: -O2 -std=c11 -fPIC -shared -fno-fast-math -ffp-contract=off -frounding-math -fexcess-precision=standard -fno-tree-vectorize -fno-tree-slp-vectorize -msse2 -mfpmath=sse. Capture full disassembly, dynamic dependencies, library/source hashes. Refuse overwrite of existing build/result artifacts. No permissive fast-math fallback.

## Existing fixed controls

Reuse25 parent results.json input_hex rows and64 saved_minors/results.json minor_hex rows in their existing order. Verify source/result hashes. For17+64 fast inputs require status/point/endpoints bit parity with the saved binary64 reference; unsigned primitive endpoints for signed minors are reconstructed by reversing the saved sign wrapper, then reapplied for signed parity. Permanent/error parity is checked against the frozen Python primitive, not a new determinant reference. For5 finite out-of-range controls require fallback-request status; run only the existing exact fallback and check its prior saved results with bit parity. Three nonfinite controls remain unknown. Preserve cancellation cases and interval widths unchanged.

Four additional environment controls only (no new matrix selection): identity under downward/upward/towardzero modes and FTZ+DAZ toggled. C must return unsupported and the test must restore the original environment in finally; abort on restoration failure. The three rounding changes and MXCSR change are test wrappers, never production adaptation. Freeze wrapper source and independent source gate before execution.

## Cost interpretation

Inspect actual reached scalar arithmetic and classification/abs/control instructions after parity. Record SSE packed/unused lanes, scalar expression counts, external calls and memory/control separately; no complete-speed or104-versus-newcount claim. Historical104 native determinant count is an upper subset, so smaller new named arithmetic cannot demonstrate actual savings. Matched native baseline capture would require a separate scope. Exact fallback arbitrary-precision cost remains explicit.

Two alternatives: compiler reassociation/fusion versus literal-tree parity (assembly and89 controls); hidden floating environment dependence versus guard failure (four environment controls). Passing finite controls does not prove arbitrary downstream mass accuracy. Source theorem/range assumptions remain required. One thread2GB<=20CPU under review lock; no native determinant/mass/model/MC calls and no shared adoption. Independent source gate before test, final raw audit before handoff.
