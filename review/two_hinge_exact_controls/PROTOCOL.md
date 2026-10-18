# Frozen T233 exact synthetic control phase

Claim 291079cc precedes implementation. This is the first implementation of the
reviewed scalar/conditional residual identities, not a target application.
T233's original proof and gauge handoff 70f2d3bd remain unchanged.

Twelve declared fixtures have five exact input values each. They cover both
slope orientations and threshold orders, both anchor-endpoint ownership cases,
coincident thresholds, zero intercepts, a narrow dyadic interval, a wide
interval, small slopes and fractional coefficients. Each uses a positive
diagonal reparameterization and a nonnegative margin multiplier (including zero).
Eight guards reject inexact/bool inputs, invalid slope domains, zero scaling and
negative margin envelopes. No cases are selected from target data.

The candidate reports the signed frozen-gate surrogate and nonnegative residual.
The harness independently uses the original ReLU-minus-surrogate identity.
Its interval expression includes the exceptional anchor endpoint: at a=0,
the residual is b+, which need not vanish. Gaussian probability formulas can
ignore that null endpoint; exact pointwise controls cannot.

`area` is the integral of the triangular residual against Lebesgue measure,
not a Gaussian expectation. Multiplying it by 1/sqrt(2*pi) gives the reviewed
conditional density-cap upper bound. The constant remains SYMBOLIC: this phase
does not evaluate Phi, phi, pi, quadrature or the cancellation-prone expectation
formula, and does not claim a certified floating bound. An independent polynomial
antiderivative checks area; exact positive-diagonal weighted-area invariance is
checked separately. State moments, downstream classification and practical cost
are outside scope.

Execution requires an independent source/resource review by recover_34, a
published immutable gate binding that artifact and every frozen source/protocol
pin, and explicit published execution scope. Gate review JSON must have
complete/pass_all=true and include/verify direct primitive, driver, fixture and
protocol pins. Merely writing this protocol authorizes no run.
recover_fundamentals owns the later independent saved-output audit.

One attempt only: external hard AND soft CPU <=20 seconds, address space <=2 GiB,
one numerical thread and /workspace/work/codex-review.lock. The harness records
both actual limit pairs before assertions and requires each bound, then enforces
17 main-scope CPU seconds at event boundaries. Use the pinned venv interpreter;
`set -e` and `ulimit -v 2097152; ulimit -t 20` before launching. The previous T230
hard-limit launcher deviation is not repeated. No retries or threshold relaxation
are authorized here. Any zero-call runtime failure remains a new gated recovery.

Result snapshots use a flushed/fsynced same-directory temporary and atomic
replacement so interrupted serialization preserves the last complete snapshot.
Fatal handling resets completion flags, including a late timer failure.
Every case begins with durable progress. All returned records are saved BEFORE
arithmetic assertions; fatal records preserve partial results. Existing outputs
and progress are refused. Expected successful progress: attempt +24 case events
+16 guard events +complete =42. Expected interface invocations:120 point,
60 interval_point,12 scale,8 guard attempts. These are harness interface counts,
not a complete primitive-call count: conditional is nested (84 successful calls),
and guard attempts may reject before it. Fraction decoding, arithmetic/GCD,
validation, hashing, allocation and I/O are research work, not free operations.
Main-scope CPU excludes top-level imports and final function return; whole-process
external limits cover all process work. No scalar-FLOP/native/speed comparison.

Keep all finite results and failure cases. A mismatch is an implementation/domain
failure until independently diagnosed, not evidence that the residual family
cannot work. Stable conditional expectation, other slope domains, truncated fresh
input laws, multiple anchors and trained utility remain untried alternatives.
