# Positive analytic inner integral in dimension five

## Question
Can a positive, cancellation-resistant inner integral reduce fixed cone-mass
quadrature dimension without hiding domain or arithmetic failures?

## Method
See PROTOCOL.md and primitive.py. First gate is fixed analytic/domain/scaling
controls, not a trained estimator. The same-sign rationalization removes an
explicit inverse Gram determinant from the endpoint difference.

## Status
Running: bounded primitive phase complete8440898e, independently4cc6f14e.
See HANDOFF.md; a separate frozen outer-rule gate is next. The original source
protocol/README snapshot is preserved atf9ca116b for reproduction.

## Results
All16 supported controls satisfy the frozen relative diagnostic;10 explicit
unsupported/singular/extreme/nonfinite outcomes remain. No source rule changed.
See HANDOFF.md for numbers, qualifications and exact reproduction command.

## Compute accounting
Named scalar ledger plus separate reference/setup CPU; integer, dispatch,
memory, library/native and hardware costs remain excluded. No model forwards.

## Claims
[inferred] The positive factorizations are exact on the stated real domain.
[speculative] They may reduce actual outer quadrature cost after later gates.

## Alternatives not ruled out
Delta/scale conditioning can dominate; source rounding and outer approximation
need independent bounds. Collinear finite segments may admit a separate branch.

## Failure modes
Nonfinite input, origin, delta underflow/collinearity, scale-factor overflow or
underflow. All remain explicit, and finite controls cannot prove general FP
accuracy or retirement of alternative primitives.
