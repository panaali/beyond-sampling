# Positive analytic inner integral in dimension five

## Question
Can a positive, cancellation-resistant inner integral reduce fixed cone-mass
quadrature dimension without hiding domain or arithmetic failures?

## Method
See PROTOCOL.md and primitive.py. First gate is fixed analytic/domain/scaling
controls, not a trained estimator. The same-sign rationalization removes an
explicit inverse Gram determinant from the endpoint difference.

## Status
Running: claim23bfb42e published; source and fixed manifest await independent
review before execution.

## Results
Pending fixed controls. No source numerical rule changed.

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
