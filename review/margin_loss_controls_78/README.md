# T259 common-design margin-loss controls

## Question

Does the implemented exact-rational fit preserve the already reviewed equivalence between fixed-readout ridge and common-design logit ridge, and between matched unweighted all-pair loss and centered logit loss?

## Method

algebra.py constructs ordinary ridge and direct pair-design normal equations, solved by exact Gauss–Jordan elimination. fixtures.py freezes 17 cases. run.py retains full matrices, coefficients, predictions, objectives and failures. PROTOCOL.md fixes expected results, counts, provenance and resource limits. The independent saved reviewer uses separate code.

## Status

Source and fixture preparation. No candidate calls or results yet. Independent source/resource gate and immutable published execution gate are required.

## Results

None yet. Sole authorized reproduction command, after all gates: `bash launch.sh` in this folder. It refuses prior outputs and logs.

## Compute accounting

Planned 36 exact linear-system attempts and 35 returns; the remaining attempt is the fixed singular-system rejection. Seventeen cases, 36 events. No trained data, model, checkpoint, forward, sampling or Gaussian calls. Matrix arithmetic and validation are counted as research work; selected call counts are not a complete native cost.

## Claims

[inferred] The reviewed theorem predicts equality only under common design, matched normalization and fixed linear solution maps. [speculative] Alternative weighted/nonlinear losses or features may improve empirical agreement; this suite does not test that.

## Alternatives not ruled out

Weighted class pairs, top-rival or input-dependent losses; nonquadratic fitting; nonnegative constraints; recurrent rollout loss; richer or target-refitted features; finite-precision differences. No trained accuracy ceiling or compression-family verdict follows.

## Failure modes

Output-specific maps, unmatched penalties, nonlinear clipping and arbitrary singular solutions violate the equivalence assumptions. The suite tests explicit examples, not every such variant. Resource or implementation failure is preserved and would not establish a scientific negative.
