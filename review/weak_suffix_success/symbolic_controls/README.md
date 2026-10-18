# T265 symbolic partition controls

## Question
Does the rational one-input construction preserve gate/readout/rank partitions and permanent tie ownership before any CDF implementation?

## Method
primitive.py builds exact affine pieces and symbolic Gaussian interval expressions. fixtures.py fixes ten models/states and six guards. run.py uses a separate direct scalar recurrence for interval/boundary checks and the single reachable synthetic prefix. Full trace serialized; independent910 saved reconstruction follows h4 source gate.

## Status
Frozen source preparation only; no calls yet. Execution requires the separately published source/resource gate and immutable owner gate.

## Results
[inferred] Symbolic equality is sufficient for equal Gaussian mass. Only one fixture claims synthetic reachability, checked from zero. No empirical results or numerical Gaussian values yet.

## Compute accounting
Planned ten candidate partitions, six rejected-domain attempts, <=400 synthetic point evaluations, three prefix steps; zero trained/model-data/CDF/RNG/fit/LP/mass calls. Named source counts are partial, not timing or full-cost bounds. Command after clearance: `bash launch.sh`.

## Claims
[inferred] Last-input identity follows reviewed THEORY.md. [speculative] A later reachable-state summary could be cheaper; these controls alone cannot establish that.

## Alternatives not ruled out
Arbitrary-root directed CDFs, richer summaries, reachable-state collisions, exact domain simplification and signed error cancellation remain open and separately gated.

## Failure modes
Missing or duplicate roots, permanent tie loss, wrong rank threshold/index, unreachable examples treated as trained merges, incorrect symbolic normalization, invalid input acceptance, and confusing reference construction with compression. Preserve any failure without automatic rerun.
