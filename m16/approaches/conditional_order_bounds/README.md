# Conditional order-statistic uncertainty bounds

## Question
Can a weight-derived uncertainty allowance preserve information from omitted
Gaussian inputs without replacing their effect by a deterministic mean?

## Method
T189 claim439be010 precedes code. First scalar phase only: eight frozen
truncated-normal absolute-deviation MGF/expectation cases, four domain/degenerate
controls, direct-density integral checks, and the convex-MGF mean precheck.
Reuse T187 global margin bound; preserve T138/T141 summary and averaging caveats.
THEORY.md states the rank-conditioned independence and all limits. No omitted
rank information is silently added. Future network controls need a new gate.

## Status
Source prepared; independent source gate required before one bounded run.

## Results
Pending. Planned command under20CPU/2GB and one thread: `flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/conditional_order_bounds/controls.py`.

## Compute accounting
Only scalar MP formulas and independent direct Gaussian integrals; all CPU and
attempts retained. No model, training, MC, checkpoint or estimator evaluation.
Native/MP operation cost and outward floating guarantees are not claimed.

## Claims
[inferred] The conditional law and MGF identities follow from Gaussian density.
[inferred] Convexity can certify vacuity of this fixed bound before a lambda sweep.
[speculative] Rank-aware uncertainty bounds may help avoid gate enumeration.

## Alternatives not ruled out
The T187 global absolute envelopes may be loose; conditional gate reachability,
restricted input ranges, weighted summaries and different anchors remain open.
Failure of this finite primitive/control phase does not retire any method family.

## Failure modes
Conditioning on additional omitted ranks violates independence; numerical CDF
cancellation can obscure values; zero margins and permanent ties need explicit
network handling. Preserve any failure, with no automatic precision relaxation.
