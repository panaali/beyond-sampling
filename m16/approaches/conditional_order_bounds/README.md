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
Eight scalar and four domain controls completed once after independent T55 source/theory gate5fae242f; saved-result independent audit passed dfb2a5f4 (h4), with erf160 contractions and no integral replay. Seven analytic network controls now separately claimed67830ea3; source/theory gate passed47e280de and all seven completed once; independent saved-network audit pending.

## Results
All8 formulas/expectations agree with direct-density controls (maximum absolute difference1.7562e-95); all4 domain/degenerate controls pass. Fixed mean-plus-quarter bounds improve below1; mean-minus-quarter controls retain the convex no-improvement diagnosis. This is scalar validation, not neural performance. Recorded1.559815508CPU plus.115609266prior/import=1.675424774processCPU. Command under20CPU/2GB and one thread: `flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/conditional_order_bounds/controls.py`.

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

## Seven analytic network controls
Claim67830ea3/source47e280de; T55 source/theory gate before the single run. All seven pass. Strong base exact-conditional numerical probability.999767056471 versus candidate[.997134735667,1]; moderate.920594736485 versus[.533159941147,1]; vacuous.382924922548 versus[0,1]; incorrect anchor.000232943529 versus[0,.00286526431034]; permanent ties exactly0 and1; strong memory.999999659294 versus[.999995570461,1]. These are finite MP conditional diagnostics, not population accuracies. Separate exact rational strong-case error upper742170348748800/80748368484388141 <.01 uses proved rival dominance and a positive exponential-series partial.

The run used7 candidate anchor forwards plus57 exact validation forwards,25 cells/25 boundaries; .027948191 workCPU plus.119474442 prior/import=.147422633 processCPU. All records and progress are retained in network_results.json/network_progress.jsonl/network_run1.log. Command under20CPU/2GB and one thread: `flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/conditional_order_bounds/network_controls.py`. Do not rerun completed controls. Native/MP/Fraction preparation is not priced against sampling.
