# T159: correctness crossing faces and Gaussian density

## Question
Can exact rank/correctness face feasibility and Gaussian distance improve the same T155 local prefix continuity bounds?

## Method
Freeze H2T4 seed0, u=1, both T155 axis/prefix intervals and x,y in [-6,6]. For every retained original boundary occurrence and both need1 rank sides, eliminate the input coordinate with larger nonzero boundary coefficient. Clip the resulting rational two-dimensional constraints, retaining points/lines. Margin faces include first/middle gates, final support and every target-minus-competitor margin. Final-gate faces conservatively use only first/middle/rank constraints. Deduplicate only after retaining any feasible occurrence. Separate all-word, word-feasible, face-feasible and rational density-damped ablations.

For row A*x+B*y+C*s+E=0, q is the minimum of (C*s+E)^2/[2(A^2+B^2)] over the ENTIRE frozen state interval, including zero if crossed. Multiply the old strip allowance by 1/(1+q), using exp(-q)<=1/(1+q). Exact proportional rows have equal damping.

## Status
Protocol frozen after TASKS-only claim7b3a264. Independent T55 theorem review passed; implementation controls and raw review pending.

## Results
No new result yet. Planned finite controls cover both elimination choices, rank sides, empty/line/point faces, correctness constraints, any-occurrence retention and density minima/scaling. Only the two prior trained intervals will be applied after controls.

## Compute accounting
No model forward, Monte Carlo, Gaussian pair integration or interpolation run. Charge exact rational feasibility, coefficient arithmetic and one checkpoint load separately; one thread,2GB, <=20CPU review cycles.

## Claims
[inferred] Closed crossing-face restrictions and whole-interval density bound preserve the endpoint-strip proof under the inherited nonzero-normal guard. No population or estimator claim.

## Alternatives not ruled out
1. Apparent gain could be invalid removal of a degenerate crossing: exact line/point controls and independent intersection enumeration discriminate.
2. Gain could come entirely from prior word pruning: freeze separate face/density ablations.
3. Conservative faces may remain loose; tighter support ownership, piecewise bounds and direct integration remain open, not attempted here.

## Failure modes
Zero input normal with state dependence stays unknown. All-off ties remain covered by broad final-gate faces. No claim about old mixed-support stage2 interpolation. Prefix bounds apply only to two local intervals and the truncated pair function; truncation/outer integration remain separate.
