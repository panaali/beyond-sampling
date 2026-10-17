# T159: correctness crossing faces and Gaussian density

## Question
Can exact rank/correctness face feasibility and Gaussian distance improve the same T155 local prefix continuity bounds?

## Method
Freeze H2T4 seed0, u=1, both T155 axis/prefix intervals and x,y in [-6,6]. For every retained original boundary occurrence and both need1 rank sides, eliminate the input coordinate with larger nonzero boundary coefficient. Clip the resulting rational two-dimensional constraints, retaining points/lines. Margin faces include first/middle gates, final support and every target-minus-competitor margin. Final-gate faces conservatively use only first/middle/rank constraints. Deduplicate only after retaining any feasible occurrence. Separate all-word, word-feasible, face-feasible and rational density-damped ablations.

For row A*x+B*y+C*s+E=0, q is the minimum of (C*s+E)^2/[2(A^2+B^2)] over the ENTIRE frozen state interval, including zero if crossed. Multiply the old strip allowance by 1/(1+q), using exp(-q)<=1/(1+q). Exact proportional rows have equal damping.

## Status
Frozen coefficient phase executed after TASKS-only claim7b3a264 and independent T55 source/theorem review. Independent T55 rebuilt all88 faces via a separate forced affine recurrence and pairwise intersection enumeration, matching every saved vertex including two degenerate faces and all sums; review hash pending notification.

## Results
Eight closed-face controls, four exact elimination identities, four density/scaling controls, two correctness-face pairs, any-occurrence retention and576 direct forced-chronology checks pass. A synthetic rank-side expectation failure and correction are preserved in execution_failures.md.

All88 frozen occurrence/rank-side feasibility problems were evaluated. Negative axis:5 unique moving boundaries to0 feasible, candidate K=0 (local invariance). Positive axis:15 to8 unique boundaries,14 feasible occurrences including2 degenerate; K8.31442262194 to.178267919995 after faces and.0910712094892 after density. Density-only ablations are1.10585586207 /8.19777060784, so face restriction accounts for most of the gain. Positive elementary prefix coefficient1.61044067122; displayed CDF-normalized coefficient.148946831413 (not outward rounded). No new pair value was evaluated. Independent exact raw audit passes all88 faces and sums.

Reproduce with project Python: `python controls.py`, then `python pilot.py`, using one thread, review lock,2GB and20CPU limit. Protocol hashes pinned in manifest.json.

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
