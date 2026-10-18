# T214 mandatory-cost proof review

## Question
Do the frozen source and stated premises support the proposed mandatory-work floor?

## Method
Read pinned Q/h9, allocation/draw code and leaderboard policy; independently derive strict-interval, dense-shape and candidate-batch inequalities. Verify all nine protocol hashes. No author imports or numerical contractions.

## Status
Source/theorem gate passes e38b37b5 under the explicitly stated premises. Later arithmetic driver remains separate.

## Results
[inferred] At least one finite distinct-top2 row implies J>=3. Dense canonical floor is7040M, with6820M classic-dot alternative. Candidate count is at least50000ceil(M/50000). See source_gate.json for exact pins and qualifications.

## Compute accounting
Nine administrative file hashes and read-only source review; zero estimator/Q/RNG/native/model/reference or quota-cost contraction calls.

## Claims
[inferred] The conditional lower bound is valid for the frozen source and accounting interpretation. It is neither a complete cost nor a hardware lower bound.

## Alternatives not ruled out
A different rule5 interpretation, tied prefixes, optimized alternative implementations, larger actual candidate counts and omitted Q work remain distinct. A lower bound below a reported charge does not validate that charge.

## Failure modes
Mixing canonical and classic units, dropping the finite distinct-top2 premise, replacing binary source arithmetic by exact decimal arithmetic, or applying the policy interpretation without DRI acceptance would overstate the result.

## Independent saved result audit

Explicit assignment0a43c9f8, checker73ee656f, independent checker-source gate6b39d20a and pre-run pin60409606. All six rows/four integer controls/22 progress records pass. The checker reconstructs binary rounding through exact integer nearest-even significands, derives dense totals from dimensions, and verifies exact rational charge comparisons. It imports no author code and calls no estimator/Q/RNG. Sole review.031990211CPU,55000KiB RSS under20CPU/2097152KiB limits.

At budget1000 the conditional canonical floor39928/15 (~2661.8667FPE) exceeds binary report4503737066323967/2199023255552 (~2048.0625); gap20246544552820751/32985348833280 (~613.8042). Other five budgets are inconclusive, as are all six dense-only comparisons. Conditional classic-dot decisions agree, but are separate units. No policy-free undercharge, full cost, historical D or trained-performance conclusion is established.

Historical sole command, with existing output protection:

```sh
ulimit -v 2097152
ulimit -t 20
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
flock -n /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/t214_mandatory_cost_review/check_saved.py
```
