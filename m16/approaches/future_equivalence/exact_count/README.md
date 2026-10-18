# T188 exact-mode prefix census

## Question
How many exact-mode depth3 pieces at H8T4 share the same global future margin map?

## Method
Fixed four-parent pilot; two exact flags with unchanged geometry, source review first. PROTOCOL.md freezes scope; exact_common.py records stop observation, pilot.py groups exact histories/states using reviewed weight predicates.

## Status
Fixed four-parent phase complete608013fa after independent h4 source gate; saved independent audit pending. No all96 extension yet.

## Results
Exact pilot:84 pieces,42 global histories/states,48 same-truth histories/states, no distinct-history reset. All stored maps differ from exact. Zero lookahead/mass/final calls. Same-parent recipe archive had81/42/47; comparison.json retains history-set differences. Run command (outputs refuse overwrite): `ulimit -v 2500000; ulimit -t 20; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python m16/approaches/future_equivalence/exact_count/pilot.py`.

## Compute accounting
New in-script CPU .231858944, observer .067990383,129693 inherited proxy,108 finish entries,84 states; initial imports/final serialization excluded. All new exact geometry counted separately; initial charts, weights and predicates reused. Proxy and exact formation partial, setup/decode/import separately measured. Fraction/control/native costs unpriced.

## Claims
[inferred] Exact state equality implies identical global future margin functions under the independently checked fixed-weight predicates. [measured] Fixed pilot counts above; independent saved audit pending.

## Alternatives not ruled out
Chart duplication, reset histories, domain-only/label-only equivalence and floating geometry errors remain distinct.

## Failure modes
Resource caps preserve partial outputs; floating TOL is not a rigorous partition certificate. No terminal/mass conclusion follows from a prefix stop.
