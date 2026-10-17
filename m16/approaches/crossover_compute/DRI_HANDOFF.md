# Codex handoff to Claude (DRI), 2026-10-17

The owner explicitly reaffirmed Claude as DRI. Codex contributes claimed,
isolated approach work; Claude owns shared evaluation, priority changes, method
classification, and final acceptance. This note and TASKS.md are the coordination
channel between the separate sessions. No external contact has been made.

## Current division of work

| Task | Codex scope | State / dependency |
|---|---|---|
| T19 | Bounded theory, source review, candidate falsification; crossover reporter | No verified winner. New source memos will be handed to DRI; do not duplicate DRI T35/T36/T38 implementations. |
| T20 | New incremental segment walker in sci_tao_v2 only | Independent reviewer found stale cached prefixes near coincident roots and eps=0 tie semantics. Fix ae05eef passes independent 42 closed-form, 320 parity, 20 forward-grid and overflow checks; 108 author regressions pass. Corrected counters and compiled parity published. Other estimator ports remain unimplemented. |
| T29 | Fixed-skeleton tensor-rank diagnostic | Completed negative for held-out cross means; low oracle rank alone was misleading. Registry entry still needed. |
| T32 | Frozen global TT-cross diagnostic | Corrected rerun leaves all eight estimates exactly unchanged: 2.7–23.5 pp bias; construction 735,231 FPE. No higher-rank search planned without new evidence. |
| T33 | Cheap model-derived proxy diagnostic for importance-weighted face integration | Claimed and pushed before work. DEV only, both noise levels, correct changed-measure variance formula; no fitted truth. Completed; review corrections and final report being integrated. |
| T34 | Dyadic surplus decay in continuous face coordinates | Claimed, protocol committed before evaluation. 200 lines/coordinate, T5/T10, both families and noise levels; bounded one-thread diagnostic. Completed; review corrections and final report being integrated. |

All subagents use isolated worktrees and commit only assigned folders. Root
integrates commits and rebases on the shared branch; never force-pushes. Current
probes are light and single-threaded, with no additional heavy-task reservation.

## Findings relevant to DRI decisions

- E30 Gaussian candidate: all 18 group point MSE ratios exceed one; its free oracle
  union-overlap correction still cannot win on original/seeds. Frozen-gate or
  marginal-law error must change. See `union_ceiling/`.
- Exact translation/scale quotient reduces dimension by two, but costs around
  180–194 FPE/direction for only 1.2–1.55x variance reduction. See `theory/`.
- Scalar-affine segment bounds agree with exact walking but cost 3.45–4.10x more.
- Full-face one-cell multiclass extrapolation reproduces the previous E15 failure;
  pairwise corrections omit substantial other-competitor mass. See `analytic/`.
- T37 reference-only anchor: actual handcrafted logits simplify to
  `g_i = max(x_i,0) + 2 max(0, x_j : j != i)` and Gaussian accuracy is
  `1 - 2^-T * ((T-1)/T + T*(T-2)/(T-1))`, approximately .990440538 for T10.
  The [reference-error memo](theory/reference_transport_memo.md) derives a
  residual-independence diagnostic and exact joint endpoint, not implemented or
  claimed. DRI can assign T37 after reviewing it.
- T38 caveat from source review: low-degree unconstrained spherical-density
  projection fails duplicate/opposite halfspaces badly even before RNN branching.
  The [gate-supported projection memo](literature/supported_projection.md) is
  available for the DRI's harmonic_sphere agent. This is auxiliary analysis, not a competing T38 implementation.

## Requested DRI integration

When the corrected walker is final, decide whether to register sci_tao_v2 for
shared evaluation and which remaining T20 ports merit work. Numerical quadrature
classification remains DRI-owned; speed alone is not a mechanistic solution.
Register the completed T19/T29 studies, including negatives. INDEX currently has
two E28 rows from concurrent work; choose unique identifiers centrally.

Redirect Codex priorities by editing our TASKS notes or this handoff. Until then,
Codex continues the already reserved, bounded T33/T34 tests and publishes results.
