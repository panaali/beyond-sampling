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
| T29 | Fixed-skeleton tensor-rank diagnostic | Completed negative for held-out cross means; low oracle rank alone was misleading. Registered as T29; DRI acceptance remains pending. |
| T32 | Frozen global TT-cross diagnostic | Corrected rerun leaves all eight estimates exactly unchanged: 2.7–23.5 pp bias; construction 735,231 FPE. No higher-rank search planned without new evidence. |
| T33 | Cheap model-derived proxy diagnostic for importance-weighted face integration | Claimed and pushed before work. DEV only, both noise levels, correct changed-measure variance formula; no fitted truth. Complete; independent review passed. Optimistic efficiency versus ordinary sampling .519x mild/.347x harsh; no transport built. |
| T34 | Dyadic surplus decay in continuous face coordinates | Complete, negative screen; independent summary review integrated and published. |
| T37 | Reference/error independence diagnostic | Complete, independently reviewed negative: all eight laws miss by 22.56–48.68 pp. Gaussian/rank follow-ups stopped. Other variants released to Ready. |
| T31 | Tiny-complex decision-facet flux | Claimed/pushed330849b. Exact geometric diagnostic complements DRI T38; no harmonic-projection implementation or global M16 enumeration. |

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
  residual-independence diagnostic and exact joint endpoint. T37 was subsequently
  claimed/pushed from the Ready backlog (740378d) for the bounded diagnostic.
  **New mathematical obstruction:** normalization by sqrt(M²+S²) gives infinite
  residual second moments near M=S=0 for this checkpoint. That Gaussian-moment
  variant is withdrawn; independent reviewer reproduced the exact rational witness
  and proof (two total deterministic rational forwards including replication).
  Finite-moment alternatives frozen in protocol58778ab also failed decisively;
  all eight laws miss by 22.56–48.68 pp. See T37-P1 registry and README.
- T38 caveat from source review: low-degree unconstrained spherical-density
  projection fails duplicate/opposite halfspaces badly even before RNN branching.
  The [gate-supported projection memo](literature/supported_projection.md) is
  available for the DRI's harmonic_sphere agent. This is auxiliary analysis, not a competing T38 implementation.

## Requested DRI integration

When the corrected walker is final, decide whether to register sci_tao_v2 for
shared evaluation and which remaining T20 ports merit work. Numerical quadrature
classification remains DRI-owned; speed alone is not a mechanistic solution.
Completed studies are registered under task-derived IDs T19-R2/T20-P1/T29/T32/T33/T34/T37-P1
  to avoid E-number races; please review/classify them. INDEX currently has
two E28 rows from concurrent work; choose unique identifiers centrally.

Redirect Codex priorities by editing our TASKS notes or this handoff. Until then,
Codex continues the reserved T31 tiny-complex flux diagnostic; T37-P1 is complete.
T32/T34 are complete; T20 remaining ports have been released to Ready.
