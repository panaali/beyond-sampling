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
| T31 | Tiny-complex decision-facet flux | Flux primitive passes checks; final boundary-count and scale-invariance audit being integrated. Synthetic count changed after removing internal interfaces; DEV compression screen remains negative so far. |
| T42 | Independent read-only review supporting DRI T35 | Complete; [actionable report](../../../review/arc_cumulants/README.md) and minimal repros published. Exact source snapshot90608af; no author/vendor edits. |

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

**T35 author action:** [T42 review](../../../review/arc_cumulants/README.md) confirms
readout tie bugs and AUGMENT K2 input cumulant loss at snapshot90608af. Tie bugs
show zero impact on the bounded DEV samples; do not attribute propagation failure
to them without evidence. Other findings qualify partial chunks, symmetrization,
zero-variance atoms and primitive FLOP accounting. All fixes remain DRI-owned.

When the corrected walker is final, decide whether to register sci_tao_v2 for
shared evaluation and which remaining T20 ports merit work. Numerical quadrature
classification remains DRI-owned; speed alone is not a mechanistic solution.
Completed studies are registered under task-derived IDs T19-R2/T20-P1/T29/T32/T33/T34/T37-P1
  to avoid E-number races; please review/classify them. DRI resolved the E28/T42 numbering collisions in ec0b500; new IDs will be
allocated after pulling and reserved immediately on push, per updated AGENTS.

Redirect Codex priorities by editing our TASKS notes or this handoff. Until then,
Codex continues the reserved T31 tiny-complex flux diagnostic; T37-P1 is complete.
T32/T34 are complete; T20 remaining ports have been released to Ready.

## User-prioritized small ladder and fundamentals (latest steering)

Claims T47-T52 were pushed in3de8f76 before work. Five isolated Codex agents cover
H2,T3/4; H2,T5/6; H2,T7/8; H2,T9/10; and H4,T4. T52 provides independent review.
These do not overlap Claude T46, whose width is H16. Seed0 checkpoints are frozen
for available cases. Zoo lacks published T7/T9 checkpoints (standard URLs404);
T53/T54 now track reproducible checkpoint work in Backlog. No unlabelled synthetic
substitutions or heavy training. First probes are bounded to120CPU seconds per
agent, one thread, with shared CPU limits preserved.

User additionally requested persistent Feynman-style first-principles reasoning.
T55 is reserved and starts when the initial reviewer releases its slot; root will
maintain successive bounded investigation/review cycles. This is a reasoning
style, not a claim to reproduce Feynman or exemption from independent checks.
New agent must read prior sci_feynman negatives and coordinate candidate ideas
with existing owners. Claude remains DRI and owns classification/final acceptance.

T31 correction is integrated and the task closed: synthetic compression5.8x,
DEV1.43x/1.89x; independent nominal-accuracy agreement within7e-16. Counts are
geometric lower bounds, not transport-compatible merging. No global win.

## Active management after restart

User directs root to keep agents progressing and dispatch subsequent work.
T55 commits through7212005 are now integrated: analytic two-input weighted-sphere
tail, independent tests, normalized3D transport derivation, and singular-stratum
counterexample. T4 total accuracy now has an independent one-angle outer integral
route. T47-T51 workers are resuming; root takes T52 independent review.
Some unpushed pre-restart worktrees are absent and must be reconstructed from
recorded protocols/results. No lost output is treated as current evidence.
Codex numerical runs share one flock token at /workspace/work/codex-compute.lock,
one thread and2GB perprocess. Bounded120CPU-second cycles checkpoint promptly.
DRI: please resolve duplicate T47 (our H2T3/4 versus your truthprecision task).

### First recovered small-case acceptance request

H2,T3: T47's two analytic tails give full accuracy with no outer sampling.
Author nine-copy results and root12-case independent geometry review agree to
1.67e-16; constant-output tie cases give1/3. Public trained accuracy~.41.
See small_h2_34 and small_ladder_review. Arithmeticupperbound is deliberately
loose and uses unit-costspecialfunctions; fullCPU comparison reported separately
(about248k-359k vectorizedforwardequivalents, equalCPU MC RMSE~.08-.10pp).
Please classify/accept this small-case result; sharedarc_eval hardcodesT10, so
no sharedleaderboardcompletion is claimed. T47 staysactive forH2T4.
