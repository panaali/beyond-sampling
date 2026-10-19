# T427: does the safe-use rating keep working as the 2nd-argmax RNN grows in H and T?

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · OWNER_QA [4.10](../../../OWNER_QA.md#q4-10), [6.7](../../../OWNER_QA.md#q6-8)

## Question
The safe-use rating (T401 / T407, OWNER_QA 4.10) is r(t) = P(network answers with the 3rd largest | g23 >= t), computed
from the weights by the frozen graded-g12 rare-event estimator and checked by conditional Monte Carlo. On M_16,10 the
target was rated only at t <= 1.5 (T407, 3 of 6 checkable points); two DEV3 retrains were rated much further (T401).
Does the rating keep working on the T421 scale-study grid (H in {16, 24, 32, 48, 64} x T in {10, 15, 20})?

## Pre-registration (2026-10-19, before any estimate or check on a T421 model)

**Estimator (frozen).** estimator_t366 (graded g12 rule, T_hi = 6), claude/t218-tie-tail-labels @5b4a1250, with
rare.py / cont_g.py from claude/t286-cheap-frontier @620eb47b; run through T407's `est_with_flag` (rare_events/
rating_t407.py), which also returns the deterministic trust flag. Those files hard-code H = 16, T = 10, so
[walkgen.py](walkgen.py) copies each used function and changes only what H and T force (listed in its docstring:
array widths, T - 4 lows, the order-statistic law of T iid N(0, 1): T(T-1) phi (1 - Phi) Phi^(T-2), v4 = max of T - 3).
Gate before any scale run ([check_equal.py](check_equal.py)): the sha256 of estimator_t366.py 6bd6822e..., t366.py
997a1b60..., t340c.py 7e6a749d..., t354.py abed36c8..., walkfix.py 8e21b6a1... are asserted, and
`Walker(16, 10)` must reproduce T407's stored run on M_16,10 (est, flag share and C, rel. diff < 1e-12) at all 7 t.
**Gate passed before this commit:** all 7 estimates, flag shares and C (3,131,027 FPE) equal to T407's, rel. diff 0.0
([results/check_equal.json](results/check_equal.json), 327 s).
The frozen K3 (T401, t354.est_seg K = 3) is NOT rerun: T407 showed the graded rule dominates it on the rating task; K3
is reported only via T401 / T407 for M_16,10. Rows are walked in chunks of 37,440 x 160 / (T H) rows (T407's
declared memory deviation; exactly T407's chunk at (16, 10)). Cost charged as executed, in FPE = 5948 FLOPs (T196).

**t grid (model-free rule, [grid.py](grid.py)).** For each T, t_j solves P_T(g23 >= t_j) = P_10(g23 >= s_j) with
s_j = .5, 1.0, ..., 3.5 (T401's grid), so each grid point is an equally rare input condition at every T
(P(condition) = .244, .0444, 5.95e-3, 5.78e-4, 4.03e-5, 2.0e-6, 7.04e-8). T = 10: T401's grid exactly; T = 15:
.434, .880, 1.334, 1.795, 2.260, 2.729, 3.201; T = 20: .399, .815, 1.245, 1.684, 2.130, 2.582, 3.037. Same grid for all H.
P(condition) is exact (quadrature); r(t) = est E1(t) / P(condition).

**Checks ([mc.py](mc.py)).** T401's conditional-law MC (truthc.py / rating_mc.py) generalised to T: x drawn exactly from
its law given g23 >= t; hit = pick == position of the 3rd largest; stop at >= 1000 hits or 1e8 samples per point. A
point is **checkable** iff its check has >= 50 hits. Sampled tie share = hits with g12 < .1. Seeds 427000 + 1000 j +
sum(ord(tag)).

**Labels per (model, t).** rated: checkable and est / check in [1/2, 2]; unrated: checkable and outside; beyond test:
< 50 hits (estimate shown, not counted).

**Models.** T421 registry (branch claude/t421-scale-study, registry.json, sha256 of each npz asserted at load,
[t421.py](t421.py)): primary panel = cpu seeds 0, 1, 2 of every grid point marked kept (fewer if fewer are kept), taken
as points land; supplementary = the zoo panel (seeds 0-4) at (16, 10) and (32, 10) if compute allows. M_16,10 itself is
cited from T407, not rerun (beyond the equality gate).

**Primary outcome.** Per grid point (H, T): the rated fraction = rated / checkable over its primary models x 7 t. The
rating **holds** at a grid point iff the rated fraction is >= 2/3. Also per model the **rated depth**: the largest
P(condition) index j up to which every checkable point is rated. "Where rating stops working": along H at each T and
along T at each H, the first grid point where it does not hold (and whether it recovers further out).

**Trust flag.** Deterministic = the estimator's own share of E1 mass from g12 nodes < .1 (T407); sampled = the check's
tie share. Over checkable points: correlation, agreement on the >= 50% call, mean |diff|, and whether det < .5
separates rated from unrated (rated fraction among det < .5 vs det >= .5).

**F vs both bars (Directive 8).** Per checkable point, F = (est - p)^2 max(C, 1000) / (p (1 - p)) with p = the check's
joint probability and C the estimator's FPE for that model (all 7 t share one call). Reported against strict F <= 1
and lenient F <= 10, but marked informative only where p C >= 1 (plain MC at equal cost expects >= 1 event; below
that, F ~ (ratio - 1)^2 p C is small for any ratio).

**Compute.** CPU, at most 2 heavy processes, each under `ulimit -v 2500000`; helper child sessions for more models.
No contact with ARC; no Fable.

## Method
(filled after the runs)

## Status
running

## Results
(filled after the runs)

## Compute accounting
(filled after the runs)

## Claims
(filled after the runs)

## Alternatives not ruled out
(filled after the runs)

## Failure modes
(filled after the runs)
