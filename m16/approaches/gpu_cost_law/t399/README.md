# t399: batched GPU bound engine for the certified rare-event branch and bound (T399, helper to T398)

↑ [gpu_cost_law](../README.md) · interface: `m16/approaches/event_cone/INTERFACE.md` on branch `claude/t373-event-cone`

**Question.** Can a batched GPU bound engine prune T398's prefix cells (CROWN-style, or LP, plus ReLU splitting) fast
enough to run the branch and bound for all 720 triples? Kill-first deliverable, from the T398 lead: on the benchmark
nodes, the fraction pruned vs splitting budget, plus nodes/s. GO only if >= 90% of the benchmark's non-event nodes
prune within a budget that is affordable at ~1e6-1e7 nodes.

**Status.** RUNNING (2026-10-19). Owner: Claude session_016BGedpds19gq95yjWmDMn5. Lead: session_01LhvQ3Abu3KY6KE3AELyQUg
(T398). Rigour: session_01GrNYV8bYV4JD7z4A4KTLve (T400). Cap 40 Colab units; spent 0 so far (all work on CPU).

## Method
- `bab_cpu.py`: the reference measurement. Exact LPs (HiGHS, through relax2.py's relaxation: triangle, phase fixes,
  event rows). Best-first ReLU splitting per bench node. In every leaf all pre-activation bounds are re-solved by LP,
  and the leaf closes iff its LP with the event is infeasible. The open leaf with the largest relaxed margin is
  split next, on the earliest-step unstable neuron with the largest triangle area. A popped leaf with NO unstable
  neuron is exact, and if open it is a real event witness. So each node ends as closed (prunable), witness (contains
  events, never prunable), or censored at the leaf budget. Any GPU implementation of the same relaxation and
  splitting rule can at best match these LP bounds.
- `verify_leaf.py`: replays one node to its exact leaf, takes the LP maximiser of the margin and forward-evaluates the
  network in float64 (argmax, ordering, gap, G x >= 0).
- `lpgpu.py`: batched GPU LP engine for the same relaxation (one LP shape for bounds, event tightening and margin; a
  margin variable t makes the event rows optional). It has two solvers:
  - PDHG, float32. Dropped: too slow to converge on these LPs.
  - Mehrotra interior point, float64, batched Cholesky.

  Every value is certified by weak duality in float64 from the returned duals y >= 0, so it is valid at any solver
  accuracy. T400 re-checks y exactly.
- `mkbench.py`: local benchmark of ALL T390 children of (1,8,2) to k = 4 with T390's stage labels (for agreement and
  soundness checks); `crown.py`: first CROWN prototype (superseded; its subgradient dual was too weak).

## Results ([measured], unverified)
(in progress)

## Compute accounting
CPU only so far: bab_cpu at ~0.8 s per leaf (192 + 1 HiGHS LPs at k = 4), 2 processes.

## Claims
(in progress)

## Alternatives not ruled out
- Other splitting rules (input splits, the neuron with the largest LP dual, re-splitting by margin gradient) can need
  far fewer leaves than the fixed earliest-step rule.
- Tighter relaxations than the triangle (multi-neuron / PRIMA-style cuts, event-aware tightening in every leaf)
  are not tried.

## Failure modes
- Witness leaves lie on cone boundaries (ties x_l = x_p3); the margin is strictly positive there, so nearby interior
  points are events too, but the witness itself is a boundary point.
- The IPM certificate is only as tight as the IPM's convergence (max gap 4.9e-3 vs HiGHS on the test set).
