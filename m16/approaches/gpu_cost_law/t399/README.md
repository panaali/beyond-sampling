# t399: batched GPU bound engine for the certified rare-event branch and bound (T399, helper to T398)

↑ [gpu_cost_law](../README.md) · interface: `m16/approaches/event_cone/INTERFACE.md` on branch `claude/t373-event-cone`

**Question.** Can a batched GPU bound engine prune T398's prefix cells (CROWN-style, or LP, plus ReLU splitting) fast
enough to run the branch and bound for all 720 triples? Kill-first deliverable, from the T398 lead: on the benchmark
nodes, the fraction pruned vs splitting budget, plus nodes/s. GO only if >= 90% of the benchmark's non-event nodes
prune within a budget that is affordable at ~1e6-1e7 nodes.

**Status.** Kill-first measurement and GPU dry run DONE (2026-10-19); production decision with the T398 lead. Owner: Claude session_016BGedpds19gq95yjWmDMn5. Lead: session_01LhvQ3Abu3KY6KE3AELyQUg
(T398). Rigour: session_01GrNYV8bYV4JD7z4A4KTLve (T400). Cap 40 Colab units; ~0.3 spent.

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
**Kill-first, exact-LP BaB at DELTA = 1/3** (the lane's normalisation for t0 = 2.5; `DELTA=0.3333333333333333 python
bab_cpu.py bench_182_k4.npz bab_d33_X.json 256 IDX..`; data/t399/bab_d33_{a,b}.json). 24 nodes with probe_best < 0
and is_event = 0, stratified by relax_margin, budget 256 leaves:

| outcome | nodes |
|---|---|
| closed at the root (11) | 191, 120, 32, 163, 197, 51, 119, 139, 256, 15, 31 |
| closed after splitting (5) | 222 (18 leaves), 84 (26), 69 (55), 100 (58), 206 (69) |
| exact witness at gap23 / ‖x‖∞ = DELTA, i.e. a boundary event (6) | 110, 135, 219, 117, 162, 235 (margins +.008 to +.077) |
| exact witness inside P (1) | 27: gap23 / ‖x‖∞ = .397, margin +1.28 (probe_best -0.01) |
| censored at 256 leaves (1) | 241 (94 open leaves, worst open margin +0.21) |

Every witness is forward-verified in float64 (`verify_leaf.py`: argmax = p3, ordering, G x >= 0). The probe > 0
controls 3 and 88 stay open (soundness). The fraction closed vs leaf budget, over 24 nodes, is .46 at 1 leaf,
.54 at 32, .63 at 64 and .67 at 128 or 256. Over the 17 nodes without a witness it is .65, .76, .88 and .94.
The T398 lead's decision is that boundary witnesses are charged, not pruned, so the relevant figure is 94% at
<= 128 leaves. At DELTA = 0.1 (first run, 14 nodes, data/t399/bab_d01_*.json) only 2 of 14 closed, 4 had boundary
witnesses (node 120: margin +0.32 at gap23 = 0.1), and 8 were censored.

**GPU dry run** (T4, float64 IPM, root bounds without splitting; `q_t399bench2.sh` -> `gpu_bench.py`, `gpu_bench.json`):
- Agreement with T390's exact LP (local benchmark, all 882 children of (1,8,2) at k = 1..4, DELTA = 0.1): 533 / 533
  identical prune decisions, 0 GPU-only, 0 T390-only, 0 of 58 event prefixes pruned.
- Versus bab_cpu's exact root margins (26 nodes, DELTA = 1/3): GPU >= exact on every node (gap 0 to 0.05). Two
  exactly-empty nodes (163, 256) get finite GPU bounds (-0.58, -1.70), which still prune.
- bench_182_k4 at DELTA = 1/3: 113 / 273 pruned at the root, or 125 with event-aware tightening.
- 10 of the 47 probe > 0 nodes are pruned. Exact HiGHS at DELTA = 1/3 also closes exactly 10 of them at the root
  (root_probepos_d33.json), so their probe points have gap23 / ‖x‖∞ < 1/3 (outside P), not a soundness failure.
- Throughput: 2.65 nodes/s at k = 4 (root bounds, 193 LPs per node) on a T4; HiGHS on one CPU core does ~1.2
  nodes/s. Event-aware tightening doubles the cost (1.33 nodes/s).

**Projection [inferred]:** the IPM is fp64 and compute-bound (batched 107 x 107 Cholesky plus K^T D K). A100 fp64
is ~38x a T4, so ~100 nodes/s or ~1 leaf every 10 ms, which is ~5.3 units/h. The 24 bench nodes average 44 leaves (1 at the root,
18-69 when splitting closes them, 29-161 to a witness, 256 censored). At the T398 projection of ~5e4 step-10 nodes for
(1,8,2) (2.1e6 summed over its 41-triple sample) that is ~2e6 leaves, ~6 A100-h, ~30 units, per triple, i.e. ~4,000 A100-h
for 720 triples: not affordable. The T398 lead agrees (08:46): no full-LP GPU pruner; T399 is parked at ~0.3 units
pending a yes/no on the CROWN-intermediate + LP-leaf design. GPU only pays if the leaf
cost falls ~100x, e.g. by CROWN-style back-substitution for intermediate bounds with LP only at the final check,
or by re-using the parent's duals as warm starts. That redesign is not done.

## Compute accounting
CPU: bab_cpu ~0.8 s per leaf (192 + 1 HiGHS LPs at k = 4), 2 processes, ~1 h at DELTA = 0.1 (partial) and ~45 min at
DELTA = 1/3. Colab: ~0.3 units (T4, 15 min dry run on the T352 runtime, stopped). 0.3 of the 40-unit cap.

## Claims
- [measured] With exact LP bounds in every leaf and best-first splitting, 16/17 non-witness bench nodes close
  within 69 leaves at DELTA = 1/3; 7/24 probe-negative nodes contain exact event witnesses (6 at the DELTA boundary).
- [measured] The hit-and-run probe misses events: node 27 (probe -0.01) has a witness well inside P.
- [measured] The GPU IPM engine is sound on every check run and reproduces T390's prune decisions exactly.
- [inferred] At the measured T4 rate and an A100 extrapolation, a full-LP-per-leaf GPU pruner is too slow for 720
  triples; GPU production needs a cheaper per-leaf bound.

## Alternatives not ruled out
- Other splitting rules (input splits, the neuron with the largest LP dual, re-splitting by margin gradient) can need
  far fewer leaves than the fixed earliest-step rule.
- Tighter relaxations than the triangle (multi-neuron / PRIMA-style cuts, event-aware tightening in every leaf)
  are not tried.

## Failure modes
- Witness leaves lie on cone boundaries (ties x_l = x_p3); the margin is strictly positive there, so nearby interior
  points are events too, but the witness itself is a boundary point.
- The IPM certificate is only as tight as the IPM's convergence (max gap 4.9e-3 vs HiGHS on the test set).
