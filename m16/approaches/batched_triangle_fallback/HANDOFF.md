# Grouped triangle fallback handoff

Task: T116 (triangle fallback; distinct from the duplicate bulk-track ID, awaiting DRI renumbering). Claim 783ebb0 preceded implementation. Research commits: 310736d, 1ea31e0, 844964f; this handoff follows them. Only this folder changes.

The implementation leaves scalar quadrature selection intact, then groups selected positive rules by order. All ten possible orders were exercised among 9,309 fixed controls. Rule-generation metadata, bounds, node arrays and counts agree. Final values can differ by floating reduction order (maximum relative difference 2.67e-15). An initial too-small chunk-cap contract was corrected with explicit rejection and retained in controls_initial.json.

The frozen T106 64-cone raw-edge pilot preserves every saved per-cone value/statistic exactly. Partial mass is .003088193770479861, not a full accuracy estimate. The same 3,641 fallbacks consume 58,256 nodes. Five fresh complete timing calls per implementation give medians .395910251 scalar and .371893075 grouped CPU seconds (6.066% gain). This fixed local timing is not a portable speed guarantee.

All original arithmetic charges are retained; grouping adds 30,120 to the inherited conservative pilot allowance, yielding 265,021,914. This is a CPU-dispatch result, not a FLOP-saving method or a new sampling comparison. Dictionary/indexing/Python overhead is included in timing; the scalar allowance is not a new instruction-level proof. Floating roundoff, original topology, outer quadrature and omitted mass remain outside this task.

Reproduction, from this folder:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python check.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python pilot.py

Use the shared review lock, a 20 CPU-second limit and 2 GB memory limit. The pilot pins the original cache hashes and T106 source hash; it does not regenerate or tune geometry. Source paths are local research dependencies, not yet a standalone portable release.

Independent T55 source/arithmetic review requested. DRI alone may confirm/register. No full integral or 93M-node run is justified merely by this speedup. Vectorized bound selection with guarded scalar decisions, a native implementation, and tighter positive primitives remain open; this phase does not establish their failure or merit. Add a separately scoped backlog row before implementing any such follow-up.
