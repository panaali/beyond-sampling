# estT_geometry_patch — T316: proposed fix for estT's missing-vertex / dropped-volume defect in k = 6 cells

↑ [TASKS](../../TASKS.md) · T270 (claude/t192-gpu-cost-law: m16/approaches/gpu_cost_law/cx6.py, remass6.py, HANDOFF.md)

**Proposal only: no shared estT copy is edited. All claims unverified.**

## Question
The defect was found in T270 by the T192 child. In some 6-D cells, estT's clip keeps a ray list c.R that misses 5–11 true
vertices (qhull finds 42 where 31 are stored). The pulling triangulation over the symbolic incidence then drops more volume, so
those masses are underestimates: subtree 125 has leaf sum 0.0031237 against root 0.0031672.

Can a drop-in replacement of `mass(c, ops)` repair it without changing the exact k <= 5 results?

## Method
`estT_geometry.py`. The cell's H-representation c.N is treated as authoritative.
- **Non-simplicial cells:**
  - rays are re-enumerated from c.N with qhull HalfspaceIntersection on the slice x·c = 1 (c = normalised sum of unit inward
    normals; interior point = Chebyshev centre by LP);
  - the slice points are triangulated by Delaunay;
  - each simplex goes to the unchanged simplicial routine (estT `orthant`, or estG's simplex buffer for the batched GPU/CPU
    masses).
- **Simplicial cells** are untouched.
- **Fallback:** if qhull fails (LP radius <= 1e-12 or a qhull error), the original routine runs and the failure is counted.

This is the same construction as T270's remass6.py `qhull_simplices`, written by the T192 child and reused with
coordination. The patch only packages it at the estT level. `apply_estT(B)` / `apply_estG(estG, B)` monkeypatch a loaded estT
module for testing.

**Scope note:** clip's ray list stays as it is, and the patch repairs masses only. A missing ray could also make clip treat a cut
cell as uncut (a wrong gate verdict, so a wrong region label, mass conserved). That is audited separately below, not fixed.

## Pre-registration (written 2026-10-18, before any T316 validation number)
**Drop-in validation**, CPU (gmass device = cpu), using T270's own code read-only at claude/t192-gpu-cost-law.
- (V1) **k = 6:** subtree 125 and two other flagged subtrees (positions from the T192 child, or chosen by T270's per-chunk
  deficit list before looking at their masses). With the patch, leaf sum = root mass (roots.npz) within 1e-9 relative.
  Without it, the estT shortfall is reproduced.
- (V2) **k <= 5 unchanged:**
  - k = 3 and k = 4: every subtree; k = 5: 20 subtrees (positions drawn with rng 316);
  - original vs patched leaf sums agree within 1e-9 relative (per subtree and in total), and both match their roots.
  - The roots for k < 6 are computed by both methods.
- (V3) **Clip audit (diagnostic):** on subtree 125's leaves, count cells whose qhull vertex count exceeds the stored ray count.
  Also count any split decision at the last step that changes when s = R·n uses qhull rays instead of stored rays.
**Pass:** V1 and V2 both hold. V3 is reported only. Cap 2 h.
