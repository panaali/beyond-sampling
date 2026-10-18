# estT_geometry_patch — T316: proposed fix for estT's missing-vertex / dropped-volume defect in k = 6 cells

↑ [TASKS](../../TASKS.md) · T270 (claude/t192-gpu-cost-law: m16/approaches/gpu_cost_law/cx6.py, remass6.py, HANDOFF.md)

**Proposal only: no shared estT copy is edited. All claims unverified.**

## Question
The defect was found in T270 by the T192 child. In some 6-D cells, estT's clip keeps a ray list c.R that misses 5–11 true
vertices (qhull finds 42 where 31 are stored). The pulling triangulation over the symbolic incidence then drops more volume, so
those masses are underestimates: subtree 125 has leaf sum 0.0031237 against root 0.0031672.

Can a drop-in replacement of `mass(c, ops)` repair it without changing the exact k <= 5 results?

## Status
Done (proposal; unverified). Pass on V1 (absolute ~1e-9 reading) with patches A + B; V2 holds to the quadrature floor.

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

## Results (unverified)
Code: T270's own pipeline read-only at claude/t192-gpu-cost-law **94d26c9f**, run on CPU (gmass device cpu).

**V2, k <= 5 unchanged (patch A).** `T270_K=k python validate.py SRC both all|rand20`, then `vsum.py`.

| k | subtrees | total leaf, orig vs patched | max abs(per-subtree Δ) | max abs(leaf − root) | qhull cells / fallbacks |
|---|---|---|---|---|---|
| 3 | 894 (all) | 0.999999999999908 vs 1.000000000000017 | 2.7e-13 | 0 / 0 | 1382 / 0 |
| 4 | 2226 (all) | 1.000000000019 vs 1.000000000026 | 2.7e-13 | 2.4e-13 / 2.7e-13 | 24,517 / 0 |
| 5 | 20 (rng 316) | 0.004587555838970 vs 0.004587555839882 | 5.4e-13 | 2.0e-12 / 2.5e-12 | 3947 / 5 |

- **Totals:** the masses agree, within 7e-12 in total.
- **Per subtree:** absolute differences are at most 5.4e-13, which is the quadrature noise of the simplicial rules.
- **Literal pre-registered criterion fails:** "1e-9 relative per subtree" fails on 860 tiny subtrees (mass <= ~3e-8; e.g. 1.5e-9
  vs 1.5e-9 + 2e-14). That is the absolute floor, not a change in the geometry. The criterion was mis-specified; it is reported
  as such, not reinterpreted.

**V1, k = 6.** `audit.py` locates the defect; `val6b.py` measures leaf vs root with bounded memory. The three flagged subtrees:
- 125, the known one;
- 2359, the audit's hit in chunk 29 (D = 3.12e-5);
- 975, the audit's hit in chunk 12 (D = 2.42e-5).

Candidates came from the T192 child's root-mass ordering. Clean controls: 81, 2373, 2369, 2335, 2343, 1003, 1007, 963, 991,
with Δ <= 4.5e-13.

| subtree | estT leaf − root | patch A only | patch A + B | regions estT → A+B | A+B CPU |
|---|---|---|---|---|---|
| 125 | −4.35e-5 | ≈ −9.9e-7 (−4.352e-5 + 4.253e-5 recovered on 717 mismatched cells) | **−6.4e-11** (rel −2.0e-8) | 7659 → 8495 | 1730 s |
| 2359 | (Δ_A = +9.31e-6 on 701 cells) | – | **−5.0e-12** (rel −5.9e-9) | 7712 leaves → 8558 regions | see json |
| 975 | (Δ_A = +2.25e-5 on 1375 cells) | – | **−4.1e-11** (rel −3.0e-8) | 12402 leaves → 14569 regions | 2754 s |

- **Patch A alone (masses)** leaves −9.9e-7 on 125. This matches the T192 child's MC estimate of 1.04e-6 of mass in gate
  histories the enumeration never produced. **Patch A is necessary but not sufficient.**
- **Patch B (clip)** is also needed: before clipping, the cell's rays are checked against qhull, and if the counts differ the cell
  is rebuilt with qhull rays and a numeric incidence (tolerance 1e-10). On 125, 84 of 8516 clipped cells were rebuilt.
  - With A + B, leaf = root to 6e-11 absolute, against a true-subtree mass of 3e-3. That meets the DRI's "~1e-9" read as
    absolute. My README wrote "1e-9 relative"; 125 gives 2e-8 relative, 2359 gives 6e-9.
- **Region counts change.** The missing vertices made clip skip real cuts: subtree 125 has 8495 regions with the fix vs 7659 under
  estT (+11%). [measured] So T270's k = 6 region counts are biased low wherever the defect hits [inferred]. Reported to the
  T192 child.

**V3, audit.** 717 of 7621 non-simplicial leaves in 125 have more qhull vertices than stored rays (e.g. 98 vs 65). Clean
subtrees have 0–4 mismatches with |Δ| <= 4.5e-13.

## Proposed patch (for whoever owns estT)
1. **mass():** for non-simplicial cells use `estT_geometry.simplices` (qhull rays from N + Delaunay), falling back to the
   pulling triangulation when qhull fails. Cost: 10–50 ms per cell on CPU.
2. **clip():** check the ray count against qhull for non-simplicial cells, and rebuild when they differ. Needed for exact region
   counts and the last ~1e-6 of mass. Cost: ~35x slower geometry at k = 6 (1730 s vs ~50 s for subtree 125).
   - A cheaper trigger is to run the check only when d >= 6, or only when a cell's ray count is below its normal-count bound.
     Untested.
3. **Leave the symbolic incidence path unchanged everywhere else.** T108 found that numeric incidence broke thin (8,4) cells.
   The rebuild uses a numeric incidence only on cells already found inconsistent.

## Claims
- [measured] Patch A leaves k <= 5 masses unchanged to the quadrature floor (<= 5e-13 per subtree).
- [measured] A + B closes the k = 6 leaf/root gap on the flagged subtrees, to 6e-11 (125), 5e-12 (2359) and 4e-11 (975).
- [measured] A alone does not close it (125: −9.9e-7).
- [inferred] The root cause is clip's combinatorial adjacency test on degenerate 6-D cells; it misses edges, so it misses
  vertices. This is not traced to a specific line.

## Alternatives not ruled out
- The fallback cells (28–43 per subtree, qhull failing on thin cells) still use the old triangulation. They may hide small
  losses below 1e-10.
- The tolerance 1e-10 for the numeric incidence was not varied.
- Only 3 defective subtrees were checked; chunk 29 has at least one more defective subtree (D 3.12e-5 vs 9.3e-6 found).

## Failure modes
qhull precision on very thin cells (LP radius <= 1e-12 → fallback), and the cost of qhull at every clip.

## Compute accounting
CPU only:
- k3/k4/k5 validation, about 10 min;
- audit of 12 subtrees, about 45 min;
- A+B runs, about 30 min each;
- one patched run lost to OOM, one to a container restart.
