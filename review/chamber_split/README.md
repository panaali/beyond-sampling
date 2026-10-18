# chamber_split: is recipe108's 4·T! ordering-chamber start necessary? (DRI review, 2026-10-18)

**Question.** estT/recipe108 starts from 4·T! half-chambers (14.5M at T=10). The truth label only needs the 2T(T-1) cones
{x_i > x_j > rest} × sign(x0). Is the chamber start needed, and how much of the cell growth along T does it cause?
**Method.** Code reading of `m16/approaches/estT_fix/estT.py` plus `merge_count.py` (no masses): patches `_finish`/`_cell_error`
to record leaf gate histories, then merges adjacent leaves with the same (truth cone × sign, gate history). Zoo checkpoints
were read with a whitelist-only loader (`pthload.py`). Leaf and region counts reproduce the saved estT_fix / width16_probe outputs exactly.
**Status.** done (DRI subagent; unverified).
## Results (memo verbatim)
**1. Why chambers and the cut**
- **sign(x0) is required.** All step-1 gate planes are x0 = 0 (:449-451). Without that cut the start cone contains R·1, so it is not pointed, and `from_normals` (:67-72) breaks.
- **Truth label and masses: chambers not needed.** The cells only need a constant `truth` (:451, :476-491), and `mass` (:185-224) handles any pointed cone. Each truth cone ∩ {σx0 ≥ 0} is simplicial; all 2T(T-1) of them sum to 1 − 1e-14 at T ≤ 5.
- **The cut** (:233-239) gives each half one class-changing facet `n_true`. `_ref` (:246-249) is the mean of the rays on that facet, and frozen gates take their state there (:386-390). This is a heuristic carried over from est_mech (`understand_m43/README.md`:238-240); idea (c) at :517 lists "O(T²) class regions" as untried.
- **No margin certificate exists.** The test at :403-417 checks finitely many points, while the margin through the later ReLUs is nonlinear.

**2. Split-only cells.** Script: `scratchpad/merge_count.py`, no masses. It groups leaves by (truth cone × sign, gate history), with adjacency checked. Same history means same logits, so same class. Leaf and region counts match the saved outputs exactly.

| (H,T) | recipe → merged | exact → merged |
|---|---|---|
| (8,3) 5 seeds | 1165 → 993 (−15%) | 2258 → 1656 (−27%) |
| (16,3) 5 seeds | 1946 → 1792 (−8%) | 6218 → 5092 (−18%) |
| (8,4) 5 seeds | 31378 → 20628 (−34%) | 47055 → 27535 (−41%) |
| (16,4) zoo0 | 25378 → 21219 (−16%) | 51552 → 36537 (−29%) |
| (8,5) zoo0, 4/40 cones | 17744 → 8123 (−54%) | 1 cone: 7844 → 2950 (−62%) |

The exact merged counts are exactly what an exact run from truth cones would give, since each region is convex. The recipe counts are post hoc: a real truth-cone run would freeze and resolve gates differently.

**3. What breaks**
- **Exact mode:** nothing.
- **Recipe:** `_ref`. A truth cone has T-1 class-changing facets, so one reference point freezes gates correctly near only one of them. Accuracy (ratio ≤ 0.27 at (8,4)) must be re-measured.
- **Class regions (2nd-argmax classes):** non-convex, so `Cone`/`clip` cannot represent them; they reduce to truth cones.
- **Keeping a unique facet:** top-3-ordered cones × halves × sign gives 4T(T-1)(T-2) cells (2880 at T = 10). That equals the current start at T ≤ 4 and saves 16% at (8,5).

**4. T = 10 saving**
- **Start cells:** shrink 80,640×.
- **Final cells:** the merge factor is 1.2 / 1.5 / 2.2 at T = 3 / 4 / 5 (H = 8, about ×1.35 per step). Recipe cells grow ×25 then ×18.5 per step; merged, ×19 then about ×13. Chambers explain about 10% of the log-growth.
- **Extrapolation:** about 8-23× fewer cells at H = 8 (still about 2e10 cells, ≥ 1e16 ops) and about 2-5× at H = 16. Merged cells also have more rays.

**Verdict:** chambers are not needed for correctness, only for the freezing heuristic. Removing them helps but does not change the scaling conclusion.

## Compute accounting
Under 40 CPU-seconds per run, no cone masses, so the counts are of geometry only.
## Claims
- [measured] Exact-mode cells from truth cones (lossless) = 0.59-0.82x of the current exact cells at T=3-4, and about 0.38x at (8,5) (1 cone sampled).
- [measured] Chambers serve only recipe108's freezing reference point (`_ref`), not the truth label, masses or a certificate.
- [inferred] The T=10 saving is about 8-23x cells at H=8 and 2-5x at H=16; chambers explain about 10% of the log cell growth per step.
## Alternatives not ruled out
The recipe counts are post hoc (a real truth-cone recipe would freeze differently); (8,5) is sampled; the extrapolation rests on 2-3 points.
## Failure modes
A truth-cone recipe needs a new freezing rule, since a truth cone has T-1 class-changing facets.
