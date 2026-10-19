# T420: independent audit of geom_fast (T415)

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [event_cone_rigour](../event_cone_rigour/README.md) · [T415_README](../event_cone_rigour/T415_README.md)

Owner: Claude child session_016iZrbyA493LMxtd9exR6aa (T420). Branch `claude/t420-geom-audit` (off
`claude/t400-bound-rigour` 7cb41899). Audited code: `event_cone_rigour/geom_fast.py` at 7cb41899 (T415 c2b0bf02 as
merged). Nothing outside this folder is edited.

## Question
Is the geometry that geom_fast feeds the certified T416 charge right? Concretely, for every cell: (a) the exact
vertex list, BFS order and incidences equal the reference `rigour.Poly.enumerate`; (b) the simplex list equals
`rigour.pulling_triangulation` and is a true triangulation of the cell; (c) every float volume enclosure
[vlo, vhi] contains the exact simplex volume (and equals the reference's).

## Pre-registration (pushed before any comparison run)
**Phase A, code review.** Line-by-line review of the exactness argument against the code: qhull proposals, exact
solve and feasibility, the neighbour lookup (Balinski closure) and the exact ratio-test fallback, the modular
face-dimension tests and their exact fallback, the arb volume enclosures and their exact fallback, and the
`cell_geometry_fast` / `patch_rigour` wrappers. Output: a gap list, each gap marked soundness / identity / robustness /
docs.

**Phase B, comparison on real cells.** Cell set (fixed now):
- B1: 200 cells drawn uniformly (numpy seed 420) from the tier-2 cells (the ones charged with geom_fast geometry)
  of every `results_t10_surv_d0.333_1-8-2_k10_*.json` committed on `claude/t400-bound-rigour` at 7cb41899, survivor
  files `surv_d0.333_1-8-2_k10_s{0,1}.npz` (T373 branch), t0 2.5, norm 1/q(1/3), event = True, k = 10. If fewer than
  200 tier-2 cells exist, all of them plus random survivors (seed 420) up to 200.
- B2: T415's 10-D regions: `data/regions_1-8-2.npz` (regenerated with `harvest.py 1,8,2 200000`) idx 0-5 plus 10 more
  regions drawn with seed 420; event = True, k = 10, norm 10.

Checks per cell:
1. **Reference identity** (where `m5t_fast.cell_geometry` finishes under `ulimit -v 2500000` within 2 CPU-h):
   identical V (values and order), incidences, simplex list S, Mlo/Mhi, vlo/vhi.
2. **Independent exact geometry** (every cell, different code):
   - vertex set of the FULL row system (before the float irredundancy step) by cddlib in GMP rational arithmetic
     (pycddlib `cdd.gmp`) equals set(V); incidences recomputed by exact slacks equal inc;
   - total exact volume Σ_S |det| / n! (my own integer-determinant code) equals the exact polytope volume from an
     independent algorithm (Lasserre-type exact recursion over the face lattice built from cdd's incidences,
     apex = the largest vertex index instead of the smallest);
   - every simplex: exact volume > 0 and vlo <= exact <= vhi (exact rational comparison);
   - where S has <= 300,000 simplices: ridge check (every ridge either lies in a facet of the cell and in exactly one
     simplex, or is in exactly two simplices whose apexes lie strictly on opposite sides). With the volume equality
     this proves S is a triangulation (constant covering degree across interior ridges, total volume = vol(Q)).

**Phase C, adversarial.** Synthetic and perturbed polytopes, comparing `enumerate_fast`, `pulling_triangulation_fast`
and `volumes_iv` against the reference functions plus the independent checks above:
non-simple vertices (cross-polytopes, pyramids, random 0/1 polytopes, cubes cut through vertices), near-parallel rows
(rows differing by 1e-12 .. 1e-30 relative), tiny facets (a vertex cut off at distance 1e-15 .. 1e-40), huge and
mixed coordinates (scales 1e-20 .. 1e20, denominators up to 2^200), coordinates engineered so that 2^61 - 1 divides
denominators or difference minors (forcing the modular tests to fail or under-report rank), and qhull sabotaged
(proposals dropped at random, shifted, or replaced by noise) to force the ratio-test fallback. Dimensions 2-9.

**Pass line:** 0 mismatches in B and C (identity where the reference runs, independent checks everywhere) and a
written gap list; or a counterexample (any cell or input where geom_fast's output differs from the reference or
fails an independent check), reported to the author with a reproducer.

## Method
(filled in after the runs)

## Status
Pre-registered, running.
