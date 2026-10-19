# T415: fast rigorous geometry (`geom_fast.py`)

↑ [event_cone_rigour README](README.md) · [T412_README](T412_README.md) · [TASKS](../../../TASKS.md)

Owner: Claude child session_01333jBMKr2Pn3Hfse5vLAUW. Branch `claude/t415-geom-fast` (off `claude/t400-bound-rigour`;
T400 merges). New files only; `rigour.py` and `m5t_fast.py` are not edited.

## Question
Can the exact geometry of a cell or event region be made >= 10x faster and still give identical results? The
geometry is: exact rows, irredundant rows, vertex enumeration, pulling triangulation and simplex volumes, i.e.
`rigour.mass_cell` and `m5t_fast.cell_geometry`.

Profile before any change (region 3: 142 vertices, 10,301 simplices, 33 s; line_profiler on
`m5t_fast.cell_geometry`):

| stage | time |
|---|---|
| exact fmpq 9x9 simplex-volume determinants (~2.9 ms each) | 91% |
| irredundant-row LPs | 3% |
| Poly.enumerate | 2% |
| pulling_triangulation | 2% |
| float enclosures (q2iv) | rest |

So the volumes are the bottleneck, not the enumeration.

## Pre-registration (pushed before any timing of geom_fast)
Reference: `m5t_fast.cell_geometry` at T400's head e529b8eb. It is rigour.mass_cell's geometry: exact_rows,
irredundant, Poly.start / enumerate, pulling_triangulation, exact fmpq volumes, then outward float enclosures.

Test set:
- the 55 k = 4 bench cells (27 + 28 of `a_m5.py`, norm 10);
- the 5 harvested k = 10 event regions (`data/regions_1-8-2.npz` idx 0-4, event = True);
- the first 20 k = 6 survivors of `certs/surv_d0.333_1-8-2_k6.npz` (norm = 1 / q(0.333)).

PASS requires both:
1. **0 mismatches** on every case:
   - identical exact vertex list V, in the same order;
   - identical incidence sets;
   - identical simplex list;
   - volume enclosures [vlo, vhi] equal to q2iv of the exact fmpq volume;
   - equal mass_cell_fast [lb, ub] at budget 2000 (k = 4, 6) or 4000 (regions).
2. **Speed >= 10x** for the geometry (rows to simplices and volume enclosures), one process: on the summed
   time over regions 0-4, and on region 3 alone.

Also reported: per-stage times, and the k = 4 / k = 6 speed-ups (no line).

## Method (`geom_fast.py`; tests `t415_test.py`)
`cell_geometry_fast` is `m5t_fast.cell_geometry` with three stages replaced. Each replacement returns the
reference's own output; floats only propose candidates, exact arithmetic decides.
1. **Vertices** (`enumerate_fast`, same output as `Poly.enumerate`).
   - Proposals: qhull (scipy `HalfspaceIntersection`, interior point = Chebyshev centre by LP) gives float
     vertices. Each is made exact by an exact n x n solve on its n smallest-slack rows, with an exact rank check (a
     greedy float-rank choice is the fallback). It is kept only if every row's exact slack is >= 0, which also
     gives its exact tight set.
   - Replay: the reference BFS is then replayed from the same start vertex (`Poly.start`). At a simple vertex,
     the reference's edge c (column c of B^-1) is the line on which the rows T \ {T[c]} are tight. That line meets
     Q in a segment whose only vertices of Q are its two ends, so the neighbour is the unique other known vertex
     whose exact tight set contains T \ {T[c]}.
   - Fallback: if there is no such known vertex, the reference's exact ratio test computes it. Degenerate
     vertices (|T| > n) use the reference's `Poly.edges` and ratio test.
   - So V (values and BFS order) and the incidences equal the reference's whatever qhull returns; qhull only
     removes ratio tests. Balinski's theorem (the vertex-edge graph is connected) is why following every edge
     reaches every vertex, in both the reference and here.
2. **Triangulation** (`pulling_triangulation_fast`): the reference algorithm and iteration order. Only the
   dimension test dim(G) == d - 1 on candidate facets G changes, and it is decided modulo p = 2^61 - 1:
   - affine rank mod p <= true rank, and the candidate G always has dim(G) <= d - 1. G is a proper subset of the
     d-face F, cut by a row that is not tight on all of F. So a mod-p affine rank of d - 1 proves equality.
   - n - rank_p(rows tight on all of G) < d - 1 proves dim(G) < d - 1, because the affine hull of a face is cut
     out by the rows tight on all of it.
   - The exact rank (the reference's) decides only if both tests are inconclusive. It was never needed in the
     tests (`rank_exact` = 0).
3. **Volumes** (`volumes_iv`).
   - Large-bit vertices (n x bits > 3000, i.e. the 10-D regions): the 9 x 9 determinant in arb at 128 bits (a
     rigorous ball), with difference vectors cached per apex. The float enclosure is the pair of floats around the
     ball when no float lies inside it; otherwise the exact fmpq determinant decides.
   - Small cases: the exact fmpq volume, which is cheap there.
   - Floats are taken with `q2iv_ref`, which reproduces the reference's `m5t_fast.q2iv` exactly (including its
     occasional 2-ulp enclosure when float(p)/float(q) double-rounds) using one exact integer comparison instead
     of rational loops.
- The dropped-row check is done for all vertices in one exact matrix product.
- `exact_rows`, the irredundancy LPs and `Poly.start` are the reference's (the same calls).
- `patch_rigour()` monkeypatches `rigour.Poly.enumerate` and `rigour.pulling_triangulation` for the arb
  `mass_cell`. Proposed one-line change for T400 (not made here): in `rigour.mass_cell`, use
  `geom_fast.volumes_arb(V, S, n)` for the root volumes instead of the fmpq determinants. They are arb balls,
  which the arb pipeline accepts, and they are 91% of the geometry time.

## Status
Done (2026-10-19). Every case that can run under the memory rule shows 0 mismatches and >= 10x on the 10-D
regions. Regions 1 and 2 cannot be compared, because the reference itself runs out of memory (details below).

## Results (`python t415_test.py all bench|surv`, `python t415_test.py case region i`, `python t415_test.py patch`)
| set | cases | mismatches | reference time | fast time | speed-up |
|---|---|---|---|---|---|
| bench k = 4 cells (28 distinct geometries; 55 cell x t0 mass checks) | 28 | **0** | 5.8 s | 5.8 s | 1.0x |
| k = 6 survivors 0-19 | 20 | **0** | 15.1 s | 11.2 s | 1.35x |
| region 0 (752 vertices, 329,670 simplices) | 1 | **0** | 1,182 s | 26.3 s | **45x** |
| region 3 (142 vertices, 10,301 simplices) | 1 | **0** | 33.7 s (29.9-33.7 over runs) | 2.25 s | **15x** (12-17x over runs) |
| region 4 (1,212 vertices, 1,122,736 simplices) | 1 | **0** | 2,826 s | 88.3 s | **32x** |
| regions 0 + 3 + 4 | 3 | **0** | 4,042 s | 117 s | **35x** |
| region 2 (1,890 vertices, 2,271,129 simplices) | - | not comparable | MemoryError | 200 s | - |
| region 1 (2,122 vertices) | - | not comparable | MemoryError | MemoryError | - |

- "Mismatch" means any difference in: V (exact values and order), incidences, simplex list, Mlo/Mhi, vlo/vhi,
  or the mass_cell_fast [lb, ub]. The mass is checked at budget 2000 for k = 4 and k = 6 (both t0 for the bench),
  and at 4000 for regions; for regions at or over budget the mass is the chunked outward sum, which is what
  mass_cell_fast returns without refinement.
- No fallback was ever needed: 0 qhull failures, 0 exact ratio tests, 0 exact rank decisions and 0 exact
  volume decisions in all cases.
- `patch_rigour()` (arb `rigour.mass_cell`, budget 200): identical [lb, ub] on bench 3, bench 44 and surv 0.
- Pass line as pre-registered: 0 mismatches holds on every comparable case. >= 10x holds for region 3 alone
  (15x) and for the comparable 10-D sum (35x). The literal "sum over regions 0-4" cannot be evaluated: under
  `ulimit -v 2500000` the reference (`rigour.pulling_triangulation`, which memoises every face's simplex list)
  runs out of memory on regions 1 and 2. The fast version keeps that algorithm and also fails on region 1.
- The k = 4 / k = 6 cells gain nothing: their geometry time is mostly the irredundant-row LPs, which are kept from
  the reference so that results stay identical.

## Claims
- [measured] Identical exact geometry and identical M5t mass on 51 cases; 15x to 45x on the 10-D regions that the
  reference can run under the memory rule.
- [inferred] Identity holds on every input by construction (Method): every decision is exact or proved by a
  one-sided modular rank, and floats only propose.
- [measured] Both the reference and the fast version run out of memory on the largest regions (>= 2e6 simplices)
  at 2.5 GB per process.

## Alternatives not ruled out / failure modes
- The identity argument for the neighbour lookup assumes every registered point is an exact vertex of Q. That holds
  because each comes from an exact n x n solve with an exact feasibility check, which makes it a basic feasible
  solution.
- Memory: the pulling triangulation's memo and the per-simplex arrays grow with the number of simplices.
  Region 1 (2,122 vertices) does not fit in 2.5 GB. A streaming triangulation (generate the simplices of each top
  facet, sum their bounds, discard them) would remove this limit. It is not done here.
- Timing noise is about +-15% (region 3 measured 12x to 17x across runs).

## Proposed follow-ups (for the DRI)
1. A streaming pulling triangulation plus chunked bounds, so that regions with >= 2e6 simplices fit the 2.5 GB rule.
2. T400: in `rigour.mass_cell`, use `geom_fast.patch_rigour()` and `volumes_arb` (one line) to get the same speed-up
   in the arb path.
