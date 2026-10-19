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
