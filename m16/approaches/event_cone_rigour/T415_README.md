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
