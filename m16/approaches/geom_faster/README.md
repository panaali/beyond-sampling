# T452: faster tier-2 (M5t) cells for the (9,3,2) charge (`geom_faster/`)

↑ [event_cone_rigour T415_README](../event_cone_rigour/T415_README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md)

Owner: Claude child session_01333jBMKr2Pn3Hfse5vLAUW (DRI session_01CD3g9Cvc3qk3pM11eyXXzj). Branch
`claude/t452-geom-faster` (off claude/t400-bound-rigour 405445d1). New files in this folder only; nothing in
event_cone_rigour/ is edited.

## Question
Can the per-cell cost of tier 2 (`geom_fast.cell_geometry_fast` + `m5t_fast.mass_cell_fast`, budget 4000, as in
`tiered10.py`) on (9,3,2) survivors be cut 3-10x? The outputs must stay bitwise identical to the audited code.

## Profile before any change (`python prof.py`)
Sample: the tier-2 cells of T400's sizing sample (results_t10sample_..._9-3-2_..._step71.json, 87 cells). First 63
cells (one more ran out of memory in the triangulation):

| part | time | share |
|---|---|---|
| geometry | 527 s | 52% |
| — volumes_iv (arb 9 x 9 determinants) | 377 s | |
| — irredundant-row LPs | 52 s | |
| — pulling triangulation | 28 s | |
| — enumeration | 19 s | |
| mass_cell_fast | 482 s | 48% |

- The time is dominated by a few cells with 1e6+ root simplices. Their budget (4000) is exceeded at the root, so
  the mass step evaluates every root simplex (~55 us each).
- Median cell: geometry 1.9 s (mostly the irredundancy LPs, ~0.7 s), mass 0.5 s.
- So geometry alone can give at most ~2x per tier-2 cell. A 3-10x per-cell cut also needs the mass step, which
  must stay bitwise identical to the audited m5t_fast.

## Pre-registration (pushed before any timing of new code)
Reference: `geom_fast.cell_geometry_fast` and `m5t_fast.mass_cell_fast` at claude/t400-bound-rigour 405445d1
(T420-audited geom_fast; T417-audited m5t_fast).

1. **Gate (identity).** On >= 1e4 cells sampled from `surv_dfs_d0.333_9-3-2_k4to10_s0of6.npz` (every k-th cell,
   all tiers' geometry), the new geometry returns bitwise identical outputs on every cell:
   - V and incidences;
   - the simplex list S;
   - Mlo / Mhi;
   - vlo / vhi.
   On the 87-cell tier-2 sample, the tier-2 [lb, ub] is bitwise identical. Cells where the reference fails (e.g.
   MemoryError) are reported separately and are not counted as matches.
2. **Speed.** Measured on the 87-cell tier-2 sample, one process: (a) geometry time and (b) total tier-2 time
   (geometry + mass), summed over the sample, new against reference.
   - PASS for T452: (b) >= 3x.
   - If (b) < 3x, the report states the floor that the unchanged parts impose (with the profile), and (a) is
     reported against a secondary line of >= 3x.

## Plan
1. **Volumes.** Use the staircase identity of the pulling triangulation. Simplex (v_0, ..., v_n) is built from
   facet rows r_1..r_n, where r_l is tight on v_l..v_n and not on v_{l-1}. Then A_R [v_0 - v_n, ..., v_{n-1} - v_n]
   is triangular with diagonal slack_{r_l}(v_{l-1}), so

   vol = prod_l slack_{r_l}(v_{l-1}) / (n! |det A_R|).

   The exact slacks are already known, and |det A_R| is cached per row set. They are evaluated in double-double
   with a rigorous error bound; the exact fmpq volume decides whenever the enclosure contains a float. The float
   enclosures are then identical.
2. Then, by the profile: the triangulation and enumeration (Python), and bitwise-identical speed-ups of the mass
   step.
