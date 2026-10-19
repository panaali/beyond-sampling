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

## Addendum (pre-registered 2026-10-19 ~23:5x, before measuring): option (b), "tier 2a" cheap root bound
DRI 22:48: option (b) is a separate, labelled option. It changes outputs, so T420's auditor reviews it before any
worker uses it, and T400 decides whether it enters tiered10. It is not part of the identity gate.

**Method** (`mass_screened.py`). After the exact geometry (geom_faster, identical to geom_fast), every root simplex
sigma_j of the exact root triangulation gets the plain M5 bound
cheap_j = c0 vol_j G(d, s_min,j, t0, inf).
- s_min,j is a rigorous lower bound on |m|^2 over sigma_j: a separating hyperplane through the vertex-centroid
  direction, and >= 1/2 on the slice, as in m5t_fast.
- B = up(fsum_j cheap_j).
- **Tier 2a:** if B < CAP (1e-16, the same CAP tiered10 uses for M2), the cell is charged with min(M2, B), lb = 0.
- Otherwise the identical tier-2 path runs (mass_cell_fast).

A first exploration of a screened-evaluation variant (full M5t only on the simplices with the largest cheap bounds,
eps = 1e-3) gained only 1.1-1.4x on two cells (639, 3337). It is dropped.

**Soundness.** The root simplices tile the cell's slice polytope exactly (geom_fast and geom_faster outputs,
identical). For each sigma_j, the M5t per-simplex quantity is
m_j = c0 vol_j E_sigma[int_t0^inf z^(d-1) exp(-z^2 s(y)/2) Phi(z y_p3)^mlow dz],
with s(y) = |m(y)|^2 >= s_min,j and 0 <= Phi <= 1. So m_j <= c0 vol_j int_t0^inf z^(d-1) exp(-z^2 s_min,j/2) dz =
cheap_j. Every quantity is evaluated with m5t_fast's audited outward-rounded pieces (idot, sumsq_up, G_fac, exp_up,
c0), and the sum is a correctly rounded fsum, rounded up. So the cell mass is <= sum_j m_j <= B. min(M2, B) is a
minimum of two valid upper bounds, and lb = 0 is trivially valid.

**Measured on T400's 201-cell sizing sample (its 87 tier-2 cells; tier-1 cells are unchanged):**
1. the number of cells charged in tier 2a;
2. ub_2a >= ub_identical: how often, and by how much (both are valid bounds, so ub_2a < ub_identical is allowed but
   reported);
3. the loosening of the sample's total charge (sum of charges, new against identical);
4. the tier-2 time per cell (geom_faster + tier 2a or identical mass) against the identical path (geom_fast +
   mass_cell_fast; reference times from results_speed.json, same sample).

No pass line: this is an option for T400 and T420.
