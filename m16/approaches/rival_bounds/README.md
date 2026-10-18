# rival_bounds — T289: joint rival-event Gaussian cone bounds (Astra r2 C1), a deductive interval estimator

↑ [TASKS](../../../TASKS.md) · review/codex_astro_review_r2 (C1, D) · [crown_tree (T279)](https://github.com/panaali/clean-neurons/tree/claude/t241-neuron-account/m16/approaches/crown_tree)

**All claims unverified.**

## Question
Can cheap shared affine bounds on the rival margins, integrated as Gaussian cone masses and combined by Bonferroni bounds,
give a valid accuracy interval [l, u] that is narrow enough to matter in the cheap regime (<= 4k FPE at (16,10))?

## Pre-registration (written 2026-10-18, before any T289 number was computed)
**Read-only reuse.**
- From T279 `crown_tree/tree.py` (branch claude/t241-neuron-account @44a3524a), copied unmodified into `vendor/` with its SHA
  recorded:
  - `roots(T)`: the 2T(T-1) bounded simplicial sectors, i.e. truth region (i, j) times the ± halves of the line R·1, label j;
  - `_relax`: the ReLU relaxation (chord upper; lower slope 0 or 1, whichever side is larger);
  - `_back` / the `crown` margin bound, used only as a tighter (expensive) reference.
- From T109 `cone_mass4.mass4_rays`: exact simplicial cone masses at d = 4.
- No new CROWN is written.

**Bounds (the C1 construction, "shared forward affine").** Per sector (vertices V, d x T, simplex weights lambda), propagate
forward, for every neuron, an affine lower and an affine upper bound in lambda. Constants fold into sum(lambda) = 1.
- z_lo = Wh+ A_lo + Wh- A_up + Wi V_t^T, and symmetrically z_up.
- The interval [min_v z_lo, max_v z_up] comes from the vertex coefficients (exact on the simplex).
- `_relax` gives h_lo = alpha z_lo and h_up = s z_up + b.
- At the readout, the rival margins m_k = (Wo_j - Wo_k) h_T get l_k = c+ A_lo + c- A_up and u_k = c+ A_up + c- A_lo. One
  propagation is shared by all 9 rivals.
- By homogeneity (no biases) these are linear bounds on the whole cone.

**Events and masses.**
- Outer error cone for rival k: O_k = S ∩ {l_k·lambda <= 0}, which contains E_k = S ∩ {m_k <= 0}. Inner cone:
  I_k = S ∩ {u_k·lambda <= 0}, contained in E_k.
- Sector error mass:
  - lower: max(max_k P(I_k), sum_k P(I_k) - sum_{k<k'} P(I_k ∩ I_k')), second-order Bonferroni;
  - upper: min(P(S), sum_k P(O_k) - sum_{tree} P(O_k ∩ O_k')), Hunter, with the spanning tree a chain in a fixed priority
    order (rivals by decreasing |Wo_j - Wo_k|, weight-derived).
- Polytope masses at d <= 4: the vertices of simplex ∩ halfspaces are enumerated, triangulated (Qhull Delaunay) and mapped to
  simplicial cones.
  - d = 3: exact Van Oosterom–Strackee solid angle.
  - d = 4: mass4_rays (stated |err| < 1e-7). The interval is widened by 1e-7 per simplex.
- Accuracy in [1 - sum_s upper_s, 1 - sum_s lower_s]; the estimate is the midpoint.

**Steps and gates (DRI / review D).**
1. **Enclosure and cost on the exact rung.** zoo (4,3) and (4,4), seed 0 (H4T4 reference 0.477013125994119; H4T3 reference =
   1e7-sample MC, diagnostic). Check l <= truth <= u for the forward bounds and the CROWN reference, and report widths. Count
   FLOPs per component (bounds, vertex enumeration, masses; multiply-add = 2). No parameter is tuned to a truth: there are no
   free parameters.
2. **Construction gate at (16,10):** the shared bounds over all 180 sectors must cost <= 1k FPE (6000 FLOPs per FPE).
   - **Prediction written now:** the forward affine step costs ~8 H^2 d = 20k FLOPs per step per sector. That is ~225k per
     sector and ~4e7 = ~6.7k FPE for 180 sectors. So the gate is expected to **fail** by ~7x.
   - If it fails: **stop before any mass work**. As a labelled diagnostic (no mass integrator at d = 10; sampling only to
     describe the bounds), report on DEV2 IDs 0 and 1 of both kernels the MC fraction of sector mass where the bounds are
     informative: P(I_k) > 0 for some k, or the outer union strictly smaller than S. This tells whether a cheaper construction
     would be worth seeking.
3. **Screen (only if 2 passes):** DEV2 IDs 0 and 1 of both kernels. Requires valid [l, u], C <= 4000 FPE and
   mean(((u-l)/2)^2)·4000/mean(min(l(1-l), u(1-u))) <= 0.8. Stop if the bounds or the denominator are vacuous.
**Alternatives written first.**
- (a) The bounds are valid but vacuous on whole sectors at T=10. T279 certified 0 of 1000 cones at (16,4). If so, the cost gate is
  moot.
- (b) The cost could pass with a different sharing (across sectors, or for steps before the first special position). That is
  untested here and would be a different construction.
