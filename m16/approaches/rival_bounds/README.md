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

## Status
**Stopped at gate 2** (pre-registered stop): negative (unverified) for this relaxation configuration. Stage 3 not run.

## Results (unverified)
**Step 1, exact rung** (`python m16/approaches/rival_bounds/step1.py` -> step1.json; checks `check1.py`).
- The enclosures are valid: 0 bound violations on 1.9M sampled rival points.
- The polytope masses agree with MC within 1σ (6 random d = 3, 4 cuts), and the sector masses sum to 1.

| rung | bounds | [l, u] | width | truth | bound FLOPs | mass FLOPs | median (u-l)/median abs(m) on sampled points |
|---|---|---|---|---|---|---|---|
| H4T3 s0 | forward | [0.000002, 1.0] | 1.000 | 0.9846 (MC 1e7) | 22.0k | 49.6k | 10.6 |
| H4T3 s0 | CROWN (T279) | [0.0155, 1.0] | 0.984 | | 31.6k | 50.0k | 5.8 |
| H4T4 s0 | forward | [-0.00002, 1.00002] | 1.000 | 0.477013 (exact) | 78.4k | 861k | 514 |
| H4T4 s0 | CROWN (T279) | [-0.00002, 1.00002] | 1.000 | | 129k | 881k | 229 |

- No inner error cone has positive mass (no certified error anywhere).
- The only information is a little certified-correct mass at H4T3: 2 sectors (forward) or 6 sectors (CROWN) of 12.
- Rival-margin bounds are informative (lower > 0) at only 0.4–8% of rival points. Boole needs all rivals at once, so the sector
  intervals stay ~[0, P(S)].

**Gate 2 at (16,10)** (`python m16/approaches/rival_bounds/gate2.py /tmp/claude-0/dev2/weights.npz` -> gate2.json)

| DEV2 model | bounds | cost (FPE, 180 sectors) | gate (<= 1k) | MC fraction with an inner cone | certified correct | median (u-l)/median abs(m) |
|---|---|---|---|---|---|---|
| fused_s0, fused_s1, torch_s0, torch_s1 | forward (shared) | **7,056** | **fail** | 0 | 0 | 0.8e6 – 2.7e6 |
| same | CROWN (T279) | 32,247 | fail | 0 | 0 | 3.1e3 – 6.7e3 |

## Claims
- [measured] The shared forward construction costs 7,056 FPE at (16,10). That is 7x the 1k construction gate; the
  pre-registered prediction was ~6.7k. So the pre-registered rule stops T289 before mass work.
- [measured] The bounds are also vacuous at T=10: their width is ~1e6x (forward) and ~4e3x (CROWN) the typical margin on the
  root sectors. No sampled point has a certified error or a certified-correct verdict.
  - So even at 7–32x the cost cap, this configuration would return [0, 1].
  - The cost failure is not the binding one.
- [measured] Already at H4T4 both bounds are vacuous (interval width 1.000). The width/margin ratio grows ~50x per step
  (T3 -> T4: 10.6 -> 514 for forward, 5.8 -> 229 for CROWN). This is the E19/T279 amplification.
- [inferred] Any affine relaxation over a whole root sector, shared or per rival, fails on this RNN. Only sector splitting
  could shrink the relaxation error, and T279 shows that at (16,4) 1e5 cones still certify only 65% of the mass at 2.8e10 FLOPs.

## Alternatives not ruled out (exhaustion checklist)
Tried:
- shared forward affine bounds (DeepPoly-style chord / adaptive lower);
- T279 CROWN as the tight reference;
- Boole/Hunter upper and second-order Bonferroni lower;
- exact masses at d <= 4.

Not tried:
- optimised lower slopes (alpha-CROWN);
- tighter multi-neuron (PRIMA) relaxations;
- input splitting before bounding (this becomes T279's tree);
- bounds in other coordinates (e.g. order-statistic gaps rather than the lambda simplex);
- certified masses at d = 10 (moot while the bounds are vacuous).

The amplification factor (~1e6 at T=10 for forward) is far beyond what a few-x slope tightening can recover, so these are
unlikely to change the verdict [inferred]. Independent re-check needed per Rule 5 before "dead (verified)".

## Failure modes
Relaxation error compounds multiplicatively through 10 recurrent steps of ~16 unstable neurons. Over a whole sector, every
neuron is unstable, so chord relaxations compound.

## Compute accounting
- Step 1: 4 runs, < 2 CPU-min (masses dominate: 0.86M FLOPs at H4T4).
- Gate 2: 4 models x 2 bound types x 180 sectors (forward 4.2e7 FLOPs, CROWN 1.9e8 FLOPs per model), plus 2,000 diagnostic
  sample points per sector. About 3 CPU-min.
- Vendored T279 tree.py SHA256 5d177abff957a14a1452ec769bfc955d56961d2fc33345a158600998fe0d9782 (44a3524a, unmodified).
