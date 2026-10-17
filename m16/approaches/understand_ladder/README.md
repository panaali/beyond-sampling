# T73 / T88 / T93: climbing the understanding ladder (fibre-traced error wedges)

↑ [Approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · [T63/T75 (M_4,3)](../understand_m43/README.md) · [ARC protocol (T36)](../arc_m43_repro/README.md)

## Question

M_4,3's accuracy was computed from its mechanism and beat sampling by about 50x in MSE (E36, T63; T75 made it
generic across seeds). Does the same recipe carry over to the next AlgZoo 2nd-argmax rungs? The recipe is: find
where each neuron switches, read which neurons are active on each true decision boundary x_i = x_j, and treat the
error as thin wedges with exact Gaussian masses. The rungs follow the owner's ladder protocol (TASKS.md): change one
axis by one step, and a rung passes only if one frozen recipe beats sampling on all 5 zoo seeds and their noisy
copies. A second question is whether its cost scales with neurons, steps and boundary families rather than with
the number of linear regions.

**Short answer.**
- **Rungs (4,3) re-run, (5,3) (T88) and (6,3) (T93) pass** with one frozen deductive recipe (F2L2). Every zoo seed
  passes individually, as does every noisy group (ARC's absolute σ = 0.05, and the leaderboard's relative σ =
  0.0025-0.02). The worst single model has MSE ratio 0.15 (on M_4,3). On H ≥ 5 the estimate is exact to rounding
  on most models. M_8,3 also passes, but it is not a rung yet: H = 7 is missing from the zoo.
- **The cost does not scale better than the region count.** At T = 3 the recipe uses about 4.5-5k ops per linear
  region at every H, which is 10-20x plain exact enumeration. The number of traces grows like H, while regions grow
  like H^1.8. Each trace, however, checks all T·H gates.
- An exploratory T = 4 version exists (fibre4.py) and is accurate on M_8,4 seed 1, but it costs 2.8·10^8 ops. Its
  2-D boundary-face cells (2.5-3.5k) outnumber the 4-D gate patterns seen in 2·10^6 samples (1.4-2.0k), so at
  this size the boundary is not "smaller" than the volume.

## What the networks do and why they err (plain language, T = 3)

Same skeleton as M_4,3 (T63), but without its mirror-pair structure. Every quantity below is printed from the
weights by `python mechanism3.py H seed`.

1. **Step 1 (store x0).** The sign of x0 picks which neurons hold |x0|: the neurons with sign(Wi_k) = sign(x0).
   M_5,3 seed 1 uses {0,3} for x0 > 0 and {1,2,4} for x0 < 0. M_6,3 seed 3 uses {0,4} and {1,2,3,5}.
2. **Step 2 (compare x1 with x0).** Neuron k switches on the plane x1 = t_k x0, with t = -(Wh diag(g1) Wi)_k / Wi_k.
   In the well-trained M_5,3 seed 1, three of the five thresholds sit at t ≈ 1 (0.99, 1.00/1.01, 1.04/1.19): these
   are "x1 > x0?" detectors. The rest sit at t ≈ -4.9, i.e. they are always on unless x1 is very negative. The
   undertrained M_5,3 seed 0 (accuracy 0.942) stores x0 > 0 in a **single** neuron, and its thresholds lie far
   from 1 (0.01, 0.13, 0.65, 1.0, 1.81).
3. **Step 3 (compare x2).** Each neuron is a linear threshold on x2 against a combination of (x0, x1). The
   combination is fixed inside each step-2 strip and given in closed form (A3 = Wh diag(g2) A2 + Wi e2ᵀ).
4. **On the true boundaries 1-5 neurons are on.** M_4,3 seed 0 used a 2-neuron ratio readout. The larger models do
   not: along each boundary arc, the number of final-step active neurons runs through 0-1 near the poles x0=x1=x2,
   then 2-4 (M_6,3) or 2-5 (M_5,3) in the middle (`mechanism3.py` prints it per arc segment). The readout is
   therefore linear in the 2-4 active neurons, with no special pair structure. This is why T63's tier B does not
   transfer and why the generic tiers are needed.
5. **Why it errs.** The errors are thin wedges attached to the true boundaries. MC check (`diag_local.py`): on the
   zoo M_6,3, 99% of misclassified mass lies within 0.07-0.41° of a boundary. On ARC noisy copies the median angle
   is 1-3° and the 99th percentile is 5-15°.

   The two failures of the simple picture are in the undertrained seeds (M_5,3 seed 0 and M_4,3 seeds 1-4):
   - (a) a wedge that starts slightly *off* the boundary, separated from it by a thin correct sliver. On M_5,3
     seed 0 it is near x0 ≈ x1 > x2, starting at u ≈ 0.01 on the fibre;
   - (b) a wedge from the neighbouring boundary that spills across the equidistance plane (the clamp) into this
     family's fibres.

**Verification of the mechanism** [measured]. The predicted gate switches on every boundary arc, from closed-form
roots, are checked against the real network: forward passes just inside each predicted segment end show the
predicted pattern at 136/136 (M_5,3 seed 1), 140/140 (seed 0) and 148/148 (M_6,3 seed 3) segment ends. Zero-length
segments at x0 = 0 exactly are measure zero and are skipped. Independently, the FX tier is the same fibre geometry
traced to the clamp. It reproduces exact enumeration (`arc_m43_repro/exact.py`, a different partition of the
sphere) to within 1e-14 on all 628 scored models. That validates the fibre partition and the closed-form wedge
masses.

## Method

Deductive: no random inputs. Every coefficient is a formula of (Wi, Wh, Wo).

**Geometry** (`fiber3.py`, model-independent, exact). Each true boundary circle x_i = x_j is split at x_k = x_i = x_j
into two arcs (x_k above or below). Every point x off the circles is written as x ∝ p(θ) + u n′, where n′ = ±(e_i - e_j)/√2
and u = tan φ ≥ 0. Here p(θ) lies on the boundary of x's ordering chamber that is nearer, of the two boundaries where
the class changes. The fibre ends at the equidistance plane (the "clamp"). The area element is cos φ dφ dθ, so a
fibre interval (u_lo, u_hi) has mass [S(u_hi) - S(u_lo)] dθ / 4π, with S(u) = u/√(1+u²).

**Mechanism quantities** (`fibre.py`).
- `trace_arc_rows`: the gate pattern *on* each boundary arc, exactly, step by step. The roots of
  A_t,k · p(θ) = cos-sin forms give the arc segments of constant pattern.
- On a fibre the network is piecewise linear in u. All breakpoints are trig-linear in θ: gate switches
  u_g = -(A·p)/(A·n′), class ties u_cd = -(w·p)/(w·n′) with w = L_c - L_d, and the clamp. Over a θ-interval of fixed
  combinatorics, the wrong mass is therefore a closed form,
  ∫ S(R cos(θ-θ0)) dθ = asin(R sin(θ-θ0)/√(1+R²)).
- Each trace also returns its exact θ-validity interval: the nearest roots of all the order constraints it
  compared. The arc is therefore split exactly where the combinatorics change.

**Tiers.**

| tier | rule |
|---|---|
| F0 | gate pattern frozen at its arc value along the whole fibre (nothing resolved off the boundary) |
| F1a | the wedge attached at u = 0, traced exactly through gate switches until the fibre is first correct |
| F1 | traced until the end of the first linear piece that ends correct |
| F2 | F1a forward from the boundary plus F1a backward from the clamp (catches spill-over (b)) |
| **F2L2 (frozen recipe, `run_t3.RECIPE`)** | F2, but the forward trace continues until **2 consecutive linear pieces are fully correct** (catches near-attached wedges (a)) |
| FX | traced to the clamp: exact (= local enumeration of every region; validation only) |

Files:
- [fiber3.py](fiber3.py): geometry and closed forms.
- [fibre.py](fibre.py): the tiers.
- [run_t3.py](run_t3.py): scoring.
- [run_t75.py](run_t75.py): T75 comparison.
- [ladder.py](ladder.py): ladder table.
- [report.py](report.py): per-group tables (→ [report_T3.txt](report_T3.txt)).
- [mechanism3.py](mechanism3.py): plain-language quantities and the switch check.
- [exact3.py](exact3.py): exact truth and region counts.
- [diag_local.py](diag_local.py): wedge-thinness check.
- [count_regions.py](count_regions.py): T = 4 counts.
- [fibre4.py](fibre4.py), [check4.py](check4.py): the T = 4 prototype.
- [arc3.py](arc3.py): first polygon version, superseded (kept as a negative).

## Status

- Done for T = 3: rungs (4,3) re-run, (5,3) and (6,3) pass. M_8,3 is scored as data.
- Not started per owner directive: T ≥ 4 and H ≥ 8 rungs. The T = 4 code is a prototype, scored on one model only.

## Results

Truth: exact enumeration (`exact3.exact_acc`, i.e. `arc_m43_repro/exact.py`, generic in H). It agrees with 10^6-sample
MC on all 20 zoo T = 3 models (|diff| ≤ 2.3e-4, within MC error) and with FX to within 1e-14.

Models per rung:
- the 5 zoo seeds;
- ARC protocol: absolute σ = 0.05 on every weight, 2 noise seeds × 64 copies of the best-of-5 seed;
- leaderboard protocol: relative σ ∈ {0.0025, 0.005, 0.01, 0.02} × matrix RMS, 6 copies each. These are identical
  to `leaderboard.models(H,3)` (checked equal).

Ops: multiply-add = 1 (the T36/T63 convention). Sampling's MSE at equal ops is mean p(1-p) / (ops / ops_per_sample),
with ops_per_sample = T·H(H+1) + T·H.

Commands:
- `cd m16/approaches/understand_ladder && OMP_NUM_THREADS=1 python run_t3.py H 64`, then
  `python run_t3.py H 64 F2L2` (H = 4, 5, 6, 8; 157 models each; 1-4 CPU-min per H).
- `python report.py 4 5 6 8`
- `python ladder.py F2L2`

### Ladder table (frozen recipe F2L2, T = 3)

| model | params | linear regions (mean of seeds) | exact-enum ops | F2L2 ops | traces | MSE ratio vs sampling: seeds / ARC abs 0.05 / rel 0.0025 / 0.005 / 0.01 / 0.02 | seeds passed (per model) | worst single model | status |
|---|---|---|---|---|---|---|---|---|---|
| M_4,3 | 32 | 66 | 17.9k | 295k | 439 | 3.5e-2 / 3.4e-4 / 2.0e-5 / 1.5e-5 / 2.0e-6 / 3.9e-4 | 5/5 | 0.15 | base rung re-run: pass |
| M_5,3 | 45 | 97 | 26.9k | 462k | 562 | 2.6e-3 / 2.9e-7 / ~0 / ~0 / ~0 / 4.4e-10 | 5/5 | <0.01 | **T88: pass** |
| M_6,3 | 60 | 134 | 37.7k | 644k | 676 | ~0 / 2.5e-10 / ~0 / ~0 / ~0 / ~0 | 5/5 | <0.01 | **T93: pass** |
| M_8,3 | 96 | 236 | 62.9k | 1.18M | 863 | 1.4e-10 / 3.2e-7 / ~0 / ~0 / ~0 / 4.5e-8 | 5/5 | <0.01 | data only (H = 7 rung missing) |

"~0" means below 1e-20: the estimate equals the exact value to rounding. CPU time per model is 0.44-0.82 s (Python;
exact enumeration takes 0.04-0.13 s).

### Other tiers (same models; MSE ratio for seeds / ARC abs 0.05; seeds passed per model)

| model | F0 | F1 | F2 | FX (exact) | T75 frozen K=(4,4) | T75 K=(H,H) |
|---|---|---|---|---|---|---|
| M_4,3 | 70 / 2.1, 1 of 5 | 0.93 / 0.45, 4 of 5 | 0.50 / 0.024, 3 of 5 | 0 / 0 (308k ops) | (T75 README) | - |
| M_5,3 | 88 / 21 | 1.08 / 3.8e-3, 4 of 5 | 0.20 / 1.5e-4, 5 of 5 | 0 / 0 (499k ops) | 0.52 / 0.11 (29k ops) | 0.40 / 0.11 |
| M_6,3 | 8e-6 / 6.5e-3 | 2e-6 / 4.7e-4 | 9e-8 / 1.0e-5 | 0 / 0 (739k ops) | 0.50 / 2.0e-3 (35k ops) | 0.52 / 2.0e-3 |
| M_8,3 | 3.1 / 0.13 | 1e-6 / 1.4e-3 | 4e-6 / 1.5e-4 | 0 / 0 (1.8M ops) | - | - |

T75 (`run_t75.py`) passes (5,3) and (6,3) at group level, at about 1/15-1/18 of F2L2's ops. Per model it fails one
seed at (6,3): seed 1 has error 3.2e-3 and per-model ratio ≈ 2.7.

### Scaling (the owner's question), T = 3, zoo seeds

| H | regions | exact-enum ops | F2L2 traces | F2L2 ops | F2L2 ops / region | F2 ops / region | FX ops / region |
|---|---|---|---|---|---|---|---|
| 4 | 66 | 17.9k | 439 | 295k | 4.5k | 4.2k | 4.7k |
| 5 | 97 | 26.9k | 562 | 462k | 4.8k | 4.0k | 5.1k |
| 6 | 134 | 37.7k | 676 | 644k | 4.8k | 3.7k | 5.5k |
| 8 | 236 | 62.9k | 863 | 1.18M | 5.0k | 3.6k | 7.6k |

Fitted over H = 4 → 8: regions ∝ H^1.84, exact enumeration ∝ H^1.81, F2L2 traces ∝ H^0.97, F2L2 ops ∝ H^2.0 and
F2 ops ∝ H^1.61. **Stated law [measured, 4 points]:** the number of boundary traces grows like H (that is the
boundary-family count, as hoped). Each trace costs O(T·H), because every gate's crossing is tested, so the total is
∝ T·H² ≈ the region count at T = 3. At T = 3 the recipe gains nothing in scaling over enumeration. Any gain has to
come along T: regions ~ H^(T-1), boundary cells ~ H^(T-2).

### T = 4 prototype (exploratory; not a rung)

On M_8,4 seed 1, 4·10^6-sample MC gives 0.98746 ± 0.00006:

| tier | estimate | ops |
|---|---|---|
| F0 | 0.98085 | 2.7e7 |
| F1a | 0.98751 | 2.7e8 |
| F1 | 0.98748 | 2.8e8 |

`check4.py` checks the closed-form vector-area integral against a 30×30 cubature: rel. error ≤ 2e-6.

Region counts (`count_regions.py 8 4`): 4-D gate patterns seen in 5·10^5 / 2·10^6 samples are 1.2-1.6k / 1.4-2.0k
per seed (a lower bound, not converged). The exact 2-D cells on the boundary faces number 2.6-3.5k. **[measured]**
At H = 8, T = 4 the boundary arrangement is not smaller than the volume arrangement. There are six boundary
hyperplanes, and each sees all ~32 gate planes.

## Compute accounting

- **Counted** (`fiber3.Ops`, multiply-add = 1; sqrt / atan2 / asin / division = 1):
  - every gate-row composition (rows_at: H·|g|·t per step);
  - every dot product with P, U, n′;
  - root and constraint evaluations (6 ops per constraint per validity interval);
  - class-tie and gate-crossing evaluations along each fibre;
  - closed-form asin terms (12 each);
  - in fibre4, clips (3 per vertex + 10 per new vertex), vector areas (20 per edge) and cubature nodes.
- **Not charged:**
  - the model-independent fibre families (arc endpoints, clamp normals; computed once, like exact.py's start cones);
  - Python bookkeeping;
  - ground truth.
- Totals per model are in the tables above. The F2L2 ops are an implemented count, not a lower bound of what is
  implementable. Incremental gate updates would cut the O(T·H) per-trace factor; this is not done, so it is not
  counted (rule 7 of the leaderboard).
- The full T = 3 scoring (4 H × 157 models × 6 tiers) took about 25 CPU-min on one core.

## Claims

- [measured] One frozen deductive recipe (F2L2) beats sampling at equal ops on every model of rungs (4,3), (5,3),
  (6,3), and on M_8,3. That covers 5 zoo seeds + 128 ARC-protocol copies + 24 leaderboard-protocol copies per rung.
  The worst per-model MSE ratio is 0.15 (M_4,3); on H ≥ 5 the estimates are exact to rounding on most models.
- [measured] F2L2 is close to exact enumeration in content: its output equals FX's to rounding on H ≥ 5, and it
  costs 66-96% of FX. So at T = 3 the "understanding" content of the recipe is mostly the choice of where to look
  (the boundary fibres), not a compression of the computation.
- [measured] The cost is 10-20x plain exact enumeration and grows like the region count (ops/region flat at
  4.5-5k). The boundary traces themselves grow like H.
- [measured] The F0 (frozen) and F1 tiers fail exactly where the plain-language mechanism predicts something new:
  - F0 fails where a gate switches inside the wedge (M_8,3 zoo seeds: errors -3e-4 to -2e-3);
  - F1 fails on the undertrained seeds (M_5,3 seed 0, M_4,3 seeds 1-4), whose error wedges are detached from the
    boundary or spill across the clamp.
- [measured] FX (local fibre enumeration) reproduces exact enumeration to 1e-14 on 628 models, so the fibre
  partition and the closed-form masses are correct.
- [inferred] The seeds-vs-noisy-copies gap at (5,3) has a structural cause. The noisy copies perturb the
  well-trained seed 1 (accuracy 0.999), while seed 0 is undertrained (0.942) and has a qualitatively different
  error geometry: a single x0 neuron, and wedges off the boundary.
- [speculative] An advantage over enumeration would appear only for T ≥ 5 and large H, where boundary cells
  (~H^(T-2)) are much fewer than regions (~H^(T-1)). The H = 8, T = 4 data point does not show it.

## Alternatives not ruled out

Before building, the hypothesis was "errors are thin wedges attached to boundaries, explained by few active
neurons". Alternatives and the tests used:

1. *Errors are interior regions, not wedges.* Test: `diag_local.py` angle-to-boundary distribution, and FX vs F1 per
   family. The hypothesis holds for well-trained seeds. It fails for undertrained ones, which motivated F2/F2L2.
2. *Many neurons are active at boundaries, so no low-dimensional readout exists.* Measured: 2-5 neurons are active,
   so the pair readout of T63 is dropped and frozen linear logits are used.
3. *The fibre geometry double-counts.* Ruled out by FX = exact.

Not ruled out:
- **The recipe was chosen after looking at the test seeds.** LOOK = 2 was chosen after seeing that F2 fails per
  model on M_4,3 seeds 1 and 4. There are no continuous parameters; the discrete knob is LOOK ∈ {0, 1, 2, 3}, and
  L = 3 was exact on all 6 spot-checked models. The noisy copies were not used for this choice, but the zoo seeds
  were.
- The thin-wedge picture may hold only because these models are tiny: at H = 8, T = 4 the boundary cells outnumber
  the regions.
- The ops counter is our own. An independent recount (as T74/T84 did for T63/T75) has not been done.
- MC truth was used only as a cross-check of exact truth. No ≥1e7 MC was run (exact truth is available at T = 3).

## Failure modes

- **Undertrained models** with interior errors (M_5,3 seed 0): F1/F2 under-count. F2L2's look-ahead fixes the
  observed cases, but an error island farther than 2 pieces from both fibre ends would be missed. The error is
  always one-sided: a missed island makes the accuracy estimate too high.
- **Cost:** F2L2 costs about as much as tracing every region in the fibre families. On these small models it is
  more expensive than exact enumeration.
- **T = 4:** the prototype is accurate on one model but costs 2.8·10^8 ops (sampling would need ~8.7·10^5 samples
  at that cost). Its face cells outnumber the regions at H = 8.
- Ties: the all-off pattern returns class 0 (argmax tie rule), as in exact.py.

## Exhaustion checklist (rung failures and what was tried)

| failure | diagnosis | tried | result |
|---|---|---|---|
| F0 fails on M_8,3 seeds and M_4,3/M_5,3 | gate switches inside the wedge make the frozen logits invent or miss wrong regions (shown on M_8,3 seed 3, x0 = x2 family) | F1: trace through gates | fixed for well-trained seeds |
| F1 fails per model on M_5,3 seed 0 (ratio 1.47) and M_4,3 seeds | (b) wedges spilling across the clamp from the neighbouring boundary | F2: second trace from the clamp; first version had a bug dropping multi-piece intervals (fixed) | 5.4e-3 → 1.8e-3 on seed 0 |
| F2 fails per model on M_4,3 seeds 1 and 4 (1.56) | (a) wedges separated from the boundary by a thin correct sliver | look-ahead L = 1, 2, 3 | L = 2: all pass; L = 3: exact on spot checks |
| cost not better than enumeration | per-trace O(T·H) gate checks; at T = 3 traces ~H but regions ~H^1.8 | (not yet) incremental crossing order, gate pruning by impact (T75-style) | open (T95) |
| arc3.py (polygon half-chambers + lazy check) | lazy wrong-sliver checks explode (0.5-1.7M ops) | replaced by fibre coordinates | superseded |

## Proposed changes to shared code

None.
