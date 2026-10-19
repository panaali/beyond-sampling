# late_triple_verify: independent re-check of the late-triple certification failure with library verifiers (T418)

↑ [m16 approaches](../README.md) · [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · T414 code: branch
`claude/t192-gpu-cost-law`, `m16/approaches/input_bab/`

**Question.** T398 (forward neuron splitting) and T414 (input-space BaB with an own float64 CROWN) both fail to certify
P(E1(t0 = 2.5)) on late triples such as (3,7,8). Is that failure real (AGENTS rule 5 re-check with different code), and
do the strongest standard verifiers, auto_LiRPA alpha-CROWN and alpha-beta-CROWN (input and ReLU splitting), change the
verdict?

**Status.** DONE 2026-10-19.
- **Pre-registered line: KILL (unverified)** on E3, E4 and E5.
- **Secondary D3 and D2 (unverified) put the cause in T414's split rule, not in the bounds.** With alpha-CROWN, a
  widest-dimension split and pop order by near-exact mass, the near-exact mass falls with slope -1.13 and projects 1.6e6
  boxes to reach 1e-10 on (3,7,8), and 2.4e7 boxes on (9,8,2) (slope -0.73). The rigorous radial charge still projects 1.6e10, because that charge is loose on
  wide boxes.
- So the late-triple failure is NOT verified dead. It is scoped to T414's margin-steered split.

Pre-registered 2026-10-19 (this section was written and pushed before any scored run, bebd3c5e). Owner: Claude
session_011FA8RfY3APG3rkDG2GxScR (own container). DRI session_01CD3g9Cvc3qk3pM11eyXXzj. CPU only.

## Pre-registration (fixed before any scored run)

**Setup (independent code).**
- `net.py`: M_16,10 unrolled as a torch module on T414's homogeneity-reduced coordinates (x = g y, g = x_p2 - x_p3 >=
  2.5, y_p3 = u_r, y_p2 = u_r + 1, y_p1 = u_r + 1 + u_a, y_l = u_r - u_l, u_a, u_l >= 0; root box u_r in [-4, 4], the
  rest in [0, 4]; domain tail 6.2e-12 charged once). Output: the 9 margins L_j - L_p3. A box is pruned iff some
  certified lower bound is > 0. Weights from `m16/common.py` (read only). Own slab sampler, own radial charge (T414's
  formula), own near-exact box mass (2-D Gauss-Legendre over (u_r, g), u_a and u_l integrated exactly).
- Libraries: auto_LiRPA and alpha-beta-CROWN from `github.com/Verified-Intelligence/alpha-beta-CROWN` (commit b58d83c,
  auto_LiRPA submodule e393035), torch 2.11 CPU, float64 everywhere (`general/double_fp`). Python 3.13 (the packages
  pin 3.11; they import and run).
- Feasibility trials (not scored, before this section): auto_LiRPA CROWN ~180 boxes/s, alpha-CROWN (20 iterations)
  ~12 boxes/s on one core. alpha-beta-CROWN input-split BaB with the 9-row OR spec runs (~800 domains/s).
  ReLU-split BaB (beta-CROWN) crashes on the 9-row OR spec, so it is run on single specs 'margin_j > 0' (top 2 j by
  CROWN lower bound); it is slow (~20 CPU-s for a 4-domain proof). Mixed input + ReLU splitting
  (`branching_input_and_activation`, marked experimental) crashes on both spec forms: recorded as not runnable.

**Bounders.** C = auto_LiRPA CROWN; A = auto_LiRPA alpha-CROWN (20 iterations, library default: intermediate bounds
fixed from CROWN); AF = alpha-CROWN with `fix_interm_bounds = False` (intermediate bounds optimised too; looseness test
only, if it runs); ABC-in = alpha-beta-CROWN input-split BaB on the OR spec; ABC-relu = beta-CROWN ReLU-split BaB on the
top-2 single specs.

**E1 Re-check (scored).**
- (a) Root u-box: C lower bounds vs T414's `margins()` (T414's own code, unchanged except the import path).
- (b) Boxes from a 600 CPU-s re-run of T414's `ibab.py 3,7,8` (its own search, here): 2,000 random open boxes plus
  its reservoir of pruned boxes. PASS iff max |lb_C - lb_T414| <= 1e-6 (1 + |lb|) on every box and spec.
- (c) Looseness on the 200 top-charge open boxes of (b): sampled minimum (2,000 uniform points per box, own forward)
  minus the lower bound, for the best spec, under C, A and AF; pruned count under each, and under ABC-in (timeout 10 s)
  and ABC-relu (10 s per spec). T414 reported a median of 176 margin units for its CROWN (10-min trial).

**E2 Certified width (scored, secondary projection).** 200 exact slab points of (3,7,8) (own sampler, seed 418; event
points reported separately). For each, the largest half-width w of the u-cube (clipped to u_a, u_l >= 0) that the
bounder certifies: geometric bisection, 20 steps for C and A; ABC-in on the first 60 points, 6 steps in [w_A, 4 w_A]
(timeout 10 s, also recording BaB domains visited); ABC-relu on the first 20 points, 4 steps in [w_A, 4 w_A]. The
covering estimate is T414's: E_slab[1 / ((2w)^9 p_u)] over the highest-density 99.8% of slab mass, ideal cubes and
w halved; for ABC also multiplied by the mean domains visited per certified cube (the number of bound evaluations).
C should reproduce T414's median w = .103.

**E3 alpha BaB (primary scored run).** `lbab.py 3,7,8 OUT 7200 --stages crown,alpha --K 256`: T414's search
(best-first by radial charge, pop K, bisect along argmax_d |c_d| width_d of the best spec's linear lower bound), bounds
C then A on C's survivors. 2 CPU-h, one thread.

**E4 beta phase (scored).** Resume from E3's open set: `--stages crown,alpha,abc --abc-mode M --abc-timeout 2 --K 64`,
2 CPU-h, with M = whichever of ABC-in / ABC-relu certifies the larger median width in E2 (rule fixed now).

**E5 (if time).** E3 on (9,8,2).

**Charge and reporting.** Primary charge = T414's rigorous radial charge (sum over open boxes + tail); near-exact
box mass reported alongside (float, not a certificate). Reported: looseness, pruned fraction, charge-vs-boxes curves,
projections. Boxes = children evaluated; for E4 also boxes + abcrown BaB domains (bound evaluations).
Projection: log-log fit of charge vs boxes over the last decade of the curve, extrapolated to 1e-10.

**Kill line (DRI's, fixed).** KILL if the projected boxes to reach 1e-10 exceed 1e9 per triple, or the charge stalls
above 1e-9 (less than a 2x decrease over the last decade of boxes). GO only if a projection lands below 1e9 boxes per
triple (E3 or E4; for E4 counted in boxes + domains).

**Alternatives written before results.**
- A failure under A may be the library default of fixed intermediate bounds (the 10-step compounding sits in the
  intermediate layers). Check: AF in E1c.
- abcrown's own split heuristics may beat T414's; E2's ABC numbers use its heuristics, E3 uses T414's.
- If E2 covering estimates fall below 1e9 while E3/E4 curves do not, the obstruction is the search (ordering or split
  rule), not the bounds; that would be reported as GO-able by a different search, not as a kill of the bounds.
- A re-check mismatch in E1 could be a T414 bug or an auto_LiRPA option difference (e.g. lower-slope rule); checked by
  comparing on single-unstable-neuron cases if it happens.

**Deviation D1 (2026-10-19, written before its run; SECONDARY, not used for the kill decision).** Mid-run, E2 gave an
alpha-CROWN covering estimate below 1e9 (ideal 2.9e5, halved 1.5e8), while E3's near-exact mass sat flat at ~2.8e-8
with pruning almost stopped. This is pre-registered alternative 3 (search vs bounds). One secondary run:
E3's search with the pop order switched to the near-exact box mass (`--order quad`), everything else unchanged, 2 CPU-h,
`lbab.py 3,7,8 OUT 7200 --stages crown,alpha --K 256 --order quad`. Reported: the near-exact curve, pruned fraction and
projection. The radial charge stays the rigorous figure.

**Deviation D2 (2026-10-19, written before its run; SECONDARY).** E3 and D1 (40k boxes so far) both sit flat at
~2.8e-8, while E2's covering estimates under alpha-CROWN are 3e5-1.5e8. A flat curve over one decade cannot tell a real
stall from a tree that has not yet reached certifiable box sizes. `dyadic.py` separates the two. For 200 exact slab
points it finds the coarsest box of a fixed bisection tree (widest dim, midpoint) containing the point that the bounder
certifies, using binary search on depth, which assumes monotone certification. Leaves of an ideal-order BaB ~
E_slab[1 / (vol p_u)] over 99.8% of the mass; nodes <= 2x. Run with alpha-CROWN and with CROWN:
`dyadic.py 3,7,8 OUT 200 alpha-CROWN` and `... CROWN`. If the alpha estimate is < 1e9, the late-triple failure is a
search/compute failure, not a bound failure, and is reported as such (not GO by the pre-registered line).

**D2 first results and deviation D3 (2026-10-19, written before D3's run; SECONDARY).** D2 with 200 points:
CROWN median certified depth 39, ~6.4e7 leaves; alpha-CROWN median depth 29, ~1.4e5 leaves. `soundcheck.py` passed on
the 200 alpha-certified boxes: 4e5 points, 0 events, 0 bound violations. CROWN certifies 15 of those 200 boxes. D3
tests whether a BaB that follows D2's tree realises the estimate: `lbab.py 3,7,8 OUT 7200 --stages crown,alpha --K 256
--order quad --split widest`, 2 CPU-h. Prediction if D2 is right: the near-exact mass falls well below 2.8e-8 once the
node count nears ~3e5. D2b repeats D2 with more points (CROWN n = 5,000; alpha n = 1,000) for the tail-dominated estimate.

## Method
- `net.py`: model, coordinates, exact slab sampler, radial charge, near-exact box mass. Root mass check: 5.59749e-8
  (T414: 5.59749e-8); root radial charge 864 (T414: 864).
- `lbab.py`: BaB with library bounds. C = auto_LiRPA CROWN on every child. A = auto_LiRPA alpha-CROWN on C's survivors.
  ABC = alpha-beta-CROWN on A's survivors (E4 only). Options: `--order radial|quad`, `--split margin|widest`.
- `recheck.py` (E1), `width.py` (E2), `dyadic.py` (D2), `soundcheck.py` (soundness), `analyze.py` (projections).
- Environment: `pip install torch==2.11.0` (CPU) in a venv; clone alpha-beta-CROWN with submodules; PYTHONPATH =
  `auto_LiRPA:complete_verifier:this folder`; extra pip packages onnx2pytorch (the pinned commit), onnx, onnxruntime,
  skl2onnx, sympy, pyyaml, pandas, psutil, sortedcontainers, termcolor, appdirs, tqdm, graphviz. `m16/common.py` is
  loaded by file path, because putting `m16/` on sys.path shadows abcrown's `state` package with `m16/state.py`.
- Runs: 1 thread each, `ulimit -v 2500000`; data in `data/t418/` (git-ignored). Run peaks: lbab ~1.8 GB RSS, under 2 GB.

## Results ([measured], unverified)

**E1 Re-check of T414's CROWN: PASS.**
- auto_LiRPA CROWN vs T414's `margins()`:
  - root box: max relative difference 6e-16;
  - 2,000 open boxes from a 600 CPU-s re-run of T414's own search: max relative difference 1e-13;
  - its 3,793 pruned boxes: max relative difference 3e-13; all pruned by both.
- T414's CROWN is therefore exactly standard CROWN (adaptive lower slope).
- Looseness on the 200 top-radial-charge open boxes of that re-run (sampled min minus lower bound, best spec):

| bounder | median | p10 | p90 | boxes pruned of 200 |
|---|---|---|---|---|
| C (CROWN) | 369 | 130 | 2450 | 0 |
| A (alpha-CROWN) | 165 | 78 | 306 | 1 |
| AF (alpha, intermediate bounds optimised) | 165 | 78 | 306 | 1 |
| ABC-in (input-split BaB, 10 s) | | | | 64 (mean 6,200 domains) |
| ABC-relu (beta-CROWN, top-2 specs, 10 s each; first 50 boxes) | | | | 4 of 50 |

- All 200 boxes are prunable in principle (sampled best-spec minimum > 0); none holds a sampled event.
- AF printed the same numbers as A, so the `fix_interm_bounds = False` setting most likely did not take effect through
  `set_bound_opts`. Not verified; see Alternatives.
- T414's 176 came from a 10-min trial. These boxes come from a 600 s re-run with K = 1024, and are larger.

**E2 Certified width around 200 exact slab points (3,7,8); 0 events among the points.**

| bounder | median w | p10 w | covering, ideal cubes | covering, w halved | note |
|---|---|---|---|---|---|
| C | .105 | .050 | 1.4e9 | 7.0e11 | T414: median .103, 2.5e8 / 1.3e11 |
| A | .142 | .093 | 2.9e5 | 1.5e8 | |
| ABC-in (first 60 pts) | .181 | .116 | 3.9e4 (x 4,436 domains = 1.7e8 evals) | 2.0e7 | A on the same points: 4.3e5 |
| ABC-relu (first 20 pts) | .146 | .110 | 5.0e5 | 2.6e8 | median gain over A 1.00 |

- alpha-CROWN widens the certified cube by 1.35x at the median and 1.9x at p10. Because the covering estimate is
  tail-dominated, that alone cuts it ~5,000x.
- alpha-beta-CROWN input splitting widens it another 1.2x, but each certified cube costs ~4,400 BaB domains. Counted
  in bound evaluations it is ~400x worse than A.
- ReLU splitting (beta-CROWN) gains nothing within 10 s.

**Pre-registered BaB runs (2 CPU-h each; projection = log-log fit over the last decade, to 1e-10):**

| run | triple | search | boxes | pruned | radial (rigorous) | near-exact mass | radial projection | near-exact |
|---|---|---|---|---|---|---|---|---|
| E3 | (3,7,8) | T414's (radial order, margin split), C + A | 1.35e5 | 0.6% | 5.65e-5 (slope -.70) | 2.81e-8 (50%), stall | 2.0e13 | stall |
| E4 | (3,7,8) | E3 continued, + ABC-in (2 s) on A's survivors | +3,456 (+3.96e6 abc domains) | 2.6% (89 by ABC) | 5.56e-5, stall | 2.81e-8, stall | stall | stall |
| E5 | (9,8,2) | as E3 | 1.42e5 | 7.3% | 5.97e-5 (slope -.79) | 2.92e-8 (52%), stall | 3.1e12 | stall |

**Verdict by the pre-registered line: KILL (unverified).** Both kill conditions hold on every pre-registered run. The
projections are > 1e9, and the charge stalls above 1e-9 (near-exact mass flat at ~50% of the slab). This re-checks
T414's KILL with different code and library bounds: under T414's search, neither alpha-CROWN nor alpha-beta-CROWN
changes the verdict.

**Secondary (deviations, not used for the verdict).**

| run | triple | search | boxes | pruned | radial (rigorous) | near-exact mass | radial projection | near-exact projection |
|---|---|---|---|---|---|---|---|---|
| D1 | (3,7,8) | quad order, margin split | 1.38e5 | 0.5% | 1.07e-4 | 2.81e-8, stall | 3.8e14 | stall |
| D3 | (3,7,8) | quad order, **widest split** | 1.37e5 | 29.6% | 3.85e-7 (slope -.71) | **1.61e-9 (2.9%)** (slope -1.13, 12.6x per decade) | 1.6e10 | **1.6e6** |
| D3 | (9,8,2) | as above | 1.46e5 | 24.4% | 8.17e-7 (slope -.39) | **4.17e-9 (7.5%)** (slope -.73, 5.1x per decade) | 1.8e15 | **2.4e7** |

D2 dyadic-leaf estimate: leaves of the widest-split tree at the coarsest certified box around exact slab points.

| triple | bounder | n | median depth | leaves (99.8% of mass) | leaves (all points) | top-1% share |
|---|---|---|---|---|---|---|
| (3,7,8) | C | 5,000 | 39 | 3.8e7 +- 0.9e7 | 4.0e7 | 80% |
| (3,7,8) | A | 1,000 | 29 | 1.25e5 +- 0.3e5 | 1.7e5 | 55% |
| (9,8,2) | C | 5,000 | 32 | 1.2e9 +- 0.5e9 | 1.2e9 | |
| (9,8,2) | A | 1,000 | 29 | 4.3e5 +- 2.1e5 | 4.8e5 | 70% |

Soundness (`soundcheck.py`, reference forward on raw x at random g > 0; the run rows are reservoirs of pruned boxes):

| boxes checked | points | events | bound violations |
|---|---|---|---|
| D2 alpha-certified | 4e5 | 0 | 0 |
| D3 run, 5,117 boxes | 1.28e7 | 0 | 0 |
| E3 run, 591 boxes | 1.48e6 | 0 | 0 |

On recomputation in other batches, alpha alone re-certifies 5,097 of 5,117 and 505 of 591 boxes; the rest were pruned
by C, or by A in a different batch. So auto_LiRPA's alpha-CROWN is not always at least as tight as its CROWN, and
batch composition matters a little. lbab prunes if either C or A certifies, so this does not affect soundness.

**Reading.** T414's margin-steered split (and my re-implementation in E3 / D1 / E5) never splits the dimensions with
small margin coefficients. T414 saw two u_l dims never split. The boxes then stay wide in those dims, and CROWN or alpha
stays loose there. Splitting the widest dimension instead makes alpha-CROWN prune 30% of the boxes and drives the
near-exact mass down 12.6x per decade. D2 independently says an ideal widest-split BaB needs ~2.5e5 nodes with
alpha-CROWN, or ~8e7 with plain CROWN, on (3,7,8). Both are below the 1e9 line. The remaining gap is the charge: the
rigorous radial charge is 1e2-1e7x looser than the box mass on wide boxes, so its curve projects 1.6e10.

## Compute accounting
CPU only, one thread per process, <= 2 GB each, at most 2 heavy processes (one ~3-min overlap of D2 with two runs).
- T414 re-run 600 s; E1 ~1.1 CPU-h (abcrown parts 1,556 s + 1,797 s);
- E2 ~1.6 CPU-h; E3, E4, E5, D1, D3 x2: 2 CPU-h each;
- D2 on (9,8,2) ~0.25 CPU-h;
- D2 / D2b ~0.3 CPU-h; soundness ~0.3 CPU-h.
- Total ~16 CPU-h. No GPU, no Colab.

Model evaluations:
- one CROWN bound ~ one backward pass per pre-activation layer (~1e6 FLOPs per box);
- alpha-CROWN ~20x CROWN;
- an abcrown domain ~ one alpha-CROWN bound.
- Sampling checks: ~1.6e7 forwards.

## Claims
- [measured] T414's CROWN equals auto_LiRPA CROWN to <= 3e-13 relative on its root, open and pruned boxes. Its KILL
  under its own search reproduces with library bounds:
  - alpha-CROWN BaB (E3) leaves the near-exact mass flat at ~50% of the slab after 1.35e5 boxes;
  - alpha-beta-CROWN input-split BaB on top (E4) prunes 89 boxes for 4e6 domains and does not move the charge;
  - the same holds on (9,8,2) (E5).
- [measured] alpha-CROWN halves CROWN's looseness on T414's frontier boxes (median 369 -> 165) and widens certified
  cubes 1.35x (median) / 1.9x (p10). Beta-CROWN ReLU splitting adds nothing within 10 s. Input-split BaB adds 1.2x
  width at ~4,400 domains per cube. Mixed input + ReLU splitting crashes in this alpha-beta-CROWN version.
- [measured, secondary] With a widest-dimension split and mass order, alpha-CROWN BaB brings (3,7,8)'s near-exact mass
  to 1.6e-9 in 1.37e5 boxes (slope -1.13), projecting 1.6e6 boxes to 1e-10. On (9,8,2): 4.2e-9 in 1.46e5 boxes (slope -.73),
  projecting 2.4e7.
- [measured, secondary] Dyadic-leaf estimates: (3,7,8) ~1.25e5 leaves with alpha-CROWN, ~4e7 with CROWN;
  (9,8,2) ~4.3e5 with alpha-CROWN, ~1.2e9 with CROWN (so plain CROWN is borderline even with an ideal split).
- [inferred] The late-triple obstruction in T414 is mainly the split heuristic, not CROWN looseness per se; T414's
  1e14-1e15 projections come from a search that cannot reach certifiable boxes.
- [inferred] A certified bound needs a rigorous box-mass bound much tighter than the radial charge, e.g. interval
  Gauss-Legendre with error bounds, or the exact 1-D Phi factors plus a rigorous 2-D enclosure.
- [speculative] At ~19 alpha boxes/s, 1e-10 per triple needs ~25 CPU-h (1.6e6 boxes). ~1e-12 per triple (1e-9 over 720 triples)
  needs ~1e8 boxes, ~1e3 CPU-h per late triple, unless better charges or CROWN-first batching cut the cost.

## Alternatives not ruled out
- **The D3 projection extrapolates one decade.** The slope could flatten when the boxes near the event boundary
  dominate; these boxes can never be pruned and must be charged by mass. Check: continue D3 for 10x the boxes.
- **D2 estimates are tail-dominated** (top 1% of points = 55-80% of the estimate) and assume monotone certification in
  depth.
- **AF did not take effect** (identical numbers), so intermediate-bound optimisation is effectively untested. Pass
  `fix_interm_bounds: False` at BoundedModule construction.
- **ABC timeouts were short** (2 s in E4, 10 s elsewhere); abcrown's own split heuristics were left at their defaults.
- **Mixed input + ReLU branching crashed.** It is untested; a later alpha-beta-CROWN version or a vnnlib run may work.
- **Float only, no outward rounding.** Soundness is checked by sampling, not proved.

## Failure modes
- The radial charge dominates the rigorous figure on wide boxes. Ordering by radial (E3, E5) wastes effort on low-mass
  boxes.
- alpha-CROWN costs ~15x CROWN per box. Without a CROWN-first filter, plain CROWN with ~300x more boxes (D2: 4e7 vs
  1.25e5 leaves) is roughly a wash in CPU.
- alpha-beta-CROWN on CPU: per-call overhead plus thousands of domains per box. It is not competitive as a leaf
  verifier here.

## Proposed follow-ups (backlog, not done here)
- Rigorous near-exact box-mass bound plus the D3 search (alpha-CROWN, widest split, mass order) to 1e-10 / 1e-12 on
  (3,7,8) and (9,8,2); kill line 1e9 boxes. Run D3 for 10x longer first.
- AF with `fix_interm_bounds = False` set at construction, and abcrown with longer timeouts on D3's frontier.
