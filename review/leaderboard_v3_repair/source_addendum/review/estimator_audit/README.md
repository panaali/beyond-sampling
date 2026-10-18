# Estimator audit for leaderboard v3 (classes T14, compute rule T16, advice, H/T generality)

↑ [AGENTS](../../AGENTS.md) · [leaderboard](../../m16/leaderboard.py)

Independent audit of the 15 `m16/approaches/*/estimator.py` entries, the code they import, and their READMEs.
State audited: working tree of 2026-10-17 (leaderboard v3 already committed as 6915d15). No approach file was
edited. Everything I wrote is in this folder.

**How to read the time check.** On this machine plain sampling runs 16k sequences in 0.025 s (vectorised numpy,
about 1.6 µs per pass). Any estimator with Python-level loops or small batches will therefore show a ratio of 3-10
from overhead alone. Treat the time ratio only as a pointer. The FLOP findings below come from reading the code, plus
an instrumented count (`fullpass_count.py`) for the segment walkers.

## Summary table

FPE = forward-pass equivalent, 6e3 FLOPs. "Time" = CPU time of `est(orig, 16000, rng)` divided by plain sampling's
CPU time at the reported cost, single thread. A second, warm call is shown where it differs.

| name | class | cost complies? | uncounted work (rough FPE) | advice (fitted on test-model outputs) | model-free precompute (declared?) | hardcoded H/T | time check (16k) |
|---|---|---|---|---|---|---|---|
| tube | quadrature | partly, idealised | Each cell is charged 2(T-t)/T but the code recomputes 2 full passes: as implemented **2.06x** the charge (measured). Root-finding and the output layer are "negligible" (≈+10-25% on late-step cells). | **yes**: q23.npy/q12.npy importance tables, SHARE, COST (orig) | Z/CDF tables, Sobol (yes, README) | T=10, crashes at (12,8) | 6.3 |
| sci_laplace | quadrature | partly, idealised | Same scheme as tube: **1.81x** as implemented | **yes**: SHARE, COST, K0 (orig) | yes | T=10 (via tube), crashes | 5.8 |
| sci_tao | quadrature | partly, idealised | Same scheme: **2.23x** as implemented. Roots and output layer not charged. | none (the n0 guess of 8 FPE per face is trivial) | yes | **generic** (ran on random 12x8 and zoo 8x6) | 10.6 (warm 8.6) |
| sci_feynman | quadrature | partly, idealised | Same incremental-cell scheme. The output-layer envelope loop (≤80 iterations per cell) is uncounted. | **probably**: SHARES 70/26/4 and COST23/12 have no stated origin (README says none fitted) | yes | T=10; **runs silently wrong** at (12,8) | **109-119** (Python overhead, 405 tiny batches) |
| boundary | sampling | yes (same idealised increments) | Envelope arithmetic only (declared). Input generation is charged, now free under rule 5 (slight over-charge). | **yes**: DEFAULT_FRAC, COST0 (orig pilot, declared). GA/GB=0.15/0.1 tuned with tune.py on orig (**undeclared**). | P_R, P_ACC, inverse CDFs (yes) | T=10; **runs silently wrong** | 6.0 |
| control_variate | sampling | partly, idealised | COST_Q=1.6 assumes rank-1 gate-toggle updates. The code recomputes 2 logit products per piece: **≈3.3 FPE** per sample as implemented (≈2x). | **yes**: alloc.json (orig pilot, declared) | precomp.npz, 20M draws (yes) | T=10, crashes | 6.0 |
| debiased_series | sampling | model passes yes. Rejection: see note | Below 128k = stratified.py. Each of 7 strata draws at least one 200k batch, so ≥1.4M candidates per call (≈56k FPE at 0.04 each, 3.5x the 16k budget). Free under rule 5 only if stratum rejection counts as "label". | **yes**: stratified_design.json alloc (orig). At ≥128k also alloc_it5.json (orig) and a 128k switch rule from an orig variance table. | P (20M), table_it5 (16M) (yes) | T=10; **runs silently wrong** | 24.6 (warm 17) |
| tie_lines | sampling | yes (over-charges generation, ~+10%, now free) | none found | **yes**: joint_design.json alloc/use_swap (orig, 1e6+1e6 evals, declared) | stratum masses, 4e7 draws (yes) | T=10; **runs silently wrong** | 32.6 cold, **1.8 warm** (JSON load) |
| great_circle | sampling | under by ≈1.3x even if implemented efficiently | Per piece: 160 gate-root arctan2 + 90 pair roots + midpoint checks. As implemented ≈7 FPE per piece (181 midpoints); efficient ≈3.8 vs 3 charged. A 512-point stratum grid per circle (model-free, ≈3%). | **yes**: BETA (2000 orig circles, pilot_0.npz), PIECES_GUESS (orig) | P from stratified_design (yes) | T=10, crashes | 21 (20 circles, Python loop over ~250 pieces) |
| branch_bound | **quadrature** at ≤144k budgets (mechanistic B&B above) | yes | none at leaderboard budgets (180 passes, model-free masses). Docstring says the switch is at "~48k"; the code puts it at 3·180·267 ≈ 144k. | none | Gauss-Legendre (yes) | T=10/H=16, crashes | 114 on a 180-FPE run (meaningless; model-free cell measure dominates) |
| rank_exact | mechanistic | yes | argmax checks (~Q·10 FLOPs per class), negligible | none | EOSK (2M draws), Sobol rule (yes) | T=10/H=16, crashes | 1.9 |
| propagation | mechanistic | roughly yes | Correlation matrix, ndtr and argsort in enumerate_gates, merge sorts (<10%) | mild: budget→eps map calibrated on orig cost (affects cost only) | none | T=10/H=16, crashes | 18 |
| cone_enum | mechanistic | roughly yes | LP charged by a FLOP proxy, not HiGHS's real work. Row normalisation, Wo@M, SVD/Cholesky ≈2-3 FPE per cone of ≈630 (<1%). | none | uses arc_eval.EOS (400k draws); README says none (minor) | T=10/H=16, crashes | 33 (scipy LP overhead) |
| reliability | mechanistic | yes (≈5% uncounted, declared) | eig/QR per limit state, Owen's T for Ditlevsen bounds (declared ≈5%) | possible selection: LADDER/CFG swept against test-model truths (sweep.py) | chamber design (n/a) | T=10, crashes | 11.7 (torch autograd) |
| observer | mechanistic | yes | clipped_bin_moments and bookkeeping (<5%) | **selection**: v0→v4 design, R_EDGES, nq, noise chosen against orig + σ-copy truths (README:43-50, est.py:36-54) | quantiles, grid, bin moments (yes) | T=10, crashes | 3.3 |

No estimator declares the v3 attributes `ADVICE`, `SUPPORTS` or `PRECOMPUTE_FLOPS` yet. Every advice item above is
declared in the README at most. Determinism check (`determinism.out`): all 10 deterministic entries return
bit-identical estimates for rng seeds 1 and 999.

## Cross-cutting findings

1. **Incremental-cell accounting (tube, sci_laplace, sci_tao, sci_feynman, boundary).** These charge a linear cell
   2(T-t)/T, with t the step whose gate flipped. The code actually reruns the full value+tangent recurrence, which
   measured at 2.06x (tube), 1.81x (sci_laplace) and 2.23x (sci_tao) of the charged cost on orig at 16k.
   The incremental scheme is implementable exactly. Steps before t keep their gates, so h_{t-1} at the new point is
   h + Δr·dh, a 16-element update. I consider it a legitimate *idealised* count. The DRI should decide whether
   rule 2 means "as implemented" or "as implementable". If it means "as implemented", these entries' MSE ratios get
   about 2x worse. Root-finding (160 divisions per step) and the output layer (value+tangent ≈ 640 FLOPs) are
   called "negligible", but a cell that flips at t=9 is charged only 0.2 FPE = 1.2k FLOPs, so they are not
   negligible there.
2. **control_variate** makes the same kind of idealisation: COST_Q=1.6 (rb9.py:181-184) versus ≈3.3 FPE for
   the code as written (rb9.py:158-176 recomputes `(D*a)@Wo.T` and `(D*w)@Wo.T` on every one of ≤19 pieces).
3. **Rejection sampling in stratified.py / debiased_series.** `stratified.draw` (m16/stratified.py:125-130)
   draws 200k-vector batches per stratum, at least 1.4M candidates per call. Rule 5 makes input generation free,
   but this is input *selection* by a model-free stratum label, and its cost is ≈3.5x the budget at 16k.
   A pooled draw (as in tie_lines/joint.py:93-105) needs ≈3.3 candidates per sample. I recommend that rule 5
   either names stratum rejection explicitly as free, or that the shared `stratified.draw` is pooled
   (shared code, so this is for the DRI).
4. **Silent wrong answers at other (H,T).** boundary, debiased_series, tie_lines and sci_feynman do not crash
   on a (12,8) model: they build length-10 inputs and score a different quantity. They must keep the default
   `SUPPORTS=[(16,10)]`. Only sci_tao is genuinely generic: random 12x8 model, truth 0.1082, gives
   0.1107 / 0.1072 at 16k / 64k; zoo 8x6 seed0, truth 0.7395, gives 0.7427 at 16k.
5. **Advice is everywhere in the sampling and quadrature entries.** Allocation, shares or importance tables are
   fitted on `orig`. Under v3 rule 3 their orig rows should be flagged "tuned". Fitted constants are
   allocation-only (unbiased) in boundary, control_variate, debiased_series, tie_lines and great_circle
   (BETA is cross-model but fixed, so still unbiased). In the quadrature entries they also move the
   deterministic error.

## Per-estimator notes

**tube** (`m16/approaches/tube/estimator.py`, `tube.py`). Quadrature: an unscrambled Sobol rule plus a fixed
Kronecker shift (tube.py:59-66, 78-120) picks face points, and each face's exact 1-D error mass is averaged.
Random face points would give conditional MC. rng unused. Cost: `walk_side` charges `2*(T-tstart)/T` per cell
(tube.py:173) while `fwd_jvp` runs all T steps (tube.py:138-149). Prefix (tube.py:223) and blind points
(estimator.py:27) are counted. Advice: `SHARE`, `COST` (estimator.py:9-10) and `Q23/Q12` (estimator.py:13,
q23.npy/q12.npy) are per-position s.d./√cost from orig, ≈2.8e5 FPE, declared at README:91-94.
Hardcoded: `T, H = 10, 16` (tube.py:22), Φ⁷/Φ⁸ face densities and 45·8 (tube.py:38-41), 11/1024, C_RAD and
`_cos8` (dimension-specific radial law, tube.py:201-211). Crash at tube.py:183 for T=8.

**sci_laplace** (`estimator.py`, `lqt.py`, `lp.py`, reuses `../tube/tube.py`). Quadrature: fixed-seed scrambled
Sobol per stratum (lqt.py:22-27, seed 0) plus a one-node tail rule (tube.py:249-269). Cost scheme as tube;
the tail node is charged (T-t0)/T value-only. Advice: SHARE=(0.66,0.27,0.05), COST=(7.5,8.0) and K0=0.04
(estimator.py:8-10) are declared as fitted on orig (README:77-78). Hardcoded T through tube. lqt.py also uses
COMBO (360) and 45 pairs. Crash at tube.py:183.

**sci_tao** (`estimator.py`, `core.py`). Quadrature: fixed scrambled Sobol, RULE_SEED=2026 (estimator.py:16,35;
core.py:167-170). The stage-2 Neyman allocation uses the model's own stage-1 faces, and those are counted
(estimator.py:40,51), so no uncounted pilot. Cost: `cost += 2*(T-restart)/T` (core.py:133), while the loop at
core.py:121-130 recomputes all T steps. Roots and output layer are declared negligible (README:121). Advice:
none. T is taken from `model[2].shape[0]` and H from `Wh.shape[0]`, and the face tables depend on T
(core.py:24-39). This is the only generic entry.

**sci_feynman** (`estimator.py`, `feyn.py`, `core.py`). Quadrature: an unscrambled Sobol net per
configuration with a crc32-keyed shift (feyn.py:21-40). Cost: the gate-diff-based increment
`2*(T-tf)/T` (core.py:83-87) on a full recompute (core.py:74-82). The output-layer envelope (core.py:94-116,
up to 80 iterations over 10 affine logits per cell) is "counted as free" (README:74). Advice: `SHARES` 70/26/4
and `COST23, COST12 = 7.5, 6.7` (feyn.py:17-18) have no stated provenance. README:23 says "No tables fitted to
any model", but the numbers resemble tube's orig-model Neyman shares and per-face costs. The author should state
where they came from. Hardcoded: `T = 10` (core.py:19), TRIP/PAIR over range(10) (feyn.py:15-16), Φ⁷/Φ⁸
(core.py:22-23). With T=8 it runs and returns 0.069 vs truth 0.108. Time: 4.1 s profiled, 3.75 s inside
`walk_mass` over 405 calls with about 4 rows each, so the time is Python overhead, not hidden FLOPs.

**boundary** (`estimator.py`, `zones.py`, `zones2.py`, `direct.py`). Sampling: random surface points, the R zone
and the blind zone (estimator.py:21; direct.py:221-241). Cost: full first pass plus slope from k0
(zones2.py:116), continuation `2*(T-K)/T` (zones2.py:94), remainder points (zones2.py:144). Input generation
is 0.04 per vector including rejected ones (estimator.py:36-38, 45-49). The pilot is 25% of the budget and is
counted and reused (estimator.py:57-59). Advice: `DEFAULT_FRAC`, `COST0` (estimator.py:26-27) are declared
(README:44, 119). `GA, GB = 0.15, 0.1` (estimator.py:23) were tuned on orig with tune.py/tune2.py
(tune.py:3-12, 1M orig samples). These are not declared as advice and should be. Hardcoded: `T = 10`
(zones.py:20, zones2.py:9, direct.py:198), Φ⁷ and 720/8 in P_R (estimator.py:25). Runs silently at T=8.

**control_variate** (`estimator.py`, `cv.py`, `rb9.py`). Sampling: random 9-prefixes in strata (cv.py:78-86,
100-104). The last input is integrated exactly. Cost: COST_Q=1.6 plus COST_DRAW=0.1 per prefix
(cv.py:91,97; rb9.py:184). As written, `Q` does 9 steps (≈5k FLOPs) plus ≈19 pieces × (two 16×10 products
+ 10 inequalities) ≈ 13-15k, about 3.3 FPE (see finding 2). Advice: alloc.json "plain" (estimator.py:14),
a 140k-prefix pilot on orig, declared (README:94-96). Precompute: precomp.npz, declared. Hardcoded: `T = 10`,
`x[:, :9]`, target index 9 (rb9.py:124,141,167). Crash at rb9.py:55 frame for T=8.

**debiased_series** (`estimator.py`, `ds.py`, `surrogate.py`, `m16/stratified.py`). Sampling. At 16k/64k it is
exactly m16/stratified.py (estimator.py:21-23 → ds.py:79-81). Model passes are counted exactly
(stratified.py:135-140). Rejection is not counted (finding 3). At ≥128k: probe thresholds are counted
(`C`, surrogate.py:26-42), plus G_COST=0.037 per sample (a timing ratio, ds.py:25). Advice:
stratified_design.json `alloc` (orig). At ≥128k also alloc_it5.json (ds.py:51-60, a 300k orig pilot), the
128k switch and the depth (variance_gain.py on orig), W=HI0=0.6. All declared in README:106-109 or the
docstring. Hardcoded T=10 (arc_eval.T, stratified.T). Runs silently at T=8.

**tie_lines** (`estimator.py`, `joint.py`, `estimators.py`). Sampling, pooled rejection (joint.py:93-105).
Every candidate is charged GEN_COST=0.033 (joint.py:81,120). Swap evaluations are counted (joint.py:118).
Complies, and slightly over-charges now that rule 5 makes generation free. Advice: joint_design.json `alloc`,
`use_swap` from a 1e6+1e6-eval orig pilot (joint.py:56-66), declared. Hardcoded: 10 columns
(joint.py:52, 98). Runs silently at T=8. The time ratio of 32.6 is a cold JSON load. Warm it is 1.8.

**great_circle** (`estimator.py`, `gcirc.py`). Sampling: Haar frames from rng (estimator.py:31,
gcirc.py:92-99). Cost: 3 per piece (estimator.py:20,36). The README itself says the "partial" scheme is not
implemented (README:87-88). As implemented, each piece costs 2 passes, plus `_next_root` on 160
pre-activations (gcirc.py:56), 2×45 pair roots (gcirc.py:64-67), and argsort/argmax at 181 midpoints
(gcirc.py:68-77). That is ≈7 FPE as implemented, or ≈3.8 with only in-piece candidates. So the charge is
under by about 1.3x at best. These are rough, hand-counted transcendental FLOPs. Per-call model-free work:
a 512-point stratum grid per circle (estimator.py:33-34), ≈21 FPE per circle, ≈3%. Advice: BETA
(estimator.py:23-24) from 2000 orig circles (pilot_0.npz); PIECES_GUESS=250 (orig). Declared at README:89-90,
118. Hardcoded T=10 (gcirc.py:15), with the CV over T=10 strata. Crash at gcirc.py:65.

**branch_bound** (`estimator.py`, `bb2.py`, `cone.py`, `core.py`). Below 3·180·_CELL ≈ 144k FPE
(estimator.py:17,22; _CELL=(FWD+BWD+CROWNZ)/6000 ≈ 267), it evaluates the model at 180 fixed,
model-independent cell representatives. The cells' Gaussian masses are model-free, and 0.5 of the truncated
mass is added (bb2.py:36-42; estimator.py:23-25). That is a weighted average of model outputs over a fixed node
set, so it is **quadrature**. On orig at 16k/64k: estimate 0.9893 vs truth 0.9532 at cost 180. Above 144k it
runs CROWN B&B with the centre rule (mechanistic plus a centre-node rule). That path is charged by an analytic
FLOP model per cell (core.py:155-159; bb2.py:47,75) plus centre evaluations. Complies. The docstring
"(~48k passes)" (estimator.py:5) disagrees with the code's ≈144k threshold. Hardcoded: `T, D = 10, 10`
(core.py:20), 16 in the FLOP formulas, 8 eps coordinates (cone.py:167). Crash for H=12.

**rank_exact** (`estimator.py`, `rank_est.py`). Mechanistic. Gates are frozen at 7,920 canonical inputs and the
exact linear map A_c comes from a tangent pass (rank_est.py:52-59). The Sobol quadrature integrates the
*surrogate's* argmax (rank_est.py:61-64). It is not an average of model outputs, and random nodes would not give
an unbiased MC of accuracy. Cost: 11 FPE per class plus Q·2T² FLOPs (rank_est.py:60,63). Complies. Q=16 always
at leaderboard budgets: cost 91.4k exceeds the budget, which the docstring declares. Advice: none. Hardcoded
`T, H = 10, 16` (rank_est.py:18). Crash at rank_est.py:56.

**propagation** (`estimator.py`, `gsf2.py`). Mechanistic: Gaussian-sum propagation with gate enumeration. FLOPs
are counted inline (gsf2.py:43,73,104,163,191,200,206). Uncounted: R/sd/ndtr/argsort at gsf2.py:74-79 (≈B·3H²)
and np.unique sorts in `merge`. Small. Advice: eps←budget calibrated on orig costs (estimator.py:12-13),
which affects cost only. Hardcoded `T, H = 10, 16` (gsf2.py:22), the 19-dim state, and `/90.0`, `11`
(gsf2.py:244,249). Estimate on orig at 16k: 0.346 (−61 pp).

**cone_enum** (`estimator.py`, `walk.py`, `cones.py`). Mechanistic: enumerates linear regions, Chebyshev LPs,
and integrates cone masses. The integrator's randomness is model-free inside a built cone with a fixed seed
(cones.py:185). Deterministic (checked). Cost is counted in a Counter (cones.py:46,67,92,207,223;
walk.py:21,28). The LP is charged by a dense-simplex proxy `2(m+d)(d+1)·nit·4` (cones.py:67), not HiGHS's
actual work. Uncounted: A normalisation and `Wo@M` (cones.py:47-55) and SVD/Cholesky (cones.py:209,214),
≈2-3 FPE per cone. At 16k it returns 0.99999: half the budget goes to seed bisections (walk.py:17) and almost
no error mass is found. Precompute: `arc_eval.EOS` (400k model-free draws), whereas README:72 says none.
Hardcoded `T, H = 10, 16`, FWD (cones.py:17-18). Crash for H=12.

**reliability** (`estimator.py`, `form.py`, `sorm.py`, `system.py`, `rosen.py`). Mechanistic: FORM/SORM design
points via gradients, and Ditlevsen bounds via bivariate normals. Cost: g=1 and grad=3 forward passes
(form.py:74,102; sorm.py:20), ×1.3 for the Rosenblatt map, +20 per limit state for SORM (estimator.py:30-31,
53,69). Uncounted: QR+eigvalsh per limit state (sorm.py:41-47) and Owen's T (system.py:139-159), declared as
≈5% (README:90-91). Advice: the LADDER/CFG configurations were swept against test-model truths
(sweep.py:1-20, `sweep_*_all.json`). This is possibly selection on test models; the DRI should ask. At 16k
and 64k the ladder picks L0, cost 35,154. Hardcoded `T = 10` (form.py:22) and G0 = 11/1024 from 10 order
statistics (rosen.py:19-27). Crash at form.py:67.

**observer** (`estimator.py`, `est.py`, `v4.py`, `core.py`, `quad.py`, `v1.py`). Mechanistic: exact local linear
maps at designed anchors, cone-clipped Gaussian "internal noise", and a deterministic Gauss-Legendre
quadrature over the top-3 order statistics (v4.py:93-148). The Latin-hypercube permutation uses a fixed seed 0
(v4.py:97,117). Cost: 17,280 anchors × 10.5 (est.py:15,29), which matches the forward pass plus the 10-column
Jacobian (core.py:63-80, ≈51k FLOPs), +10 blind probes, + quadrature FLOPs (est.py:28). Total 214,628 at any
budget; the module does not set `FIXED_COST = True`. Advice: no fitted constants, but the design (v0→v4) and
R_EDGES/nq/noise were chosen against orig and noisy-copy truths (README:43-50; est.py:36-54). Hardcoded `T = 10`
(core.py:7), 720 triples, Φ⁷ (quad.py:111). Crash at v4.py:70.

## Recommendations for the DRI

- Classes: mechanistic = rank_exact, propagation, cone_enum, reliability, observer. Quadrature = tube,
  sci_laplace, sci_tao, sci_feynman, and branch_bound (at leaderboard budgets). Sampling = boundary,
  control_variate, debiased_series, tie_lines, great_circle.
- Decide whether the incremental "as implementable" accounting is allowed (finding 1). It changes the quadrature
  entries and control_variate by about 2x. If it is allowed, ask the authors to add the root-finding and
  output-layer FLOPs.
- Ask for `ADVICE` declarations: tube, sci_laplace, boundary (including GA/GB), control_variate,
  debiased_series, tie_lines, great_circle. Ask for provenance from sci_feynman, and for selection disclosure
  from observer and reliability.
- Ask observer to set `FIXED_COST = True`.
- Keep `SUPPORTS` at the default for all but sci_tao. boundary, debiased_series, tie_lines and sci_feynman
  would give silently wrong numbers elsewhere.

## Commands run (repo root, `source .venv/bin/activate`, `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`)

```
# time check, cold (all 15) and warm second call (selected)
for n in tube sci_tao sci_feynman sci_laplace boundary control_variate debiased_series tie_lines great_circle \
         branch_bound rank_exact propagation cone_enum reliability observer; do
  timeout 150 python review/estimator_audit/timecheck.py $n; done > review/estimator_audit/timecheck_16000.jsonl
for n in sci_feynman debiased_series tie_lines great_circle cone_enum propagation sci_tao reliability; do
  timeout 200 python review/estimator_audit/timecheck.py $n; done > review/estimator_audit/timecheck_16000_warm.jsonl
# implemented vs charged cost for the segment walkers
python review/estimator_audit/fullpass_count.py > review/estimator_audit/fullpass_count.out
# (H,T) = (12,8) random model, budget 4000
for n in ...same 15...; do timeout 90 python -W ignore review/estimator_audit/htcheck.py $n 12 8 4000; done \
  > review/estimator_audit/htcheck_H12_T8.jsonl
# determinism (rng seeds 1 vs 999, budget 16000) for the 10 deterministic entries -> determinism.out
# sci_tao generality: random 12x8 at 4k/16k/64k vs 2M-sample truth; LB._zoo(8,6,0) at 16k (inline script)
# sci_feynman profile: cProfile around est(orig, 16000) (inline script)
```
The time-check outputs include the estimate and cost of every run on orig at 16k (e.g. tube 0.9542 at 16,079;
sci_tao 0.9515 at 16,154; observer 0.9467 at 214,628). Total CPU used by this audit is under 5 minutes, single core.
