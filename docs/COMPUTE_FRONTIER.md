# Compute frontier: what would more compute settle?

↑ [Project README](../README.md) · [INDEX](../INDEX.md) · [STATUS](../STATUS.md) · [Owner Q&A](../OWNER_QA.md) · [Protocols](PROTOCOLS.md)

**Preliminary, order-of-magnitude estimates for a safety-case write-up (owner request, 2026-10-19).** Counts and unit
costs come from the linked INDEX rows. Totals, prices, 5-year projections and agent effort are this note's own
arithmetic and guesses: read every figure as about ±10x unless a range is given. Nothing was run for this note.

**Conventions**
- **S1, certificate:** the exact accuracy, or a rigorous upper bound on the rare-event probability, for the target
  M_16,10. **S2, ARC's bar:** beat sampling at equal compute, F = MSE·C / mean p(1-p) <= 1 strict, <= 10 lenient
  ([Directive 8](PROTOCOLS.md)).
- **COMPUTE-LIMITED:** a finite computation whose parts exist would settle S1 at cost X. **METHOD-LIMITED:** more
  compute does not help (F is a ratio at equal compute, or a bound converges too slowly). **BOTH:** S1 compute-limited,
  S2 method-limited.
- **Prices (October 2026):** CPU core-hour $0.02 spot to $0.05 on-demand [1]; A100 $1.5-3.4/h [2]; H100 $2-4.4/h [3].
  Central: $0.03/core-h, $1.5/A100-h, $2.5/H100-h. 1 FPE = 5,948 FLOPs. Compiled CPU code is assumed to sustain
  ~1e9 FLOP/s per core on LP and geometry work (T46's Python: ~4e8).
- **5 years:** GPU FLOP/s per dollar doubles every 2.1-2.5 years [4], so ÷4-5 by 2031. For CPU and fp64 work I assume
  the slower 10x per 10-16 years [5], so ÷2-3. Both are purchase-price trends.
- **Agent effort:** guessed agent-days (one child session for one day ≈ 1-5e7 processed tokens, mostly cached
  context; the repo keeps no token accounting).
- **Markers:** (u) = INDEX row unverified; ‡ = this note's own inference. IDs link to the [registry][reg] or the
  [ledger][led]; item 4's numbers are the far-gap event E1 on the target (pick = 3rd largest, gap23 >= t0).

## 1. Exact region enumeration of M_16,10
- **Why stopped:** regions grow x23, x27, x30, >= x31.2 per step for k = 2..6 (N6 >= 19,740,596; confirmed to k = 4)
  ([H-E13][led]); dead for S2 by DRI ruling.
- **Quantity:** ~2e13 (growth stays x31) to 1e14 regions at k = 10, each an LP, an outcome split and an exact 10-D
  Gaussian volume: 6.5 ms/region/core at k = 6 (estT), 0.2 s with fixed geometry, x10-100 assumed at 10-D
  ([OWNER_QA 6.3](../OWNER_QA.md#q6-3)).
- **Compute:** 3e12-8e14 core-s = 1e9-2e11 CPU-h, ~1e21-1e24 operations (OWNER_QA 6.3).
- **Cost:** $2e7-1e10 today, $7e6-5e9 in 5 years ‡.
- **Agent ‡:** 10-30 agent-days (distributed enumerator, T316 patch-B geometry, streaming triangulation [T422][reg],
  rigorous masses [H-C1][led], independent audit). Side item ‡: the patched k = 6 recount (~35x [T270][reg]) is
  ~1e3-2e3 CPU-h, ~$50, and would settle N6 and g_delta.
- **Verdict: BOTH.** S1 is finite and embarrassingly parallel; for S2 it costs 1e12-1e16x sampling's ~74k FPE for 0.1
  pp, and F = 0 meets only the letter of the bar. Confidence: low (k = 10 extrapolated from k <= 6).

## 2. T46 exact ladder, extrapolated to T = 10
- **Why stopped:** exact cost at T = 2..5 is 17 / 666 / 3.9e5 / 1.24e8 FPE; T = 6 has 2.40e7 cells (x33.3, matching
  T270), >= 1.5e9 FPE, best estimate 1.4e10 FPE = 5.1e13 FLOPs, ~35 CPU-h per model in Python ([T46][reg], u). The
  hybrid (exact high-mass cells + walk) loses to the walk alone at T = 5-6; its exact set is empty at T = 10.
- **Quantity ‡:** 2.4e7 x (31-39)^4 ≈ 2-6e13 cells; 2.1e6 FLOPs/cell at T = 6 (grew x4.5-16 per step), assumed x2-4.5
  per step → 3e7-9e8 FLOPs/cell.
- **Compute and cost ‡:** 1e21-5e22 FLOPs ≈ 3e8-1e10 CPU-h per model: $6e6-7e8 today, $2e6-3e8 in 5 years (x100 for
  a 100-model panel).
- **Verdict: BOTH** (its own cost law agrees with item 1). The hybrid is METHOD-LIMITED: cells above τ hold
  2.5-11% of the wrong mass and remove 0-4% of the walk's variance. Confidence: low.

## 3. Event-cone enumeration for a certified rare-event probability (T373)
- **Why stopped:** the event-cone count does not depend on t0 and equals the full-space count on late triples:
  >= 2.7e15 regions projected at k = 10 vs a GO line of 2e8 ([T373][reg], u).
- **Unit cost:** 16-32 LPs per cell-step; (1,2,0) did 161,705 k = 6 regions in 46 min (~17 ms each); 20-100 ms at
  k = 10 assumed ‡.
- **Compute and cost ‡:** 1.5e10-7.5e10 CPU-h: $3e8-4e9 today, $1e8-2e9 in 5 years.
- **Verdict: COMPUTE-LIMITED, superseded by item 4** (same certificate, ~1e5x fewer cells on early triples).
  Confidence: low (a lower bound).

## 4. Branch and bound for a certified bound on P(E1, t0 = 2.5)
Truth ~1.55e-10 (MC, 1.1e11 draws; upper 4.2e-10 at 1 - 1e-6) ([T400][reg]). Every one of the 720 triples carries slab
mass 5.6e-8, so all must be certified, to ~1e-12 each for a total near 1e-9 ([T398][reg]). Mass charge per k = 10 cell:
30-60 s after [T412][reg] (fast M5t, H-C1 confirmed), perhaps 3-6 s after [T415][reg].
- **4a. One early triple, (1,8,2), DELTA 1/3.** 15,116 survivors at k = 10 from ~6 CPU-h of pruning; 259,723 /
  259,723 certificates verify exactly; charge 125-250 CPU-h ([T398][reg], [T400][reg], u). **~$3-13 per triple
  today, $1-6 in 5 years. COMPUTE-LIMITED, in reach:** T416 is charging; (9,3,2), holding all 17 MC hits, is running.
  Confidence: medium.
- **4b. All 720, forward neuron splitting.** The 240 late triples (middle top position b = 6-8) keep ~all N_b regions
  through step b: >= 1e9-1e12 ([T398][reg], u); continuing at the x2-4/step seen on (1,2,0) and (1,8,2) gives
  ~1e12-1e13 to k = 10 ‡, close to full enumeration. At ~0.1 s per child cell (1,229 s for 11,683 at k = 7) plus 3-60 s
  per charged survivor: 4e7-2e10 CPU-h ‡. **$1e6-1e9 today (central ~$3e7), $4e5-5e8 in 5 years. COMPUTE-LIMITED:**
  every part is built, most independently checked; the obstruction is a count (BOTH against sampling). The DELTA .1
  designs ([T390][reg] 2.1e8, T398 9.2e7 cells, for the 41 event triples alone) hit the same wall. Confidence: low.
- **4c. GPU full-LP pruner ([T399][reg], u).** A T4 does 2.65 nodes/s (~2 cores); an A100 ~100 nodes/s ≈ 80 cores,
  ~1.6x cheaper per leaf ‡. Per triple ~6-250 A100-h (see flags): 720 early-like triples would need 4e3-2e5 A100-h
  ($6e3-6e5), but the late triples' 1e12-1e13 nodes need ~1e8-1e9 A100-h ‡ ($2e8-4e9). A CROWN-intermediate redesign
  (~100x per leaf, not built): ~$2e6-4e7 ‡. **COMPUTE-LIMITED;** no order-of-magnitude gain as built.
- **4d. Input-space CROWN ([T414][reg], u).** ~1,275 boxes/s/core, ~1e6 FLOPs/box; the charge falls only as
  boxes^-0.5, projecting 1e14-1e15 boxes per late triple to reach 1e-10. For 240 triples: 5e9-5e10 CPU-h, or
  ~7e5-7e6 H100-h at an assumed 1e7 boxes/s ‡ ($2e6-2e7), reaching only 2.4e-8 in total; ~1e-12 per triple needs
  ~5e3x more ‡ (~$1e10 or more). **METHOD-LIMITED:** CROWN is loose by a median 176 margin units; [T418][reg] (library
  alpha/beta-CROWN) is running.

## 5. Quantised nets (T352)
- **Why stopped:** no tested net is both >= .95 accurate and small-state: int4 H = 16 (accuracy .72) has 864,652,487
  distinct step-10 states in 1e9 inputs; int8 and int4 H = 64 (.95) have ~one state per input; binary nets (4.5k-12k
  states) reach .30 ([T352][reg], u).
- **What exact DP needs:** reachable states x input intervals x steps x label summaries. With ~one state per input
  that is at least item 1's region count ‡. Binary: 1e11 FLOPs ties sampling; exact panels ~1e14 FLOPs per model.
- **Compute-limited corner ‡:** int4 activations with int4 inputs (accuracy .84) have 16^10 = 1.1e12 inputs; brute
  force is ~7e15 FLOPs, ~$10, but off target.
- **Verdict: METHOD-LIMITED** (an architecture question). Agent ‡: 3-10 agent-days to seek an accurate few-state net,
  uncertain. Confidence: medium.

## 6. Deductive estimator on the all-at-once MLP (T345, T408)
- **Why stopped:** the 51-52 'both-low' units give 1.3-1.7k gate subregions per rank-order cone (sampled lower bound)
  x 10! cones ≈ 5e9 pieces ([T345][reg], u); per-case linear rules leave residual sd 2-3 and overstate P(err | case)
  2.2-3x at every conditioning depth ([T408][reg], u).
- **Count ‡:** D1w128's first layer is 128 hyperplanes through 0 in R^10; in general position that arrangement has
  2·Σ_{i<10} C(127, i) ≈ 3.8e13 regions. At 0.01-1 s per piece: 1e8-1e10 CPU-h, $2e6-5e8.
- **Verdict: BOTH.** A cheap deductive estimator needs a new idea (an exact aggregate of the ~50 both-low statistics,
  untried); fixed great-circle walks already pass the strict bar here (F .045-.37, [H-A3][led]). Agent ‡: 5-15
  agent-days. Confidence: medium.

## 7. Late-step error-pattern enumeration (T413)
- **Why stopped:** the top 1e2 / 1e4 / 1e5 / 1e6 / 1e7 k = 10 patterns carry .045 / .27 / .49 / .73 / .91 of the error
  mass (4.34e7 distinct in 1e10 inputs, coverage .943) ([T413][reg], [T395][reg], u). A free oracle (c = 1 FPE per
  pattern) gives MSE ratio .73 at 256k; realistic c >= 1e2-1e3 gives >= .987 at every ARC-scale budget.
- **Larger budgets ‡** (T413's R = q(1-q)B / (p(1-p)(B - cN))): with c = 1e3, R ≈ .31 at 1e10 FPE (N = 1e6) and .10 at
  1e11 (N = 1e7); with c = 1e5, B = 1e12-1e13. That is 6e13-6e16 FLOPs, $0.5-800, per model.
- **Verdict: BOTH, close to METHOD-LIMITED:** F < 1 only at >= 1e10 FPE per model, where sampling's sd is ~2e-6, and
  the estimator stays sampling-class. Agent ‡: 3-5 agent-days to measure c. Confidence: medium-low.

## 8. Walk-quadrature strict bar at T = 10 (T10, T380, T389, T391, E33/T397)
- **Why stopped:** on 100 retrains (DEV3) no recipe has a CI upper bound below 1. F at 16k / 64k: T10 recipe 1.15 /
  .97; T389 .74 [.51, 1.006] / 1.22; T391 .86 [.66, 1.08] / .88; E33 at executed cost 2.28 / 1.45 ([T380][reg],
  [T389][reg], [T391][reg], [T397][reg], u). At T = 5 the family wins ([H-E16][led]).
- **Does more compute help?** A larger budget lowers F = MSE x C / p(1-p) only if the error falls faster than 1/C.
  T391's RMSE halves (.193 to .097 pp) for 4x budget: pure 1/C, F flat. The T10 recipe falls x.84 per 4x; T389 rises.
  T10's gain law, gain = (1 - err)/(err·CV²·c), is a per-line variance identity ([T10][reg]). ‡ For a discontinuous
  integrand over ~9 effective dimensions, low-discrepancy rules gain at best ~B^(-1/9) in F, consistent with x.84 per
  4x; if that held, F .97 → .5 needs ~1e7 FPE per model, testable for ~$10 on a 100-model panel with 1e9-sample truths.
- **Measured (T429, 10-19 16:5x, unverified):** F FALLS with budget for both frozen recipes on DEV3 (100 models, 1e8 truths): T391 recipe .86 / .88 / .79 / .60 / .43 / .20 and T10 recipe 1.15 / .97 / .84 / .59 / .47 / .36 at 16k / 64k / 256k / 1e6 / 4e6 / 1e7 FPE; slope about x.75 per 4x budget. Strict bar (CI upper < 1) holds from 1e6 FPE on for both ([T429][reg]).
- **Verdict: COMPUTE-LIMITED (revised; was METHOD-LIMITED).** More compute per model does help: the strict bar is met at ~1e6 FPE per model, about $1e-3-1e-2 of CPU per model. Open: ARC acceptance of the class ([OWNER_QA 4.7](../OWNER_QA.md#q4-7)) and the sealed HP5 test (T431).

## 9. Interval certificates (T344)
- **Why stopped:** boxes certify only up to 1e-4 wide (|Wh| compounds over 10 steps); the best rigorous bound was
  8,200-27,500x loose ([T344][reg], u). Covering the 9-D slab at 1e-4 needs ~(5/1e-4)^9 ≈ 2e42 boxes ‡.
- **Verdict: METHOD-LIMITED,** superseded by item 4. Confidence: high.

## 10. Strict bar for the T317 paradigm (H-S1)
- **Why stopped:** dead (verified, scoped) ([H-S1][led]): the walk-level integration error must fall ~4.6x at no added
  cost; the exact joint set J2 costs ~21k FPE extra, ~17x the ~1.2k allowance ([T333][reg]); v4 at its true position
  ~4e5 FPE ([T304][reg]); T317f scores F 2.63 on sealed HP4 ([H-L1][led]).
- **Verdict: METHOD-LIMITED:** each fix adds more C than it removes MSE. Compute per estimate is negligible.
  Confidence: high.

## 11. Weight-derived bound on the walk quadrature's node-set error (T449)
- **Why stopped:** pre-registered NO-GO (u) (branch `claude/t449-walk-bound`, `m16/approaches/walk_bound/`). Lipschitz x
  transport and Koksma-Hlawka / sphere-Sobolev bounds are infinite: the walk integrand has creases and unbounded
  switch-point sensitivity where lines graze boundaries. Cell enclosures are finite, but even oracle-tight ones are
  >= 330x the actual error (2e-4 at 1e6 FPE) at 8e13 cells, and their width falls only as cells^-0.07 (dimension 9).
- **Compute to finish:** 10x at 1e6 FPE needs ~2e29-6e39 cells ‡, each >= 1 exact walk (~8 FPE ≈ 5e4 FLOPs), so
  >= 1e34 FLOPs ‡ (~3e21 CPU-h, ~$1e20 today, ÷2-3 in 5 years). An LP enclosure per cell (T398) costs 100-1000x more.
- **Verdict: METHOD-LIMITED** (rate N^{-1/9}; more compute does not help). The only route is crease-aware cells, which
  is item 1's exact-region problem. Agent effort to re-open: none useful. Confidence: high.
- **T451 re-check (u):** zeroth-order numbers reproduced, but first-order (affine) enclosures on crease-free cells are
  oracle-tight to ~2-20x the 1e6-FPE error at 8e13 cells (crease-free cells are <= .4% of cells within 10 B). For that
  family the 10x point is COMPUTE-limited at ~1e13-1e14 cells per model ‡: the item-1 region enumeration plus >= ~10 walks per
  cell for a certified affine model (~4e19-5e20 FLOPs on top), ~1e9-2e11 CPU-h, ~$3e7-6e9 today, /2-3 in 5 years. It helps S1
  only; an oracle, not a bound (`claude/t451-walk-bound-check`, `m16/approaches/walk_bound_check/`).

## Summary

| # | approach | verdict | compute needed | $ today | $ in 5 yr | extra agent effort ‡ | confidence |
|---|---|---|---|---|---|---|---|
| 1 | Exact regions (H-E13) | BOTH | 2e13-1e14 regions; 1e9-2e11 CPU-h | 2e7-1e10 | 7e6-5e9 | 10-30 d | low |
| 2 | T46 ladder to T = 10 | BOTH | 1e21-5e22 FLOPs/model | 6e6-7e8/model | 2e6-3e8 | shared with 1 | low |
| 3 | Event-cone enumeration (T373) | COMPUTE (superseded) | >= 2.7e15 regions; 1.5-7.5e10 CPU-h | 3e8-4e9 | 1e8-2e9 | 0 | low |
| 4a | B&B, one early triple (T398b, T400) | COMPUTE, in reach | 1.5e4 cells; 130-260 CPU-h | 3-13/triple | 1-6 | 1-3 d/lane | medium |
| 4b | B&B, all 720, forward splitting | COMPUTE | 1e12-1e13 cells; 4e7-2e10 CPU-h | 1e6-1e9 | 4e5-5e8 | 10-20 d | low |
| 4c | GPU full-LP B&B (T399) | COMPUTE | 1e8-1e9 A100-h | 2e8-4e9 | 4e7-1e9 | 5-10 d | low |
| 4d | Input-space CROWN (T414) | METHOD | 1e14-1e15 boxes/late triple (to 1e-10) | >= 1e10 | >= 2e9 | new verifier | medium |
| 5 | Quantised exact DP (T352) | METHOD | >= item 1; int4-input corner 7e15 FLOPs | ~10 (corner) | ~2 | 3-10 d | medium |
| 6 | All-at-once MLP (T345, T408) | BOTH | 3.8e13 pieces ‡; 1e8-1e10 CPU-h | 2e6-5e8 | 7e5-2e8 | 5-15 d | medium |
| 7 | Error-pattern hybrid (T413) | BOTH (≈ METHOD) | >= 1e10 FPE/model for R ≈ .3 | 0.5-800/model | ÷2-3 | 3-5 d | medium-low |
| 8 | Walk quadrature, T = 10 | COMPUTE (revised by T429) | strict bar from ~1e6 FPE per model | ~0.1-1 per 100-model panel | less | done; HP5 test T431 | medium |
| 9 | Interval certificates (T344) | METHOD | ~2e42 boxes | n/a | n/a | 0 | high |
| 10 | T317 strict bar (H-S1) | METHOD | J2 +21k FPE vs 1.2k allowance | ~0 | ~0 | new paradigm | high |
| 11 | Walk node-set error bound (T449) | METHOD | >= 1e29 cells, >= 1e34 FLOPs for 10x at 1e6 FPE ‡ | >= 1e20 | ÷2-3 | 0 | high |

**Reference point:** the sampling-class bound in hand (P <= 4.2e-10 at confidence 1 - 1e-6, 1.1e11 draws, [T400][reg])
cost ~6.5e14 FLOPs, ~$5 ‡. A deterministic all-triple certificate (4b) costs ~1e5-1e8x more.

## What a large budget would buy (today's prices)

Scale limits of the methods that work, and the compute to reproduce each result: [OWNER_QA 4.17](../OWNER_QA.md#q4-17).

- **$1M** (~3e7 CPU-h or ~4e5 H100-h): every cheap item many times over: per-triple certificates for all early
  triples (b <= 2 at ~$10 each; probably b = 3-5 if each stays <= 1e7 cells ‡), the patched k = 6 recount, the
  large-budget walk test, the int4-input brute force, and the T413 hybrid at 1e10-1e11 FPE on a 100-model panel.
  Probably not the all-720 certificate (central ~$3e7).
- **$100M** (~3e9 CPU-h): likely the all-720 certified bound at t0 = 2.5 (4b) and, in the central case, the exact
  accuracy of the single target model (item 1, low end $2e7); not the worst corners.
- **$10B** (~3e11 CPU-h; about a 1 GW site for 6 months, OWNER_QA 6.3): exact accuracy of M_16,10 in almost every
  scenario, the event-cone count (3) and exact enumeration of the MLP (6). It still would not buy ARC's bar, a cheap
  mechanistic estimate beating sampling at equal compute (items 2 and 5-10 are method-limited there), nor a useful
  input-space CROWN bound.
- In 5 years each figure falls ~2-3x (CPU) or ~4-5x (GPU).

## Flags for the DRI
- **T399 per-triple cost:** INDEX says ~250 A100-h per triple, the README ~6 A100-h (~4,000 for 720); they use
  different node counts (DELTA .1 vs 1/3). The INDEX row should say which.
- **Proposed backlog rows (not added; this note does not edit TASKS.md):** walk-quadrature F at 256k-1e7 FPE on DEV3
  (cheap; tests item 8's B^(-1/d) reading); a measurement of T413's per-pattern cost c.
- **No post-January-2026 ARC/AlgZoo source was read,** so nothing goes in [lit/web_log.md](../lit/web_log.md).

## Sources (hardware prices and trends only)
[1] AWS c7a.8xlarge us-east-1, 32 vCPU: on-demand $1.6422/h, spot $0.6212/h (2026-10-19), https://calculator.holori.com/aws/ec2/c7a.8xlarge/us-east-1
[2] Jarvislabs, "NVIDIA A100 GPU price in 2026": $1.49-3.43 per GPU-hour, https://jarvislabs.ai/blog/a100-price
[3] GMI Cloud H100 2026 pricing ($2.00 PCIe, $2.40 SXM), https://www.gmicloud.ai/blog/2026-cost-of-renting-or-uying-nvidia-h100-gpus-for-data-centers ; Spheron ($2.64/h), https://www.spheron.network/blog/h100-price-per-hour-2026/
[4] Epoch AI, "Trends in GPU price-performance" (2022): FLOP/s per $ doubles every 2.46 yr (all GPUs), 2.07 yr (ML GPUs), https://epoch.ai/blog/trends-in-gpu-price-performance
[5] AI Impacts, "Recent trend in the cost of computing" (2017): cheapest-hardware FLOPS/$ ~10x per 10-16 years, https://aiimpacts.org/recent-trend-in-the-cost-of-computing/

[reg]: ../INDEX.md#3-experiment-registry
[led]: ../INDEX.md#4-hypothesis-ledger

## 12. White-Box Phase 2 public cumulant baselines (T473 → T474)

**T474 update (2026-10-21):** the memory obstacle is resolved for a row-blocked V25 port, not yet for the strongest public submissions. [INDEX T474](../INDEX.md) records measured memory/cost and the public-reference result. Exact seed reconstruction was verified against stored weights; projected public reference columns avoid regenerating expensive truths. The older estimates below describe the alternative of generating new independent references, not compute now needed for this completed four-row reproduction. T475 continues with stronger public ports and cost reduction.

- **Why the next phase is bounded:** the completed first-layer CV pilot is
  method-limited against covariance propagation on its two full-size cases
  ([INDEX T473](../INDEX.md)); more IID samples alone do not establish a better
  cost-adjusted method. Public cumulant implementations are much stronger on
  their authors' evaluations, but the reviewed full-width V29/logxio code reports
  memory above our 2 GB/process cap. No full-width public baseline was run here.
- **What could finish a useful comparison:** reproduce original small-width
  predictions and costs; profile V25 (which omits recursive Strassen pools), or
  stream source-memory contractions without changing predictions. Then evaluate
  a frozen baseline on ~10 separate full-width networks with a sufficiently
  precise independent reference. A pinned public answer-key dataset would avoid
  regenerating truths but requires a separate storage/download decision.
- **Compute estimate, not measured:** a V29-like estimator costs about
  `0.25 * 2^41 ≈ 5.5e11` metered FLOPs/model. Ten models are ~5.5e12 FLOPs.
  The difficult reference is ~1e8 samples/model: ~3.36e15 forward FLOPs/model,
  or ~3.36e16 across ten. At an assumed sustained 0.1–1 TFLOP/s per CPU core,
  this is roughly **9–93 CPU-h** (allow **10–100 CPU-h** with overhead).
  This precision target is inferred from T473's measured activation variance,
  not a guarantee of resolving small differences between the best methods.
- **Cost and effort scenarios:** using this page's existing $0.03/core-hour
  convention, not a fresh price quote, the reference would cost **$0.30–3** now;
  under its assumed 2–3× CPU price/performance improvement, **$0.10–1.50** in
  five years. Baseline estimation itself is much smaller. Memory-safe transport,
  parity tests, reference handling and an independent review are roughly
  **0.5–2 agent-days**, highly uncertain. More tokens cannot replace a precise
  answer key or sound cost accounting.
- **Verdict:** resource/engineering-limited for reproducing a strong baseline;
  method-limited for turning the tested simple CV into a competitive solution.
  More compute helps validate or falsify a new cumulant variant; it does not
  itself imply a winning estimator. A low-memory V25 path is plausible but
  unmeasured, not an exemption from the memory limit.


## White-Box exact-line/circle transfer: deadline-limited screen (T477, 21 October)

The radial arithmetic and traversal screen are in [T477](../m16/approaches/whitebox_lines/README.md); this is not a family-wide negative. A bounded exact-primitive and gate-count pilot is estimated at 2–6 agent-hours and 0.1–2 CPU-hours before any useful full-network estimator is established. At an illustrative $0.05/CPU-hour this is under $0.10 compute today, and under $0.02 if unit compute becomes five times cheaper in five years; these are scenarios, not current vendor quotes. Agent engineering effort dominates. More compute alone does not establish a favorable variance-versus-cost tradeoff or a compact incremental representation. The sprint stopped this avenue for deadline/engineering reasons, not exhausted hardware.

### White-Box deadline follow-ups (T481/T478, 21 October)

The sprint ended for the submission deadline, not local memory exhaustion. Layer/term attribution and sensitivity allocation remain untested beyond the saved rank/pruning sweeps. Rough next bounded study: 2–8 agent-hours plus 1–4 CPU-hours for 100–400 current-size runs, about 1e14–4e14 instrumented FLOPs depending on variant. At the illustrative $0.05/core-hour assumption this is $0.05–$0.20 today; a hypothetical 5x price improvement gives $0.01–$0.04 in five years, not a vendor forecast. More compute helps paired statistical comparisons but does not establish a better closure or overcome remote RPC latency by itself. Formal attribution and independent checks are the main agent-effort requirements.
