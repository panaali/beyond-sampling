# Protocols: how work is run and scored

↑ [Project README](../README.md) · [AGENTS](../AGENTS.md) · [TASKS](../TASKS.md) · [Owner Q&A](../OWNER_QA.md)

Moved verbatim from [TASKS.md](../TASKS.md) by T396 (2026-10-19) so the board only lists work. Section names and directive numbers are unchanged, so "TASKS Directive 7/8" in older notes means the directives below.

## Continuous work and negative-result review (owner instruction, 2026-10-17)

- Workers commit each bounded phase, report evidence, and select the next eligible Ready backlog item. Root or an explicitly delegated Codex worker pushes the claim before implementation. Codex workers now have standing authority to claim the next eligible, unowned Ready task after completing a phase and its handoff. Use a fresh isolated administrative worktree at latest origin/ccr-6a61f25c-wn80ii, edit/commit/push TASKS.md alone, re-check ownership after any push race, then send root the claim hash and work in the owned research folder. Never push an old research branch wholesale, force-push, edit another owner's files, or take a Blocked/DRI-only task. Root continues integrating and checking worker status; no empty polling or unbounded compute. Existing resource locks and limits remain.
- Every positive AND negative step needs independent verification appropriate to the claim: derivation/implementation controls, data and cost checks, and a different reviewer or implementation where possible.
- A failure must name its scope: implementation defect, insufficient precision/compute, a tested variant failing the metric, or a proved obstruction under explicit assumptions. Runtime/memory exhaustion is not proof a method cannot work.
- Before closing a method, record a finite, justified alternatives checklist (formulation, conditioning, representation, numerical accuracy, cost), run discriminating tests for plausible rescues, and obtain independent review. Unexamined plausible variants stay Ready/Blocked or inconclusive. A finite sweep does not establish exhaustion of every conceivable variant; broad impossibility requires a proof.
- Preserve all failures and corrections. Root maintains the review inventory and backlog; Claude remains final DRI for confirmed claims. T63/E36/R2 remains a scoped positive.

## Ladder protocol (owner directive, 2026-10-17)

Climb from the solved rung M_4,3 (E36/E37) toward M_16,10 **one axis, one step at a time**. Axes: sequence length T
(zoo: 2,3,4,5,6,8,10), neurons H (zoo: 2,3,4,5,6,8,16,32), training amount n_seqs (2^20..2^30), weight-noise sigma;
side axis: task (argmax / 2nd-argmax / argmedian). Missing sizes (H=7, 9-15; T=7, 9) are trained with
`alg_zoo/training.py` after checking our training reproduces zoo models (T86).
1. Each rung changes ONE axis by ONE step from a rung that already passes.
2. A rung passes only if the SAME frozen recipe (deductive, coefficients from weights) beats sampling (MSE ratio < 1)
   on all 5 seeds and their noisy copies, with exact or >=1e7-sample truth, and its cost is reported next to the
   region count.
3. If a rung fails: stop, diagnose exactly which new behaviour appeared, fix the recipe, re-run ALL earlier rungs,
   keep an exhaustion checklist (rule 5). Never skip a failed rung.
4. Each rung is its own board row; the next rung is unblocked only when its predecessor passes.
5. **Stay in the target regime (DRI, 2026-10-17).** M_16,10 is ~95% accurate with errors in thin near-tie bands. Zoo
   accuracy (5 seeds): H=4: T3 0.76-0.98, T4 0.48-0.60; H=5: T3 0.94-1.00, T4 0.63-0.75; H=6: T3 1.00, T4 0.60-0.86;
   H=8: T3 1.00, T4 0.97-0.99, T5 0.81-0.88; H=16: T3-T10 well trained. Small-H rungs at larger T are poorly trained
   (bulk errors) and test a different regime. Main path: H up at T=3 (4->5->6->7->8), then T up at H=8 ((8,3)->(8,4)),
   then H up toward 16 (trained 9-15 via T86) and T up at H=16. Off-regime rungs stay on record as separate results.
   **Second track (owner, 2026-10-17):** a general method must also handle poorly trained / heavily noised models with
   bulk errors (ARC's protocol includes copies whose accuracy drops to 0.45-0.77 even for M_16,10), so off-regime rungs
   ((4,4), (5,4), ...) are climbed in parallel as their own track (T116), one step at a time.

**DRI directives (Claude DRI, 2026-10-17, supersede earlier priority text)**

1. **Evidence rule.** Work counts as evidence only when its folder AND the cited commits are on `ccr-6a61f25c-wn80ii`.
   As of 2026-10-17, most Codex folders (e.g. fading_influence, future_equivalence, e46_e48_review, third_step_pairs,
   margin_k28_native, leaf_mass_review, recipe112_recovery) and every sampled Codex hash are missing from the remote.
   Every Codex lane must push its research folders and commits to `ccr-6a61f25c-wn80ii` before its next claim or
   phase. Reviews whose artifacts are unreachable stay "unverified" until pushed.
2. **Single priority queue** (replaces all older "Take tasks in this order" lists):
   (a) T195 split attribution, T194 width-16 probe, T193/T192 independent check of E46-E48 and the cost law;
   (b) T189 mechanistic coordinates. Its first phase is the cheapest decisive test: at (4,4) and (8,4), the mass NOT
   decided by the weight-derived bound must be below sampling error and grow no faster than the region count. Open to
   the first free worker;
   (c) T188: its next phase must report the decisive count (distinct future maps per cell at (4,4)/(8,4) vs cells), or
   pause until T195 reports;
   (d) T190: measure the cache hit rate on existing (8,4) cell trees in one phase, else pause;
   (e) T191 parked (off-regime);
   (f) open reviews and DRI rulings.
3. **Freeze.** The native-cost / quadrature-certificate chain (T11579-T11595, T156, T175, T180, T109, T115, T11590,
   T146, T149, T100, and T184 after its current audit) finishes its current phase, then starts no new phases. It
   refines costs on fixed small panels and does not address cost growth along T. A cost-convention ruling (T196) comes
   first.
4. **IDs.** New T-numbers continue from T196 upward (max of the T1xx stream + 1). No new T115xx IDs. Existing IDs are
   kept; duplicates (T47, T116, T164, T165) are referred to by ID plus folder.
5. **CPU.** Claude child sessions run in their own sandboxes, and Colab is separate; the 2-heavy-slot rule applies per
   machine.
6. ~~**Main route = exact enumeration made cheaper by structure (2026-10-18).**~~ **Superseded (DRI, 2026-10-18 pm): the exact route is dead for M_16,10 (INDEX H-E13: T270, region growth >= x24.5 per step, ~1e13 regions at T = 10; T316 geometry patch). The current routes are the sampling-free walk estimators (T317f; strict bar open, T356) and rare-event estimation (T340/T350/T354), with exact computation only as a component (T346 brainstorm).** Original text kept for the record: Recipe108 fails (8,5) seed 2; from T>=4 exact costs only 1.2-1.5x the recipe with ~0 error. Codex lane assignments: m16/approaches/crossover_compute/DRI_HANDOFF.md (Claude DRI reply, 2026-10-18 00:35): 34 T193 -> independent exact-mass reference / bias check; 56 T188 future-equivalent exact cells; 78 T189 undecided mass vs sampling SD; h4 independent audit of T195; 910 T197; T55 reviews T192 port/FLOP audit. Codex root dispatches; the Claude DRI sets priorities and confirms.
8. **Two bars, always both (owner, 2026-10-18).** Every estimator result reports (a) the STRICT bar: MSE ratio <= 1 vs plain sampling at equal FLOPs, and (b) the LENIENT bar: ratio <= 10, the level ARC called "roughly match" for its own M_4,3 estimators (E32: 2.4-10x). Each is reported per leaderboard group (orig, seeds, sigma .0025/.005/.01/.02) with class (mechanistic/quadrature/sampling). Current best mechanistic per LB3 group: 56 / 120 / 69 / 130 / 37 / 47 (cone_enum, rank_exact; single-run min-over-budget caveat), i.e. 4-13x short of the lenient bar. **Amendment (DRI 2026-10-18, from T274b):** LB3 scores each run at its own charged cost, so both bars are degenerate for cheap estimators: at cost C any estimate within sqrt(10 p(1-p)/C) passes the lenient bar (a constant 0.95 at 1 FPE passes all six groups; branch_bound's 5-19x is bias^2 x 180 FPE). From now on a pass is judged at ONE stated budget B in {16k, 64k, 256k} FPE with charged cost >= B/2, upper 95% cluster CI <= 10 (lenient) or <= 1 (strict) on all six groups, every step labelled deductive or inductive. At p(1-p) ~ .045 the lenient bar then needs |bias| <~ .53/.27/.13 pp at 16k/64k/256k. LB3 rows with charged cost far below their budget are restated under this rule before they are cited. branch_bound's per-cone centre evaluation and quadrature's fixed nodes are inductive under ARC's definition (Hilton talk). Final T274 claims are scored on the sealed held-out panel HP1 (T276; Directive 7 noise levels incl. abs sigma .05), not on LB3, which was used for development. **Strict-bar wording fixed (DRI 2026-10-18, from T281):** with 6 base-model clusters even plain sampling fails 'upper CI <= 1' (control uppers 1.1-5.8), so that test needs a true ratio of about 0.3-0.5. The strict bar is now: point ratio R_g <= 1 in every group, CI reported; 'strict, significant' additionally needs the upper CI <= 1. The lenient bar stays: upper CI <= 10 in every group (sampling's control passes it). **Amendment 2 (DRI 2026-10-18; Hilton talk + T281): own-cost figure of merit and an ARC-faithful panel.** ARC scores over a distribution of models ('a random seed or maybe adding noise to the weights'; MSE averaged over 'randomness over the weights') and names no budget, so a fixed deductive estimator is one (cost, MSE) point compared with sampling at that cost. For a frozen estimator with charged cost C (FPE = FLOPs/5948) and errors e_m on a panel: C_eff = max(C, 1k FPE) (the informative floor: at 1k sampling's RMSE ~.8 pp is below the ~3 pp seed-to-seed sd, so a constant estimator fails there); F = mean(e_m^2) * C_eff / mean(p_m(1-p_m)); strict F <= 1 (point, CI reported), lenient F <= 10. Equivalently, break-even budget n* = mean p(1-p) / RMS^2, and the estimator wins iff C_eff <= n*. At p ~ .92 the strict bar needs RMS <~ .84 pp at 1k FPE and .42 pp at 4k. Estimators with a resolution knob report the whole curve. The 1k-FPE floor is a project convention (Fable r2 A1: the principled 'beat the prior' floor is ~80 FPE, where a constant at the panel mean has RMS ~3 pp); report the whole (cost, MSE) curve and say which floor a claim uses. Model-free precomputed tables (cell masses, order-statistic moments) are not charged but must be listed. **Cost convention, written out (DRI 2026-10-18, from T312):** charge every operation whose inputs depend on the weights, at T196 rates (multiply-add = 2), including special-function evaluations at weight-dependent points (Phi, Phi2, exp, log: 20 FLOPs each unless the implementation is instrumented). Do not charge functions of the input law alone (order-statistic densities at fixed nodes, Gauss-Legendre nodes and weights, model-free cell masses), provided they are precomputed once, listed with their size, and identical for every model. The estimator's own forward passes are charged at their instrumented FLOPs, not at the sampling constant; the sampling side stays at 5,948 FLOPs per FPE. The primary panel is HP2 (T284: >=100 fresh retrains of the zoo config, sealed truths); HP1's noisy copies are the robustness panel; LB3 stays the development panel.
7. **Rung/claim protocol (DRI ruling 2026-10-18).** A rung passes only if the same frozen method beats sampling on all
   5 trained seeds AND on a noisy panel of >= 8 copies per seed at each of ARC absolute sigma 0.05 and leaderboard
   relative sigma 0.005 and 0.01. Truth must be exact or >= 1e8-sample MC with an independent RNG stream per model
   (no common random numbers). Cost is in T196 FLOPs ([review/cost_convention](../review/cost_convention/README.md):
   multiply-add = 2, other scalar ops = 1; op counters charged 2·ops until split). Every model and budget is reported
   individually with a CI; no minimum over budgets. Exact methods report their cost on the same noisy panel, because
   noise may multiply regions. AGENTS rule 6's arc_eval.py is superseded by leaderboard.py v3 for M_16,10 and by this
   directive for rungs. Background: [QA_2026-10-18](../review/dri_handover/QA_2026-10-18.md) #7.

**Codex manager update (2026-10-18):** User explicitly authorized submission/git push of our work to this repository; all six agents notified. Root published the five blocked prepared research/review packages through `68d6232d`, preserving scoped owner trees and cited original history. The export authorization blocker is resolved. [Current coordination and Claude reply block](../m16/approaches/crossover_compute/DRI_HANDOFF.md) records conservation checks, current dispatch, and pending DRI responses. Workers resume only eligible claimed/reviewed phases under the DRI priority queue:34 T193,56 decisive T188 counts,78 T189, h4 independent T195 evidence review,910 T197, T55 priority review. These are dispatch intentions; authoritative task ownership is the claimed row. T191 remains parked and frozen native-cost chains do not restart. Root and workers publish bounded evidence promptly using fresh remote checks; no force pushes, stale tree overwrites or silent broadening of claims. Scientific sign-off remains Claude's.

## DRI ruling on budgets (2026-10-19 17:0x)

ARC names no compute budget. Estimators may be scored at any budget B above the earlier 16k-256k FPE range, if B is
pre-registered before the sealed run and the comparison with plain sampling is at the same B (F definition unchanged).
Results at large B are reported as such, with the cost per model in dollars ([docs/COMPUTE_FRONTIER.md](COMPUTE_FRONTIER.md) item 8).

