# Owner Q&A: later updates (archive)

↑ [Owner Q&A](../../OWNER_QA.md) · [INDEX](../../INDEX.md)

Moved verbatim from [OWNER_QA.md](../../OWNER_QA.md) by T396 (2026-10-19). Each block was a "Later updates" note under the question named in its heading. These are snapshots as of their timestamps; the current number and status of every claim is in [INDEX.md](../../INDEX.md).

<a id="u1-1"></a>
## 1.1 Is there any hope for the methods we are testing?

**Later updates:**
- 00:24: T192 found that the recipe loses at (8,5) seed 2 (MSE ratio 3.4–4.6), and that exact enumeration costs only 1.2–1.5× the recipe from T=4 on.
- 00:37: the 4·T! start is not needed for exact enumeration, but it explains only about 10% of the growth ([review/chamber_split/](../../review/chamber_split/)).
- **The exact route is dead** (H-E13).
  - Region counts of 34 / 792 / 21,032 for k = 2..4 are confirmed independently.
  - T270 found at least 15.5M regions at 6 steps, so growth from step 5 to 6 is at least 24.5×. That is an exact lower bound. The 31–33× estimate is biased low by the T316 geometry bug.
- T195 pruning is negative: it saves only constant factors.
- **Lenient bar reached on retrained models:**
  - T317c/T317e score F 5.07 [3.41, 7.03] on fresh HP3 (H-L1).
  - T317f scores F 2.63 [1.93, 3.39] on fresh HP4.
  - Both fail on noisy copies.
- **The strict bar is still unsolved.** Fable's round-5 review gives about 5% odds within 3 months.

<a id="u1-2"></a>
## 1.2 Owner report: a new ARC member has a solution

**Later updates:**
- 00:42: the owner ruled "Unless we have a candidate solution, we should not contact Arc." The draft was withdrawn, and T41/T203 record the ruling.
- Hilton's talk from about a month earlier calls the challenge "completely open" (1.7), so any ARC solution would be from the last month.
- Fable round 1 put 25–35% odds on ARC's reported solution meeting the strict bar.

<a id="u1-3"></a>
## 1.3 Could we prove infeasibility for random weights?

**Later updates:**
- T232 (03:56): the trained network has as many linear pieces as random networks of the same size: 21,032 after 4 steps, against 19,000–26,000. The difference lies in what the pieces compute: 95% accuracy against about 10%.
- T271's activity-matched random control found 0.78×. The independent T280 found 0.8–1.0×, now confirmed for k ≤ 4 (H-E14).
- Hilton's talk says ARC (Mike Winer) did match sampling on randomly initialised networks. Our earlier adaptation of ARC's wide-random-MLP method (E35) was off by 41–100 pp on M_16,10.

<a id="u1-4"></a>
## 1.4 Would a 100%-accurate network be a solution ARC accepts?

**Later updates:**
- Hilton's talk: a hand-built network with about 21 neurons gets 100% (or 100 minus ε). With 16 neurons "I expect you can do better", because training gets stuck in local minima.
- The training study (T234) found trained 16-neuron networks above 95.3%; see 2.1.

<a id="u1-6"></a>
## 1.6 If ARC asked for 16 neurons and 7 steps, would we win?

**Later updates:** T277 found that the deterministic methods' rate of improvement with compute stays about at sampling's rate from 3 to 10 steps. What changes is the starting error, which grows about 1000×: the method's error is 1/1000 of sampling's at 3 steps and 1–2.5× at 10.

<a id="u1-7"></a>
## 1.7 The Hilton talk transcript

**Later updates:**
- 07:03 reframing: per the talk, ARC scores the average error over many retrained models (or noisy copies) and names no budget, so cheap budgets count. That led to the 1k-FPE floor, the "own-cost" scoring rule (Directive 8, amendment 2) and the sealed retrain panels HP2–HP4.
- 11:26: the DRI confirmed that "roughly match" does not appear in the talk. The lenient result meets our 10× bar, not ARC's stated goal.

<a id="u2-1"></a>
## 2.1 Maximum accuracy of this architecture

**Later updates:**
- The owner asked for the training study, which ran as T234 on Colab.
- A measuring tool split M_16,10's 4.69% error into blind region 0.84%, 2nd-vs-3rd near-ties 2.86%, 1st-vs-2nd near-ties 0.72% and other 0.27%.
- Interim results at 07:07:
  - With zoo settings, mean accuracy rises with data: 92.7% at 2^30 sequences, 93.8% at 2^31, 95.5% at 2^32 (best run 97.1%).
  - Large-batch runs reached 96.9% (batch 8,192, 2^33).
  - One linear learning-rate schedule run collapsed to 51%.
- Fused-kernel validation (40 vs 40 seeds, means 91.9% vs 92.1%) showed that retrain outcomes are chaotic: same-init correlation 0.11, spread about 3 pp. M_16,10 sits near the top of 40 retrains, whose best was about 96%.
- The 1,050-model grid (about 100 runs per setting) was still running at 13:07; its final stage ends about 17:05 UTC. Hilton's talk independently expects 16 neurons can beat 95.3%.

<a id="u2-2"></a>
## 2.2 Accuracy for 2 to 10 numbers

**Later updates:** the complete table, unverified, was measured with 100M samples per model at 05:22 (T234):

| Numbers | Accuracy per seed | Mean |
|---|---|---|
| 2 | 99.988–99.997% | 99.991% |
| 3 | 99.939–99.951% | 99.945% |
| 4 | 99.73–99.83% | 99.78% |
| 5 | 98.72–99.25% | 99.02% |
| 6 | 97.04–98.46% | 98.00% |
| 7 (newly trained, 3 seeds) | 97.45–97.86% | 97.65% |
| 8 | 95.76–97.63% | 96.68% |
| 9 (newly trained, 3 seeds) | 95.90–97.34% | 96.73% |
| 10 | 88.02–94.85% | 91.21% |

M_16,10 (95.31%) is a separately trained model, better than every zoo seed at 10. Accuracy falls slowly up to 9 numbers and then drops at 10. The 7 and 9 results, from only 3 seeds each, are level with 8 within noise.

<a id="u2-3"></a>
## 2.3 Did we find what each of the 16 neurons does? The owner's intuition

**Later updates:** T241 measured the neuron roles quantitatively (see 2.5), and T232/T239 tested the intuition (see 1.3 and 2.4).

<a id="u2-4"></a>
## 2.4 Experiments to prove or disprove the intuition

**Later updates:**
- **T232:** the trained network has as many pieces as random ones, so the flood of pieces is not special to the error part.
- **T239 (04:34):** the intuition is right about where the complexity is and wrong that it is random.
  - Near ties there are 26–27× more pieces per unit of probability than far from ties.
  - Error-carrying pieces multiply 2.4–3× faster per step.
  - The near/far fragmentation ratio is 44× for the trained network against 1.3–6.6× for random ones. Near ties the trained network has 358,000 pieces per unit of probability against 32,000–170,000 for random networks; far from ties it has 2,900 against 11,000–25,000.
  - Training moves pieces onto the tie walls (H-E15).
- **T239 follow-up:** wrong pieces occur only near ties or in the blind region. At (16,4), far from ties there are 2,052 correct pieces, 31 mixed and 0 wrong. 42% of the probability sits in correct pieces that never touch an error, and 10 blocks hold 72% of the probability, but each block has 1,500–4,000 pieces.
- **T280 confirmed both findings** (scoped k ≤ 4): region count 0.8–1.0× random, and concentration at near-ties about 37× against about 5× for random.
- **Experiment 4:**
  - T240 (Codex) ended as a bounded diagnostic. Its board row reports certified-correct mass of about .068 and .059 under the gap settings tested, and makes "no large easy-set/full-cost claim".
  - T279's cone certification proved only 24% of clearly separated inputs with 10k cones even at 4 steps, so it was closed.
- **Experiment 5:** not reported in this session.

<a id="u2-5"></a>
## 2.5 Experiments to find each neuron's role quantitatively

**Later updates (T241, core confirmed by an independent review):**
- **Causal map.**
  - Disabling any single neuron at the last step costs 21–67 pp, except n4 (about 4 pp, up to 27 at step 1) and n9 (nearly inert).
  - In the stricter same-structure swap, the comparators cost 48–54 pp, the clocks 24–35 and n2 27.
  - n4, n5 and n9 barely matter at the last step.
- **Formulas.** n1, n2, n4, n6, n7 and n8 fit at ≥ 99.9%, for example n7 ≈ ReLU(−13.2·last + 10.9·max + 2.0·second max). The clocks fit at 93–99% and n3/n13/n15 at 98.5–99.4%.
- **Replacement.** Agreement with the real network is 95.5% (96.6% in the review), against the 99% target. The accuracy gap was 0.11 pp in the original and 0.82 in the review, so that match is not robust. Reaching 99% would need formulas about 1,000× more precise.
- **Leftovers.** 78.5% of disagreements sit near ties or in the blind region (19.6% of inputs).
- **Recurrent test.** With the formulas taking over at a given step, agreement is 100% at steps 1–2, 99.0–99.95% at steps 3–7, and 98.3 / 97.8 / 95.5% at steps 8 / 9 / 10.
- **Corrections.**
  - The "delayed comparator" finding for n5/n9 is weaker than first stated.
  - The most damaging mode is the slower clock (8.37-step period), and a random direction in the same space does about as much damage.
  - The n3/n13 "special importance" claim was withdrawn.
- **Compression.** T260 (Codex) found that the formulas (about 50k bits) are longer than the weights themselves (about 35k bits).

<a id="u2-6"></a>
## 2.6 The right intuition, first version

**Later updates:**
- T239 and T280 sharpened point 3: the band is a trained "fingerprint", finer than a random network's near ties and coarser elsewhere (see 2.7).
- Fable round 1 found that about 14% of the error sits in a soft tail away from ties, so "errors only near ties" was overstated. Only the fully wrong pieces are confined to ties.

<a id="u2-7"></a>
## 2.7 The right intuition, in simple terms

**Later updates:**
- T280 (08:14) confirmed the region-count and tie-concentration findings (scoped to 4 steps).
- T292 found that weaker retrains differ mainly by a widening of the soft tail away from the closest ties: 60% of the model-to-model differences, spread thinly over many position patterns.
- T337 found that weight noise creates errors far from ties (see 3.12).

<a id="u3-1"></a>
## 3.1 "Better than sampling only on the clean model"

**Later updates:** on the sealed panels the fixed-point averaging method (v5q, inductive) is about equal to sampling:
- HP1 at 16k FPE: F 0.9–1.7 across groups.
- HP2 retrains at 1k–64k: F 1.06–3.5.

T271 found that the Sobol method improves at exactly sampling's rate and never pulls ahead. It is not an ARC candidate (see 3.2).

<a id="u3-2"></a>
## 3.2 The fixed set of test points

**Later updates:**
- Hilton's talk confirms this reading (see 1.7): no sampling even on a subset, and no extrapolating from observed behaviour.
- T263 lists repeated shuffled-Sobol averaging as sampling.

<a id="u3-3"></a>
## 3.3 Our solution for 4 numbers at 16 neurons

**Later updates:** INDEX T194 lists the (16,4) ratios as ≤ 0.016 in the older, pre-T196 cost units, with cells about equal to regions (0.80–0.96) and recipe 2.7–3.0e9 ops against 4.2–4.9e9 for exact enumeration.

<a id="u3-5"></a>
## 3.5 A formula over the top values and their gaps

**Later updates:**
- **T269 was negative (05:46).**
  - The exact integral is accurate to < 1e-5.
  - A per-model fit matches its own model only tautologically.
  - Formulas don't transfer: they are off by 0.4–19.5 pp on other models.
  - As a control variate the gain is only 1.15–1.45×, against 4× pre-registered.
  - Phase 2 was not run.
- **The idea's deductive descendant became the lenient result.** It walks the network exactly along the two gap directions (g12, g23) of each top-three placement:
  - T292: the top-three closure, 0.51 pp floor.
  - T300: F 6.0 on DEV2 and 9.7 on HP2, partly inductive.
  - T302: the exact walk, 21×.
  - T304: 11.7×.
  - T317 family: HP2 9.57, T317b 7.75, T317c 6.72, then HP3 5.07 (T317c–e), then **T317f HP4 F 2.63 [1.93, 3.39]**.

<a id="u3-6"></a>
## 3.6 The mechanism-based method we hoped would pass the lenient bar

**Later updates:** the probe family is **dead (verified, scoped)** (E7 in INDEX):
- T271: its error does not shrink with compute. The 0.74 → 0.25 sequence compared three versions, not three budgets.
- T274a: v5a narrowly failed the pre-registered 4k line (upper CI 12.6). The fixed-point control (v5q) passes, but it is inductive.
- Hilton's talk makes the probe borderline inductive anyway.

The lenient bar was reached instead by the walk family (see 3.5 and 3.8).

<a id="u3-7"></a>
## 3.7 Where is our hope now?

**Later updates:**
- Item 1: the probe is dead and T269 negative, but the walk family (T317) passed the lenient bar on sealed retrains: HP3 F 5.07, HP4 F 2.63.
- Item 2: the exact route is dead. T270 found growth ≥ 24.5×/step at step 6 (kill line 12).
- Item 3: T272 was a kill: freezing shifts the estimate by 2–21 pp.
- Item 4: HP1 measured ARC's σ 0.05 at 10 steps as 11–34% accuracy, close to chance.
- **Strict-bar arithmetic.**
  - T326: cost cannot drop below about 2.2k FPE. The DRI corrected the child's arithmetic: at the current error that floor gives F ≈ 1.9, not about 6, because F scales linearly with cost.
  - Fable round 5 estimates the approach tops out at F ≈ 1.2–1.7 even if every fix works (about 5% odds).
  - T333: the exact two-dimensional combination would add ≥ 13–19k FPE.
  - T339: the spread closure brought the model-to-model scatter near strict level (F ≈ 1.06 from variance alone) but exposed a bias of about +0.76 pp. **Withdrawn 14:15:** an independent check on 100 fresh DEV3 models (T343) did not replicate the scatter reduction (sd ratio .912 vs .68 on DEV2); it was specific to the development models.

<a id="u3-8"></a>
## 3.8 Do we pass the lenient bar?

**Later updates (12:33):** **T317f** (T317e plus a declared early-stop rule) scored **F 2.63 [1.93, 3.39]** on fresh sealed HP4, scored once. RMS is 0.75 pp, bias +0.44, at about 3,300 FPE; under stricter counts F is 2.67–2.98, and the plain-sampling control is about 1. Sampling still has about 2.6× lower error, so strict fails. HP4 is now spent and HP5 is held in reserve (H-L1).

<a id="u3-9"></a>
## 3.9 What are the "80 approximations"? Would ARC accept them?

**Later updates:**
- T333: every free way to combine the two walks is far worse, and the exact 2-D answer adds ≥ 13–19k FPE, so the product rule cannot be removed cheaply.
- T327: a fixed alternating spread of the 7 low inputs cuts the closure floor from 0.48 to 0.34 pp (0.26 with the reverse pattern) at no cost.
- T339 then exposed a bias of about +0.76 pp; nothing was frozen.

<a id="u3-11"></a>
## 3.11 Did our reproduction of ARC use noisy copies? How much noise breaks our models?

**Later updates:** see 3.12 for the breakdown point (between 0.1% and 0.25%).

<a id="u4-1"></a>
## 4.1 ARC's goal, our mechanistic progress, and whether we used the neuron roles

**Later updates:**
- T327's fixed spread pattern reached 0.26 pp, so T338's go line was raised to beat 0.255.
- **T338: NO GO (13:02).** The weight-derived choice scored 0.41 pp, no better than a simple "most recent inputs" rule, and it lost to the fixed spread's 0.26. Lead: the weights rank the *earliest* low inputs as most important.
- **T341 refuted the running-max reading (13:12).** Giving the closure the true running maximum overestimates accuracy. The reason the spread works is open again (speculative: a sign-alternating, period-2 recurrent mode).

<a id="u4-2"></a>
## 4.2 Why the neuron information was not that helpful

**Later updates:** T338 was NO GO and T341 refuted the running-max reading (see 4.1). So far no estimator improvement has come from neuron-level understanding.

<a id="u4-3"></a>
## 4.3 Are we closer to estimating rare bad behaviour?

**Later updates:**
- 13:05: T337 showed that noise errors are exactly these far-from-tie islands, so T340 also checks whether the walks capture them.
- 13:14: T340 is the top priority, and T344 adds certified upper bounds on the same events.
- **13:17, first preliminary results** (judged against a smaller sampling check; final billion-sample truths were due about 45 minutes later):
  - **T340:** the cheap version (with the usual shortcuts) misses the far-from-tie error islands entirely, by factors of 80 up to 10¹². A careful version that walks each line at many more points, costing 200,000–300,000 FPE, lands at 0.6–0.9× the true probability on two retrains but only 0.24–0.33× on the main model. The main model's shortfall comes from the walk's simplified treatment of the off-line directions; the closure keeps the rate within about 30%. At that cost sampling would see only a handful of events, if any, so even a factor-of-3 estimate is in the same range. Whether it beats sampling waits on the final truths.
  - **T344:** a fully rigorous guarantee is much harder than expected. Proofs work only on tiny input boxes, because the 10 steps amplify uncertainty, and the only rigorous bound so far is the obvious one, 250× to 24,000× too loose. Next: test whether exact enumeration is feasible in the far-gap area, falling back to a bound that is rigorous given the declared closure. T340's scan confirms the rare far-gap errors exist; on the main model they are about 10,000× rarer than the obvious bound allows.

<a id="u4-4"></a>
## 4.4 Questions our approach can answer better than sampling

**Later updates:** T344 (certified upper bounds on T340's events) was added at 13:14 to cover point 1 for rare events. Its first finding (13:17): rigorous bounds work only on tiny boxes, and the best so far is 250×–24,000× too loose. T340's careful walks reach 0.6–0.9× of the truth on two retrains and 0.24–0.33× on the main model, at 200k–300k FPE (preliminary; see 4.3).

<a id="u4-5"></a>
## 4.5 When will T340 be resolved?

**Later updates:** preliminary T340 and T344 results arrived at 13:17 (see 4.3). The final truths were still being computed.

<a id="u4-6"></a>
## 4.6 Rare events: the first results (update to 4.3-4.5)

**Later update to 4.6 (14:18):** the independent re-check on 6 fresh DEV3 models (T350, pre-registered, own code) did NOT replicate: the family rule won 11, tied 6, lost 13 (46% of decided cases vs the 2/3 line). The cheap variant still beats sampling on the "both gaps large" events (7 of 9 decided); the careful variant loses on the far-gap events and was 6-10x low on one model. A correctness bug was then found in the long-line walks (simultaneous gate crossings when an input crosses zero), so the careful-variant numbers (T340 B, T340c, T354) are suspect until fixed and rerun; the main accuracy estimator T317f is believed unaffected (verification pending).

**Later update (14:35), strict bar:** H-S1 marked dead (verified), scoped to the T317 paradigm (point closure + exact gap-line walks + declared combination, zoo retrains, C <= ~6k FPE), after the T356 checklist and the independent T357 re-check. The closure is not the binding floor (perfectly integrated it gives F .89 at 3.3k FPE on DEV3); the walk integration error would have to fall ~4.6x at no added cost. A lattice quadrature (the most promising untried scheme) missed by ~5x. Not a proof; other paradigms remain open.

**Later update (14:47), rare events:** T358 (20 more fresh DEV3 models, rarer thresholds) is formally a GO but substantively limited: the cheap estimator is usable only near p ~1e-4 to 1e-5 and is 5-20x low at p <~ 1e-6, where it only beats sampling because sampling returns zero. The rare-event primary criterion is now 'within 2x of the truth'. No rare-event estimator yet is both accurate and cheaper than sampling on fresh models; the accurate careful version (T340c) is being rerun after the walk-bug fix.

**Later update (15:20):** T361 reproduced Fable r6's four-term split of the strict-bar integration error and killed the one free idea (homogeneous closure: floor .762 > .45). T359 cut the accurate rare-event estimator's cost to ~.53x at unchanged accuracy. Fable's T346 brainstorm (DRI branch review/fable_t346): exact routes for the accuracy number < 1%; kept as components: exact 1-D walks, exact sensitivities, shared products, sub-cone certificates for p <= 1e-8; new tasks T362 (conditional MC along exact lines, sampling-class, rare events) and the T348 census extension (error-pattern growth, the most informative single experiment).

**Later update (15:25), rare events:** T360 (independent agent, own code, 6 fresh DEV3 models) confirms the accurate estimator for far-gap rare errors: 18/18 within 2x (medians .99-1.06), registered as ledger row H-R1 (confirmed, scoped). The interior 'both gaps large' event stays one-signed low (76% within 2x). At equal compute it still mostly loses to sampling (21 win / 22 tie / 67 loss), because it costs 1-1.5M FPE (E1) to 3.7-6.7M (E2).

**Later update (16:20):** T362 (conditional MC along exact lines, sampling-class) unbiased but gain only 1.2-4.7x (no go). T363 traced the interior event's low bias to the lows closure (56%) and the v4 position (21%); T365's spread-lows fixes reached 20/32 within 2x vs the 24/32 line (scoped negative; a real fix needs ~10-100x the cost). T364 (p ~1e-8 to 1e-10) pre-registered with an exact conditional-law truth method. T348 census interim: entropy-effective pattern growth x5.1 -> x3.2 per step by k = 9.

**Later update to 6.4 (16:36), GPU census result (T348):** on 1e9 inputs, the probability-weighted number of gate patterns grows ~x4-6 per step (all inputs and error inputs alike, slowing from ~x5-6.5 at k = 5 to ~x3.6-4.8 at k = 8), far below the x23-30 growth of raw region counts but above the ~x3 needed to revive exact counting for rare events. Late steps carry most of the pattern diversity. Raw distinct counts beyond k = 5 are limited by the sample size, not the model. A sharper answer would need ~1e10 samples (~7 GPU-hours, out of budget).

**Later update to 2.1 (18:24), training ceiling final (T234, 100 retrains per setting, unverified):** mean accuracy by training data 2^30 / 2^31 / 2^32 / 2^33 / 2^34 sequences: .911 / .936 / .938 / .944 / .952; the best single models reach ~.98 from 2^31 on and do not improve further. So the typical model keeps improving with more data (~0.6-0.8 pp per doubling) while the best models have saturated near .98. The blind-input error (~0.7%) is a floor of the recipe. Doubling the learning rate at 2^31 matches 4x more data (.9465). M_16,10 (.9531) is at the 53rd-80th percentile of the largest-data retrains, below the ~.98 ceiling. Colab: 164.4 units left; all runtimes stopped.

**Later update to 4.6 (18:50), rare events:** T367 (fair test of the near-tie fix, truth-blind selection, sealed truths) narrowly missed its bar: 6/8 within 2x (7 needed), while plain K3 was within 2x on 3/8 and missed by up to ~3e5x on dev3_s541, where every rare mistake is a three-way near-tie; the fix got s541 right (1.25, 1.35) but was 2.1-2.4x high on s547. T368 (independent agent, own code) confirmed the deep-tail extension on ordinary networks: 10/10 within 2x, so H-R1 now reads 'within ~1.4x from 1.8e-10 to 7e-7 on non-triple-tie models'. T369 showed the s547 overshoot is not quadrature (the graded rule is converged) but the closure: pinning the smaller inputs at mu(v4) and fixing v4's position inflates the near-tie islands; with exact lows and v4 position the estimate is 1.04x of the truth. So the one remaining failure mode of the rare-event estimator is that simplification, which also causes E2's low bias. A proper fix (T370, backlog) costs 10-100x more per estimate: a correctness demonstration rather than a cost win. Answer to 'are we closer to estimating rare bad behaviour?': yes for far-gap mistakes on ordinary networks (confirmed); not yet on near-tie-dominated networks (including the target at its deepest tails).

<a id="u4-8"></a>
## 4.8 Do we have several solutions for the lenient bar and the main challenge, with only ARC acceptance unknown? (owner, 10-19 06:2x)

<a id="u5-1"></a>
## 5.1 Which task needs a GPU?

**Later updates:**
- T192 produced the answer key: 95.31175% ± 0.00004 pp from 300 billion seeded samples.
- It found the (8,5) recipe failure, built the fused training kernel and ran T270 and HP1–HP5.
- At 04:14, 27.8 of 305 units were used. By 08:16, about 43 of the 115 units set aside for that phase were spent.
- The exact route it was built to test is dead (H-E13).

<a id="u5-2"></a>
## 5.2 Questions for the previous DRI

**Later updates:**
- 00:37: the subagent's chamber check answered #1. The chambers exist only for the recipe's gate-freezing rule; exact enumeration can start from 180 truth cones, and the chambers explain about 10% of the growth.
- The owner relayed the previous DRI's answers at 00:37, saved with confidence labels at [review/dri_handover/QA_2026-10-18.md](../../review/dri_handover/QA_2026-10-18.md). They establish that:
  - brute force was excluded only by the project's own "not understanding" framing, not by an ARC rule;
  - "≥ 1e17" was an extrapolation;
  - no protocol was ever fixed;
  - the ARC claim came from the owner's private message.
- The DRI then set one rung protocol, queued noisy-copy cost checks, and added leaderboard-repair and mechanism-audit rows.
- 00:41: the low exact masses came from reused random streams, not bias.

<a id="u5-3"></a>
## 5.3 What the Codex managers wrote in DRI_HANDOFF.md

**Later updates:**
- Codex later delivered T240, T260, the T261 kernel and two Astra reviews.
- It went silent from about 08:30. Astra's round-3 review never arrived, and the DRI suggested the owner check whether the Codex manager is running.

<a id="u5-4"></a>
## 5.4 Why does Google reclaim the Colab node?

**Later updates:**
- The CLI was already 0.7.4 and the machines really were gone, which rules out the CLI bug.
- The blocking keep-alive went live at about 03:45, and there were no reclaims until one A100 was lost between 07:23 and 08:12. That cost about an hour of the training grid, so progress now syncs every 8 minutes.
- At 13:07 the HP5 job was found to have silently stopped at 11:31 and was restarted.

<a id="u5-5"></a>
## 5.5 When will all these experiments be done?

**Later updates:**
- The neuron review landed at 04:56 and the 2–10 table at 05:22.
- The training ceiling was reorganised into a 1,050-model grid after the speed-ups (5.6–5.8), and its final stage is now due about 17:05 UTC.
- T240 ended as a bounded diagnostic.
- The owner's 06:56 directive ("Do not stop until we have solved the challenge") reopened the main question, which led to the lenient-bar work.

<a id="u5-6"></a>
## 5.6 Why is the training-ceiling study slow? Better GPU, CUDA or C?

**Later updates:**
- After the owner's 04:51/04:52 push, the kernel became mandatory for T192, with a Codex backup (T261) running in parallel.
- **T192 found the real bottleneck (05:13).** 90% of each step went into one badly shaped weight-gradient matmul that ran on a single core cluster. Chunking it gave 61 → 2.4 ms per step at batch 8,192 and 9.4 → 0.81 ms at the standard batch, and 0.03 ms per model per step with 40 models packed. Gradients still match to 6e-16.
- **The fused Triton kernel was built anyway.** It runs at about 0.006 ms per model per step, and 108 models at once take 0.64 ms per step. It matches the reference within about 1e-6.
- The 40-vs-40-seed comparison showed no significant accuracy difference; an apparent early gain was chance.
- Codex's independent kernel also passes on an A100 (within about 5e-7, 0.010 ms per model per step).

<a id="u5-7"></a>
## 5.7 Can one fused kernel do data generation, forward and backward?

**Later updates:** T192's Triton kernel does exactly this: data, labels, forward, loss, backward and updates in one kernel. Its correctness checks passed (05:19 and 05:41).

<a id="u5-8"></a>
## 5.8 Could one fused kernel run all the experiments?

**Later updates:**
- At 06:32 the DRI approved a 1,050-model grid in a single launch (about 100 runs per setting: 5 data sizes, learning-rate and warm-up variants, three hard-case mixes) at about 39 GPU units.
- The kernel's in-kernel accuracy matches the reference evaluator within about 0.0001.
- The grid's copies of HP2's 100 models were bit-identical to the separately trained panel.

<a id="u6-1"></a>
## 6.1 Would a different architecture make it easier?

**Later updates:** at 13:23 the owner asked to add this as a low-priority backlog item. It is **T345** on the board: 1–2-layer ReLU networks with about 432 parameters (and a 2× budget), accuracy reported, then our exact tools vs plain sampling at equal compute, with both caveats recorded. Not started.
