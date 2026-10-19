# T373: exact enumeration inside a rare event's own cone (kill test)

↑ [TASKS](../../../TASKS.md) · [INDEX](../../../INDEX.md) · spec: `review/fable_t346/README.md` section 3.3
(branch `origin/claude/project-handoff-priorities-zpzfm8`, read as a past record only) · T344 section of
`m16/approaches/r1_cont/README.md` (branch `origin/claude/t218-tie-tail-labels`)

## Question
Can the far-gap rare event E(t0) = {the network picks the position of the 3rd-largest input, and g23 = x_(2) - x_(3) >= t0}
be given a **rigorous** probability (exact, or a certified upper bound) by enumerating the exact activation regions
only inside the event's own cone, at a cost near plain sampling? This is the kill test for that idea (Fable T346 idea
E, re-scoped).

## Pre-registration (written and pushed before any test count is run; 2026-10-18)

**Model.** orig = M_16,10 (`alg_zoo.example_2nd_argmax`, weights via `m16/common.py`, float64). t0 = 1.5.

**Cone.** For a position triple (p1, p2, p3) the true event cone is K = {x_p1 >= x_p2 >= x_p3 + t0, x_p3 >= x_l for
every low l} (no proxy: g12 and all low gaps are free).

**A remark made before running (it predicts the result).** The network has no biases, so every activation region
is a polyhedral cone C. C meets K in a set of positive volume iff C meets the homogeneous ordering cone
O = {x_p1 >= x_p2 >= x_p3 >= every low} in positive volume: a generic interior point of C ∩ O has g23 > 0, and
scaling it by a large positive factor puts it in K. So **the exact region count inside K does not depend on t0**,
and K's count equals that of O, which leaves the 7 lows unordered. Consequences:
- I count on O (homogeneous, with T344's [-1,1]^10 box only as LP normalisation). Using the affine K with T344's box
  [-1,1]^10 would be wrong: the box makes g23 >= 1.5 cut cones by a scale artefact. As a check that could fail, I
  also count the affine K with the box [-1e4, 1e4]^10 for k <= 4 on two triples and require identical counts.
- Every region of the full space meets at least one of the 720 ordering cones, so the sum over all 720 triples of
  count_k is >= N_k (T270: N_6 >= 19,740,596). The scaled 10-triple sum should therefore be >= ~2e7 at k = 6 (up to
  sampling noise of the 10 triples), and the GO line N10 <= 2e8 needs an average growth 5->6 below ~1.8x.
  **Prediction: STOP**, with projected N10 ~1e9-1e11.

**Code.** `cone_count_t344.py` is T344's `cone_count.py` copied verbatim (sha256
`0b87cd21828cfc7b9ceb9cfec0efec17cb173bfc49689cd24224c6d185dda801`, from origin/claude/t218-tie-tail-labels).
`cone_count.py` here is that file with only these changes, each listed in its header:
1. `cone()` returns O (or the affine K for the check above);
2. the triple list is the pre-registered random one and the model is orig only;
3. an optional speed-up: an interior point per cell decides which side of each neuron's hyperplane is
   non-empty, so one LP per neuron is solved instead of two. It must reproduce the unmodified algorithm's counts
   exactly for k <= 4 on two triples, otherwise it is not used;
4. the cap is on distinct regions (2e5), checked as in T344 (the step is marked incomplete and counting stops for
   that triple when exceeded). A capped count is a lower bound.

**Triples.** 10 distinct triples from `np.random.default_rng(373)` (`rng.permutation(10)[:3]` repeated, duplicates
skipped): (1,2,0), (9,7,8), (8,9,6), (2,8,7), (5,2,7), (2,8,5), (1,7,8), (1,0,8), (4,1,3), (0,4,2).

**Steps.** k = 1..6 (k = 2..6 reported). Cap 2e5 regions per triple. At most 2 processes, `ulimit -v 2500000`.
Total compute budget 8 CPU-h; if exceeded, the remaining triples are reported as censored at their last complete k.

**Projection.** N10 = 72 x sum over the 10 triples of count_6 x (count_6 / count_5)^4. For a triple capped at
k = 6, its count_6 and growth are lower bounds, so its term is a lower bound.

**GO** iff projected N10 <= 2e8 AND max / median over the 10 per-triple projected terms count_6 x g^4 <= 10.
**STOP** otherwise (scoped negative, with the AGENTS rule-5 exhaustion checklist). If a capped triple makes the
verdict depend on the cap, the verdict is reported as "STOP (lower bound)" only if the lower bound already
exceeds 2e8; otherwise "inconclusive (censored)".

**Event-feasible fraction at k = 6.** The event is decided at step 10, so feasibility at k = 6 is not an LP on the
k = 6 cell. I report a **sampled lower bound**: draw points in K by a proposal (lows iid N(0,1); x_p3 = max(lows) +
Exp(1); x_p2 = x_p3 + 1.5 + Exp(1); x_p1 = x_p2 + Exp(1)), run the network, keep those where it picks p3, and count
the distinct 6-step gate prefixes among them, divided by count_6. The same samples (all of them, not only event
points) validate the enumeration: **every sampled 6-step prefix must be among the enumerated regions** (any miss is
a bug).

## Method
- `cone_count.py`: T344's LP cone splitter (the verbatim copy is `cone_count_t344.py`), changed only as listed in its
  header: the true ordering cone O, the pre-registered triples, an interior-point speed-up (one LP per neuron
  instead of two), the cap, and (post-launch, memory only) last-step cells keep only their history.
- `event_pass.py`: event-point search with T340's e_p3 line walk (post-hoc, below).
- `analyze.py`: summary helper. Logs are in `logs/`, raw counts in `results_*.json`.
- Run as `cd m16/approaches/event_cone; ulimit -v 2500000; python cone_count.py KMAX i,j,... [--events]`.

## Status
**Done: STOP** (scoped negative, unverified). Post-hoc diagnostic on the event-carrying triples: see below.

## Results

### Checks (all passed, except the affine box at 1e4, explained)
- **Speed-up == T344's two-LP rule** [measured]: identical counts for k <= 4 on (1,2,0) and (4,1,3): 2/20/83/822
  and 2/20/212/1,676 (`logs/val_fast.log`, `logs/val_slow.log`).
- **t0 does not change the counts** [measured]. On the affine K with box [-1e4, 1e4]^10, the k = 3 counts are 82 and 209,
  against 83 and 212 on O. The missing regions are exactly those whose max g23 over the unit box is below
  1.5/1e4: 4.8e-5 for one, and 1.4e-5, 2.6e-5, 5.6e-5 for the others. With box 1e6 the affine counts equal O's
  (83, 212; `logs/val_aff1e6.log`).
  So the 1e4-box check failed as pre-registered, and the failure is the predicted scale artefact, not a
  discrepancy. The k = 4 affine check at box 1e6 was not run.
- **All-late triples equal the full space** [measured]. (9,7,8) and (8,9,6) give 34 / 792 / 21,032 at k = 2..4, the
  confirmed full-space N_k of H-E13. This is forced: when the three positions are all >= k, the cone's projection
  onto x_0..x_{k-1} is all of R^k.
  More generally, count_k depends only on the order that the cone induces on positions 0..k-1. Three triples
  confirm this: (2,8,7), (2,8,5) and (5,2,7) all give 34 / 414 / 5,616 / 129,116.
- **Enumeration completeness** [measured]: on (1,2,0), every 6-step gate prefix of 2e6 proposal samples in K is among
  the 161,705 enumerated regions (0 missing; also 0 missing at k = 3).

### Counts table (orig, true event cone, regions = cells)

| triple | k=2 | k=3 | k=4 | k=5 | k=6 | growth 4->5 | growth 5->6 |
|---|---|---|---|---|---|---|---|
| (1,2,0) | 20 | 83 | 822 | 12,434 | **161,705** | 15.1 | **13.0** |
| (9,7,8) | 34 | 792 | 21,032 | = N5 = 632,394 * | = N6 >= 19,740,596 * | 30.1 | >= 31.2 * |
| (8,9,6) | 34 | 792 | 21,032 | = N5 * | = N6 * | 30.1 | >= 31.2 * |
| (2,8,7) | 34 | 414 | 5,616 | 129,116 | censored | 23.0 | - |
| (5,2,7) | 34 | 414 | 5,616 | 129,116 | censored | 23.0 | - |
| (2,8,5) | 34 | 414 | 5,616 | 129,116 | censored | 23.0 | - |
| (1,7,8) | 20 | 212 | 3,994 | 77,688 | censored | 19.5 | - |
| (1,0,8) | 20 | 158 | 1,985 | 29,133 | censored | 14.7 | - |
| (4,1,3) | 20 | 212 | 1,676 | 27,637 | censored | 16.5 | - |
| (0,4,2) | 16 | 131 | 1,599 | 11,986 | censored | 7.5 | - |

`*` Not enumerated to the cap here. By the projection identity above (verified at k = 2..4), they equal the
full-space counts: N5 = 632,394 (T192, recomputed independently, H-E13) and N6 >= 19,740,596 (T270 final, an exact
lower bound, unverified). The first launch of these two triples was stopped at k = 5. It was heading to the 2e5 cap
at 1.2 GB per process (`logs/run*_killed.log`), which would only have returned ">= 2e5", a weaker figure than the
known one.
"censored": the k = 6 step was not run. The pre-registered 8 CPU-h budget ran out after a container restart killed
the first k = 6 launch (`logs/runB.log` holds (5,2,7) to k = 5).

### Projection and verdict
- **Pre-registered rule with lower bounds** [measured counts, projection inferred]. Each censored triple contributes
  at least count_5. The all-late triples contribute N6 x (N6/N5)^4 >= 1.87e13 each. So
  **N10 >= 72 x (4.6e9 + 2 x 1.87e13 + ...) = 2.7e15**, against the GO line 2e8.
  - Without the all-late triples (that is, without T270's number), (1,2,0) alone gives 72 x 161,705 x 13.0^4
    = **3.3e11**. Its unscaled term, 4.6e9, already exceeds 2e8.
- **Heterogeneity** [inferred; the surrogate is not pre-registered]. With growth 5->6 taken equal to growth 4->5 for the
  censored triples (on the two triples where both are measured, 15.1 vs 13.0 and 30.1 vs 31.2), the per-triple terms
  run from 2.8e8 for (0,4,2) to 1.9e13 for the all-late triples. **max/median = 36** (median 5.2e11), against the
  line 10. Both GO conditions fail.
- **Verdict: STOP** (the pre-registered "STOP (lower bound)" case). The brainstorm predicted N10 ~ 1e9-1e10; the
  measured lower bound is 5-6 orders above that. The prediction made in the pre-registration (sum >= N6, so growth
  5->6 must be < 1.8 for GO) holds: the measured growth 5->6 is 13-31.
- **Event-feasible fraction at k = 6** [measured, lower bound]: **0** on (1,2,0), the only triple with a complete
  k = 6. The pre-registered proposal found no event points in 2e6 samples. The post-hoc line walk (below) also
  finds none on any of the 10 random triples. The fraction is unbounded above by this test; an exact value needs
  the full 10-step enumeration. On the event-carrying triples (post-hoc) the fraction is 0.3-0.7% at k = 5 and
  <= 0.1-0.4% at k = 6, below the 1e-2 line of the spec.

### Why the true cone is so much larger than T344's proxy [inferred]
- T344's proxy (g23 >= 2 g12, and g23 >= 2 x every low gap) is a strict sub-cone, and it pins the lows near x_p3.
- With no biases, the true event cone has the same regions as the plain ordering cone, which leaves the 7 lows'
  order free.
- The growth that T344 saw fall to 2-3x per step in the proxy stays at 7.5-31x per step here: the lows' unread
  ordering keeps fragmenting the regions.

### Post-hoc diagnostic: the event-carrying triples (requested by the DRI, NOT part of the verdict)
**Where the event lives** [measured; `event_pass.py`, `logs/event_pass_global.log`]. Method: T340's e_p3 line walk
with 20 grid points per line on (v4, v2 - 1.5), random rank-to-position assignment, 4e7 line points.
- 33,917 line points are events (pick = p3, g23 > 1.5), spread over only **41 of 720 triples**.
- Six triples hold 89% of the hits: (1,8,2) 7,984; (8,1,2) 7,220; (3,9,2) 4,300; (9,3,2) 3,753; (0,9,7) 3,554;
  (9,0,7) 3,368. Next come (2,9,3) 1,217 and (9,2,3) 1,066, then a tail.
- Line points are not probability-weighted, so these shares describe support, not mass.
- None of the 10 random pre-registered triples has an event point in 4e6 line points each (`logs/event_pass_triples.log`).

**Region counts in two event triples** [measured]. The pairs are equal by claim 3: (8,1,2) = (1,8,2) for k <= 7, and
(9,3,2) = (3,9,2) and (0,9,7) = (9,0,7) for k <= 9.

| triple | k=2..5 | k=6 | growth 5->6 | distinct event prefixes k=5 / k=6 (sampled lower bounds) |
|---|---|---|---|---|
| (1,8,2) | 20 / 83 / 1,272 / 21,869 | >= 200,010 (cap) | >= 9.1 | 62 / 209 (from 545k event points) |
| (3,9,2) | 34 / 414 / 5,315 / 64,983 | >= 200,001 (cap) | >= 3.1 | 443 / 845 (277k) |
| (9,0,7) | 16 / 251 / 4,220 / 82,184 | not run | (4->5: 19.5) | 262 / 556 (253k) |

**Projection restricted to the six event triples** [inferred]. Using the pre-registered rule with lower bounds:
- 2 x (2e5 x 9.1^4) + 2 x (2e5 x 3.1^4) + 2 x 82,184 >= **2.8e9 regions at step 10**, above the 2e8 line.
- An estimate restricted to these triples is not exact in any case: 35 more triples carry event points.

**The event itself grows slowly** [measured lower bounds, inferred reading].
- The distinct event prefixes grow x3.4, x1.9 and x2.1 from k = 5 to k = 6. The event-feasible fraction is 0.3-0.7%
  at k = 5 (62/21,869, 443/64,983, 262/82,184) and <= 0.1-0.4% at k = 6.
- So the slow ~2-3x growth that the brainstorm hoped for is real, but it belongs to the event's own regions. The
  explosion is in the regions of the cone that the event shares with everything else.
- An enumeration that expands only event-feasible prefixes would need ~1e3-1e4 regions per triple at k = 10, if
  x2-3.4 holds for 4 more steps [speculative].
- **What blocks it is rigour.** Pruning a non-event prefix at step k needs a proof that no continuation over the
  remaining 10 - k steps picks p3, i.e. an outer relaxation (an LP/triangle relaxation of the ReLUs over the remaining
  steps, on that prefix's cone). Two things are untested:
  - whether such relaxations are tight enough on whole prefix cones (T344's interval bounds failed on 1e-4 boxes,
    but those were IBP on boxes, not LP relaxations on cones);
  - whether the 679 triples with no observed hits can be excluded the same way at the cone level.
  This is proposed as a backlog item (below), not run.

**Cost of a certificate along the pre-registered route** (the DRI's question). Each of the ~714 non-event triples must
be shown to carry no event mass.
- With exact enumeration, that is one LP per step-10 region (~30 FPE each) over every region of every triple. That
  is the same 1e11-1e15 count as above: cone-level LPs cannot decide it, because the step-10 pick is not linear on a
  cone.
- The only cheaper route is the relaxation-based pruning just described.


## Compute accounting
- Model evaluations are LP solves (HiGHS via scipy), not forward passes. Each cell-step solves 16-32 LPs in 10
  variables.
- CPU time: about 8 CPU-h in all.
  - Validation: about 0.1 CPU-h.
  - First launch, killed at k = 5: 1.4 CPU-h.
  - Second launch: (1,2,0) took 0.82 CPU-h (k = 6 alone 46 min). (2,8,7) and (5,2,7) were lost in the container restart.
  - k = 5 runs of the 7 censored and 2 event triples: about 3.4 CPU-h.
  - Post-hoc k = 6 diagnostic: 2.2 CPU-h ((1,8,2) 0.99 h, (3,9,2) 1.25 h), above the 1-2 CPU-h asked.
  - Event passes: about 0.4 CPU-h.
- Forward passes: about 1e8 line-walk and proposal points in the event checks (numpy float64).

## Claims
1. [measured] Exact region counts inside the true event cone, k = 2..6, as tabulated (k = 6 only for (1,2,0)).
2. [measured] The count is independent of t0. Equivalently, it equals the ordering-cone count; this also holds
   numerically up to the box artefact. [inferred, proved in the pre-registration] It holds for every t0 > 0.
3. [measured on 5 triples; proved] count_k depends only on the order that the cone induces on positions 0..k-1.
   Triples whose positions are all >= k reproduce the full-space N_k.
4. [inferred] Projected N10 >= 2.7e15 by the pre-registered rule, or >= 3.3e11 without using T270's number.
   The heterogeneity is about 36x.
5. [measured] Far-gap rank-3 events at t0 = 1.5 concentrate on a few position triples: 41 of 720 have line hits,
   and 6 hold 89% (post-hoc, unweighted).
6. [measured lower bounds; inferred reading] Inside the event triples, the distinct event prefixes number in the
   hundreds and grow x1.9-3.4 from k = 5 to 6, while the cone's regions grow >= x3-9 and number >= 2e5.
   **[Caveat added 2026-10-19 (T398)]** The line-walk counts are lower bounds only.
   - Hit-and-run finds event points in 47 of 273 benchmark cells at k = 4, against 35 found by the line walk.
   - T399's exact branch and bound finds witnesses in 4 of 9 probe-negative cells (unverified), on the normalisation
     boundary g23 = 0.1 ||x||_inf.
   - So the event-FEASIBLE set is much larger, and the "slow growth" reading holds at most for the mass-carrying
     part of the event.

## Proposed backlog item (for the DRI; not added to TASKS.md by me)
**Event-guided branch and bound for a certified upper bound on the far-gap event.**
- Method: enumerate prefixes inside each triple's cone, but prune a prefix as soon as an LP relaxation of the
  remaining steps (ReLU triangle relaxation on the prefix cone) proves logit_p3 < max other logits. Charge unpruned
  step-10 regions their exact event mass.
- Why it could work: the event prefixes grow only x2-3.4 per step (claim 6).
- Kill test: on (1,8,2), report the fraction of k = 6 prefixes pruned by the relaxation. GO if >= 99% are pruned
  and the unpruned count stays within 3x of the event-prefix count.
- Also: a cone-level relaxation on the 679 triples with no observed hits. The certificate needs every one of them
  pruned at small k.
- Pointers: this README, T311 (CROWN along lines), T344 (IBP failure on boxes).

## Alternatives not ruled out
- **Extrapolation.** Growth could fall sharply after k = 6. It does in T344's proxy, once the top values are read.
  Here, though, k = 6 growth is 13x even for (1,2,0), whose top three are all read by step 3. A collapse to < 1.8x per
  step at k >= 7 would sit badly with T348's probability-weighted census, which shows x3.6-5 growth per step
  through k = 8 in the full space. That is not a strict bound on region-count growth. Not excluded by a count.
- **Censored triples.** Their k = 6 counts are not measured; the verdict does not depend on them (claim 4).
- **T270's N6** is from estT's geometry (T316 defect, undercounts) and unverified. It enters only the all-late terms,
  and only as a lower bound.
- **Region merging.** Regions with the same affine map, or the same eventual decision, could be merged. That is a
  different method (H-E13's untried list) and is not tested here.

## Failure modes
- The LP tolerance (1e-9) decides thin slivers. It was not varied here; T344's choice was kept.
- Memory: about 6 KB per stored cell. 2e5 stored cells approach the 2 GB-per-process limit.
- The pre-registered proposal does not reach the event at all on orig, so the event-fraction measurement needs the
  line walk.

## Exhaustion checklist (AGENTS rule 5)
Tried:
- (a) T344's proxy cone (T344, killed against the 1e6 line).
- (b) The true event cone, 10 random triples, k <= 6 (this test).
- (c) The t0-scaled affine cone with a large box (equal counts).
- (d) Event-carrying triples only (post-hoc diagnostic, below).

Not tried, with why:
- (i) Region merging by affine map, or by decision, across histories: a different method, H-E13's open line.
- (ii) Enumerating backwards from the decision (cone_enum's C idea): T346 idea C, judged by T348's census
  (backward branching about 4.5k at j = 1); not in this budget.
- (d') The event-guided branch and bound (proposed above): the most promising untried variant; not run.
- (iii) Coarser certificates on the non-event triples: whole-triple LP infeasibility at the level of the cone, not of
  each region. For a certificate, each of the ~714 other triples must be shown to carry no event mass (or a bounded
  one). The event is "logit_p3 >= every other logit at step 10", which is not linear on the cone. So a cone-level LP
  cannot decide it, and the only exact route is an LP per step-10 region (~30 FPE each) over that triple's regions,
  i.e. the same 1e11-1e15 count. A Lipschitz or interval argument fails at <= 1e-4 widths (T344).
- (iv) A different t0: the counts are t0-independent (claim 2), so no t0 helps the count. Only the event mass changes.
- (v) Other models (zoo seeds): not run. The full-space growth is similar (H-E13), and claim 3 makes the all-late
  triples equal to N_k for any model.
- (vi) Tighter LP tolerance: not run. It can only add slivers, i.e. raise the counts.

---

# T390: event-guided branch and bound (kill test), pre-registration (2026-10-19, before running)
**Question.** The event's own prefixes grow only x1.9-3.4 per step (T373 claim 6). Can a branch and bound expand only
those, pruning every other prefix *with a proof*, at <= ~1e7 regions at step 10?

**Pruning test (`relax.py`).**
- A prefix cell at step k is a cone {G x >= 0} with h_k = A x on it.
- Relaxation R_k:
  - x in [-1,1]^10, G x >= 0, and g23 >= DELTA = 0.1;
  - steps k+1..10 with each ReLU replaced by its triangle relaxation (pre-activation bounds by LP, stable neurons
    exact);
  - the event "logit_p3 >= every other logit at step 10", with slack 1e-7.
- The cell is **pruned iff R_k is LP-infeasible.** Since R_k contains every event point of the cell (after scaling),
  a pruned cell has no event point there.
- **The normalisation is rigorous.** The network is positively homogeneous and the cone is scale-invariant, so the
  only event points lost are those with g23 < 0.1 ||x||_inf. With g23 >= 1.5 these have ||x||_inf > 15, and the
  Gaussian mass of that is <= 20 Phi(-15) < 1e-49.
- Not rigorous yet: floating-point LP infeasibility. A certificate would have to verify each Farkas ray in exact
  arithmetic. That is not done in the kill test.

**Branch and bound.** Enumerate each step exactly (T344/T373 splitter); test every new cell with R_k; expand only the
survivors. Starting at k = 1, up to k = 6.

**Kill test.**
- Triple (1,8,2), the top event triple; then (3,9,2), the event triple with the most event prefixes, if the budget
  allows.
- Report survivors S_k for k = 1..6, the pruned fraction at k = 6 (>= 1 - S6/2e5, since N6 >= 2e5 for (1,8,2)), and
  the cost per cell.
- **Validation (could fail):** every k-prefix of the line-walk event points (T373 event pass, regenerated) must be
  among the survivors. Any pruned event prefix = a bug or an unsound relaxation, and the test is void.
- **Projection:** N10 = 41 x max over the measured triples of S6 x (S6/S5)^4.
  **GO iff N10 <= 1e7.** STOP otherwise.
- Side measurement (no verdict): S_k on 2 triples without event hits, (1,2,0) and (4,1,3). A certificate needs those
  to die out, since there are 679 such triples.
- Caps: 2e5 survivors per step, 6 CPU-h in total, 2 processes, `ulimit -v 2500000`.

**What it would certify if GO.** An **upper bound**: P(E) <= the sum over surviving step-10 regions of their event
mass (computed exactly, or bounded by the region mass) + 1e-49. Every pruned branch must be proved event-free, which
needs the Farkas rays verified exactly. It is exact (rather than a bound) only if each surviving region's event
polytope mass is integrated exactly. An "estimate" (survivors only on the 41 triples with hits) is not a certificate.

## T390 results (2026-10-19)
**Kill test on (1,8,2)** [measured; `bnb.py`, `relax.py`, `logs/bnb_182.log`, `results_bnb_1-8-2.json`]. Event
prefixes come from 276,120 line-walk event points.

| k | children tested | survivors | pruned by the g23 normalisation | pruned by the relaxation | event prefixes (pruned) |
|---|---|---|---|---|---|
| 1 | 2 | 2 | 0 | 0 | 2 (0) |
| 2 | 20 | 19 | 1 | 0 | 10 (0) |
| 3 | 80 | 55 | 25 | 0 | 11 (0) |
| 4 | 780 | 273 | 490 | 17 | 35 (0) |
| 5 | 4,224 | 1,511 | 2,538 | 175 | 61 (0) |
| 6 | 19,501 | **7,689** | 11,054 | 758 | 188 (0) |

- **Validation passed:** no event prefix was pruned at any k.
- Pruned fraction of all k = 6 regions: >= 1 - 7,689 / 2e5 = **96.2%**. Most of it is the normalisation, which
  removes cells where g23 can only be tiny relative to ||x||_inf. The triangle relaxation of the remaining 4-9
  steps adds only 3.9% at k = 6 (758 / 19,501).
- Survivors grow x5.0, x5.5 and x5.1 (k = 4->5->6). The event prefixes grow x1.7-3.1. The ratio of survivors to
  event prefixes widens from 7.8 to 25 to 41 (k = 4, 5, 6).
- **Projection:** N10 = 41 x 7,689 x 5.09^4 = **2.1e8 > 1e7. Verdict: STOP** (scoped negative, unverified).
  (1,8,2)'s own term, 5.2e6, is already half the line. Using the max over triples, adding (3,9,2) could only raise
  it.
- (3,9,2): 2 / 34 / 414 / 3,132 survivors at k = 1..4, with no event prefix pruned. It was stopped at k = 5, about
  2.5 CPU-h short of finishing, because it cannot change the verdict.
- **Side measurement, triples with no event hits** [measured]:
  - (1,2,0): survivors 2 / 19 / 77 / 358 at k = 1..4, against 822 regions at k = 4 (44% survive).
  - (4,1,3): 2 / 19 / 100 / 508 at k = 1..4, against 1,676 regions at k = 4 (30% survive).
  - The relaxation does not tell the 679 no-hit triples apart from event triples. A certificate would have to carry
    all of them at survivor rates similar to (1,8,2).

**What the method would certify** (answering the DRI): the pruning is sound in exact arithmetic, but the LPs here are
floating point. A certificate would also need each pruned cell's Farkas ray verified exactly. With that it gives a
rigorous **upper bound**:

  P(E) <= the sum of event masses over surviving step-10 regions, over all 720 triples, + 1e-49.

It is exact only if every surviving region's event polytope mass is integrated exactly. Restricted to the 41 event
triples it is an estimate, not a certificate. As measured, neither version comes in under 1e7 regions.

**Why it fails [inferred].** The triangle relaxation over 4 or more remaining steps is loose. |Wh| compounds the
relaxation gap step by step, as with T344's interval bounds. So it proves infeasibility only near the end, or where
the cell itself forbids the event. The useful pruning (96%) comes from the geometry of g23, not from the dynamics.

**Alternatives not ruled out / not tried.**
- Tighter relaxations: split some unstable neurons (partial branching, as in BaB-style verifiers); multi-neuron or
  PRIMA-style constraints; bounds from the event constraint propagated backwards.
- A lower prune threshold: compute the triangle bounds with the event constraint already in the LP.
- Pruning only from k >= 7, where fewer steps remain.
- A larger DELTA: with g23 >= 1.5 and ||x||_inf <= 7.5 at DELTA = 0.2, the excluded mass is about 20 Phi(-7.5)
  = 6e-13, still far below the event's 2.2e-7. It would prune more by geometry. **Not tried. This is the cheapest
  rescue to test**, but the normalisation prunes only cells where g23 is tiny relative to ||x||_inf, so it cannot touch
  the survivors that carry most of the growth.
- Each of these trades LP cost per cell for pruning. At about 0.2 s per cell for k = 6, 2.1e8 cells is 1.2e7 CPU-s,
  so a 20x tighter relaxation would be needed to reach the line.

**Compute (T390):** (1,8,2) 1.18 CPU-h; (3,9,2) about 0.8 CPU-h (stopped); side triples about 0.2 CPU-h. Total about
2.2 CPU-h of the 6 CPU-h cap. Cost is dominated by the relaxation LPs: 16 neurons x 2 bound LPs x the remaining
steps, at about 1-3 ms each.

---

# T398: tighter pruning (event-aware bounds + ReLU splitting), pre-registration (2026-10-19, before the kill test)
**Pruner (`relax2.py`).** It runs four stages in order; each is sound, i.e. it removes only non-event points.
0. The g23 normalisation (as in T390).
1. T390's relaxation: triangle relaxation, forward LP bounds, and the event constraint.
2. **Event-aware bound tightening.** Every unstable pre-activation's bounds are recomputed by LP over the full
   stage-1 relaxation *including* the event constraint (one Jacobi pass), then re-checked.
3. **ReLU splitting**, up to DEPTH = 6 unstable neurons, fixed in advance. The order is earliest remaining step
   first, then largest triangle area u·(−l)/(u−l). Splitting is depth-first, and a cell is pruned iff every leaf LP
   (event included) is infeasible.

CROWN-style back-substitution bounds are a relaxation of the same LP, so they are no tighter than the LP bounds used
here; CROWN would only be faster. I did not implement it, because CPU time is not the binding constraint at this cap.

**Development diagnostic (done before pre-registering; it informs the expectation).** I sampled (1,8,2) cells that
T390 cannot prune. On them:
- the relaxation's upper bound on the margin (logit_p3 − max other) is +0.03 to +31;
- event-aware tightening lowers it by 0 to 0.7, and makes some k = 5 cells negative, i.e. pruned;
- hit-and-run plus local ascent inside the cells finds best margins of −0.15 to −1.9, so those cells are probably
  event-free and the relaxation is loose;
- 2 cells, both found by the line walk, have positive best margins (the probe found an event point in only one of
  them, so the probe is weak).

At k = 4, a 60-cell sample showed no gain from stages 2-3 at DEPTH = 4.
**Expectation: STOP**, with gains concentrated at k = 5-6 where fewer steps remain.

**Kill test (unchanged from T390).** Branch and bound on (1,8,2) for k = 1..6 with the new pruner, with stage
attribution per pruned cell. Validation (could fail): no line-walk event prefix may be pruned.
**GO iff 41 x S6 x (S6/S5)^4 <= 1e7.**
Side measurement: (1,2,0) to k = 5. Caps: 12 CPU-h in total, 2 processes, 2e5 survivors per step.
If (1,8,2) cannot finish k = 6 in the budget, the result is reported as censored.

**What it would certify:** the same as T390. Sound pruning plus exactly verified Farkas rays gives a rigorous upper
bound summed over the surviving step-10 regions of all 720 triples. It is an exact value only with exact polytope
masses.

## T398b pre-registration (2026-10-19, before running): the normalisation scales with t0
- **Facts.** T399 found exact event witnesses in 4 of 9 benchmark nodes whose hit-and-run probe was negative
  (unverified). They sit on the normalisation boundary g23 = 0.1 ||x||_inf, i.e. at ||x||_inf >= t0/0.1 in original
  scale: real but with negligible mass. So **event feasibility is a poor pruning target**; what matters is mass.
- The cheapest mass-aware lever: DELTA may be as large as t0/7.5 (loss 20 Phi(-7.5) = 6e-13, below 1e-3 of a 1e-9
  target).
  - On 60 benchmark nodes (`delta_test.py`, base relaxation only), the pruned share is 3/60 at DELTA 0.1, 20/60 at
    0.2 and 29/60 at 0.33.
  - No line-walk event node was pruned (0/8). Pruning one would not be unsound at the larger DELTA anyway, since
    t0 = 2.5 events need g23 >= 2.5.
- **Target t0 = 2.5 (provisional; truth to be measured with T362's cmc.py), DELTA = 1/3.**
- **Run:** branch and bound on (1,8,2) for k = 1..6 with the T398 pruner (relax2, depth 6) at DELTA = 1/3.
  - Same validation as before: the line-walk event prefixes must survive. Their points have g23 >= 1.5, not >= 2.5,
    so a pruned one is only a flag, not a soundness failure. They are reported separately.
  - Same GO line: 41 x S6 x (S6/S5)^4 <= 1e7.
  - Certificates of the pruned cells are dumped in T400's format v1 (tri, k, stage, leaves (L, 10, 16)) to
    `certs/` (git-ignored if large).
- Side measurement: (1,2,0) to k = 5 at DELTA = 1/3. The T398 side run at DELTA 0.1 was stopped at k = 4 (326
  survivors) to free the core.
- **Scope (DRI condition, written before T398b's result was read).** A bound from this lane covers **t0 = 2.5
  only** (DELTA = 1/3).
  - For t0 = 2.0 (the T402 threshold), DELTA can be at most 2.0/7.5 = 0.267, so pruning is weaker and node counts
    are higher.
  - Its node count will be projected separately from a DELTA = 0.267 run (or interpolated between the 0.1 and 1/3
    runs, labelled as such).
  - Nobody should read a 2.5 bound as covering 2.0.

## T398b results: (1,8,2) at DELTA = 1/3 (t0 = 2.5), k = 1..6 [measured; `logs/T398b_182_k6_final.log`, `certs/`]

| k | children | survivors | pruned by stage 0 / 1 / 2 / 3 | line-walk event prefixes (pruned) |
|---|---|---|---|---|
| 1 | 2 | 2 | 0 / 0 / 0 / 0 | 2 (0) |
| 2 | 20 | 18 | 2 / 0 / 0 / 0 | 10 (0) |
| 3 | 79 | 39 | 39 / 1 / 0 / 0 | 11 (0) |
| 4 | 544 | 146 | 368 / 17 / 11 / 2 | 35 (0) |
| 5 | 2,305 | 275 | 1,664 / 113 / 188 / 65 | 61 (0) |
| 6 | 4,294 | **781** | 3,115 / 160 / 113 / 125 | 188 (0) |

- **Projection: 41 x 781 x (781/275)^4 = 2.1e6 <= 1e7. GO by the pre-registered line** (unverified; nobody
  confirms their own claim).
  - For comparison, at DELTA = 0.1 (T398, same pruner): S5 = 1,314 at k = 5; k = 6 is still running.
  - T390, DELTA 0.1 without stages 2-3: S5 = 1,511, S6 = 7,689.
- Survivors / line-walk event prefixes = 4.2 at k = 6 (T390: 41).
- Cost: about 25 CPU-min for k <= 6.
- **What the GO does NOT cover (open):**
  1. The certificate needs all 720 triples; this projection scales by 41. If the no-hit triples survive like
     (1,8,2), the full count is 720 x 781 x 65 = 3.7e7 (a 20-triple random sample has been requested from workers).
  2. Growth beyond k = 6: (1,8,2) is now being run to k = 10, which ends the extrapolation for this triple.
  3. Soundness of each prune: float LPs; T400's exact verifier is pending on `certs/`.
  4. The mass of the surviving step-10 regions (T400's mass_upper). The bound is only useful if their summed
     event mass is close to the truth at t0 = 2.5 (truth worker running).
- **[Update] Exact verification (T400, independent code, unverified by the DRI).** All 5,983 pruning certificates in
  `certs/cert_d0.333_1-8-2_k{2..6}.npz` verify, with 0 failures. That is 2 / 40 / 398 / 2,030 / 3,513 cells at
  k = 2..6; my earlier note of 5,476 was a miscount.
  - Method: Neumaier-Shcherbina safe dual bounds, with every bound LP re-solved independently.
  - Cost: 590 s in total.
  - Controls: the verifier rejects all 47 bench cells with a found event point, and confirms 717/717 float prunes on
    a bench sample.
  - Code: claude/t400-bound-rigour @ 1b013590.
  - So open item 3 (soundness) is closed for (1,8,2) k <= 6. **Mass (item 4) is the weak side.** T400's cheap
    rigorous bound is median 365x MC at k = 4; the exact-marginalisation bound (M5) is 6.9x but costs 25 s per cell.
- Worker B: (2,1,9), a triple with no event hits, is pruned completely by k = 3, with survivors 1 / 1 / 0.

## T398 verdict (DELTA = 0.1, the pre-registered T398 test) [measured; `logs/bnb398_182.log`]
- (1,8,2) survivors at k = 1..6: 2 / 18 / 53 / 263 / 1,314 / **5,825**. Stages 2-3 add 188 prunes at k = 5 and 1,420
  at k = 6. Line-walk event prefixes pruned: 0.
- Projection: 41 x 5,825 x 4.43^4 = **9.2e7 > 1e7. STOP** (scoped negative, unverified). Event-aware tightening and
  depth-6 splitting cut the survivors 1.3x versus T390 (7,689), not enough.
- The side run (1,2,0) was stopped at k = 4 (326 survivors) to free the core for T398b.
- Reading: at DELTA 0.1 (rigorous for t0 >= 0.75) the method is out of reach. The normalisation does most of the work
  at DELTA = 1/3 (T398b).

## T398b, k = 7 [measured; `logs/bnb398b_182_k10.log`]
- (1,8,2): 11,683 children, **1,494 survivors** (x1.91 over k = 6). 294 line-walk event prefixes, 0 pruned.
- Cost at k = 7: 201 s split + 1,028 s relaxation.
- Re-projected from k = 7: 1,494 x 1.91^3 = 1.0e4 step-10 regions for this triple.
- **T400's MC curve** (1e9 draws, t0 = 2.5): the survivors hold 81% / 74% / 45% of the triple's slab mass
  (5.6e-8) at k = 4 / 5 / 6.
  - So charging at k = 6 costs >= 2.5e-8, against a target of ~1e-9.
  - The charge must be made at k = 10 with the event rows, or after cutting each cell to the projection of its
    event relaxation (T400 option (c), in progress).

## The late-triple obstruction (2026-10-19) [measured on 3 triples; inferred in general; unverified]
- Worker B's **(3,7,8)**: survivors 2 / 34 / 792 at k = 1..3, the full-space N_k with nothing pruned. It was censored
  in k = 4 at the 1.5 CPU-h cap.
- Root relaxations (k = 0, DELTA 1/3, event-aware) bound the margin at **+659 (9,7,8), +435 (3,7,8), +961 (8,9,6)**,
  against a truth <= 0. Branch and bound from the root is hopeless.
- **General reason.** Until the triple's high positions are read, the normalisation and the event relaxation are
  both satisfiable. So a triple with second-smallest position b keeps about all N_b full-space regions through
  step b.
  - There are 6 b (9-b) such triples: 108 at b = 6 (N6 ~ 2e7 each), 84 at b = 7 and 48 at b = 8. Forward enumeration
    over them needs >= 1e9-1e12 regions.
- **Charging them unresolved fails.** Every triple carries the same slab mass, 5.6e-8 = P(g23 >= 2.5)/720. So one
  uncertified triple costs 56x the 1e-9 target.
- **Reading.** A certified bound over all 720 triples at t0 = 2.5 is out of reach for forward enumeration with LP
  pruning. A per-triple certificate is feasible for early triples such as (1,8,2), pending the k = 10 mass cost.
- Untried: an abstraction of the reachable hidden state after the low-only steps, with a sound charge for its
  preimage mass.

## T398b progress, k = 9, and the (1,2,0) side measurement (2026-10-19) [measured]
- (1,8,2), k = 9: **13,040 survivors** from 77,857 children, ×4.26 over k = 8. Growth jumps at the step that reads
  p2 = 8. 801 line-walk event prefixes, 0 pruned. k = 10 is running.
  - The k <= 6 projection (1.3e4 at k = 10) is already exceeded at k = 9.
- (1,2,0) at DELTA 1/3, a no-hit triple whose three positions are all read by step 3: survivors 2 / 11 / 35 / 168 /
  973 / **3,609** at k = 1..6 (×3.7 at 5->6).
  - So "two positions read early" is necessary for pruning, not sufficient. This triple has no line-walk event
    points, yet it keeps growing.
- Worker sample (DELTA 1/3, per-triple caps 0.6-1.5 CPU-h). "Censored" means the run was stopped before k = 6.

  | triple | survivors by k | status |
  |---|---|---|
  | (2,1,9) | 1 / 1 / 0 | pruned completely by k = 3 |
  | (1,9,3) | 371 at k = 6 | complete |
  | (8,4,0) | 509 at k = 6 | complete |
  | (9,0,1) | 912 at k = 6 | complete |
  | (1,9,7) | 1,239 at k = 5 | censored |
  | (2,4,3) | 671 at k = 5 | censored |
  | (3,7,8), (5,7,6) | 2 / 34 / 792, nothing pruned | censored at k = 3 |
  | (4,6,2) | 290 at k = 3 | censored |
  | (4,9,2) | 277 at k = 3 | censored |
  | (2,9,7) | 220 at k = 3 | censored |
  | (6,9,1) | 82 at k = 3 | censored |
  | (8,4,7), (4,9,5) | 34 at k = 2 | censored |

  Details are in the workers' logs and json files in this folder.

---

# Lane summary: certified far-gap rare-event bound (T398 lead; T399, T400, T412, T414, workers A/B, truth worker), 2026-10-19
All claims are unverified by the DRI unless stated. Event: E(t0) = {pick = p3, g23 >= t0}, orig, target t0 = 2.5.

**Truth** [measured, truth worker, `truth_t398/`; T362's conditional MC, validated at t0 = 1.5]:

| t0 | P(E) | relative s.e. |
|---|---|---|
| 2.0 | 7.6e-9 | 12% |
| 2.25 | 1.1e-9 | 18% |
| 2.5 | 6.6e-11 | 44%: 8 nonzero lines, so an order of magnitude only |

**What was built and works** [measured]:
1. **Exact branch and bound** on the triple's true event cone, cell by cell over the 10 steps. Pruning stages:
   - the normalisation g23 >= DELTA ||x||_inf, rigorous for DELTA = t0/7.5 (loss 6e-13 at t0 = 2.5);
   - the T390 triangle relaxation;
   - event-aware bound tightening;
   - depth-6 ReLU splitting.
   Code: `bnb.py`, `relax2.py`, `k10_stream.py`.
2. **(1,8,2) complete to k = 10** at DELTA = 1/3. Survivors per step k = 1..10:

   | k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | survivors | 2 | 18 | 39 | 146 | 275 | 781 | 1,494 | 3,058 | 13,040 | **15,116** |

   - Every k = 10 survivor passes the exact event LP.
   - All 995 distinct full histories of 550,924 line-walk event points are among them, with 0 missing.
   - CPU about 6 h.
3. **Exact verification** (T400, independent code, safe dual bounds): 38,557 of 38,557 pruning certificates for
   k = 2..8 verify. **[Update] All (1,8,2) certificates for k = 2..10 verify: 259,723 of 259,723, with 0 failures**
   (T400). The k = 10 mass charge (T416) is running on 8 workers.
4. **Mass** (T400): plain MC at 1e9 draws shows the survivors' slab mass falls from 45% at k = 6 to 4.6% at k = 7,
   then stays flat at 2.6e-9 through k = 8. So the charge must be made at k = 10 with exact event polytopes; T400's
   option (c), relaxation-projection cuts, did not help.
   - 10-D rigorous mass (M5t) is about 6.8x over MC.
   - The cost is ~30-60 s per cell after T412, and possibly ~10x less after T415. So charging the 15,116 cells is about
     125-250 CPU-h now.
   - Pending as T416. This is the lane's remaining deliverable: a rigorous **per-triple** upper bound for (1,8,2).

**What does not work** [measured on samples; inferred in general]:
5. **All-triple certificate at t0 = 2.5: negative (unverified), scoped to forward neuron splitting and input-space
   CROWN, pending T418's independent re-check with library verifiers (alpha-CROWN, alpha-beta-CROWN).**
   - Late triples cannot be pruned before their high positions are read, so they carry the full-space region
     counts N_b: (3,7,8) and (5,7,6) give 2 / 34 / 792 with nothing pruned.
   - Root relaxations bound the margin at +435 to +961.
   - Input-space branch and bound with CROWN (T414) projects 1e14-1e15 boxes per late triple, a KILL. Its CROWN
     looseness has median 176 margin units.
   - Every triple carries the same slab mass, 5.6e-8, against a total truth of 6.6e-11, so no triple can be left
     uncertified.
   - Worker sample over 20 random triples:
     - 4 complete to k = 6: (8,4,0) 509, (9,0,1) 912, (1,9,3) 371, and (2,1,9) pruned completely;
     - 16 censored at 0.6-1.5 CPU-h, mostly at k = 2-3.
6. **DELTA = 0.1 (T390, T398): STOP.** Without the t0-scaled normalisation the projections are 2.1e8 and 9.2e7, both
   above 1e7.
7. **Cheap rigorous masses** (M2, wedge boxes) are 1e2-1e9x loose on k = 10 regions; only exact triangulation (M5t)
   is usable.

**What the lane can certify** (answering the DRI):
- Pruning is exact once T400 verifies the certificates.
- The bound is an UPPER bound per triple: sum of verified M5t event masses over the k = 10 survivors, plus 6e-13.
- It is not an all-triple bound, and not a bound at any t0 other than 2.5. t0 = 2.0 would need DELTA <= 0.267 and
  was not run.

**Alternatives not ruled out:**
- An abstraction of the hidden state reached after the low-only steps, with a sound charge for its preimage mass
  (no design yet).
- Symmetry or sharing across late triples: their early prefix trees coincide (claim 3), which saves a factor <= the
  number of triples sharing a prefix, not the N_b growth.
- Tighter multi-neuron relaxations for late triples.
- A different event decomposition than by position triple.

## Amendment (2026-10-19, before running): T398b on (9,3,2) and (3,9,2)
- Same pre-registered setting as T398b: t0 = 2.5, DELTA = 1/3, relax2 depth 6, `bnb.py` to k = 9, then `k10_stream.py` for k = 10.
- Why: T400's 1.1e11-draw MC found 17 events at t0 = 2.5, all on (9,3,2) and none on (1,8,2).
- Soundness test (could fail): every one of the 17 hits must lie in a (9,3,2) k = 10 survivor. A hit inside a pruned
  cell refutes a certificate.
- The k = 10 files go to T400 for verification and charging.
- **Side check, truth estimator vs the MC hits** [measured; `walk_test.py`, `logs/walk_test_hits.log`].
  - On all 17 of T400's MC hits (top three (9,3,2), g23 2.50-2.84), the truth worker's conditional walk (the same
    code path) covers the hit's own point.
  - It gives Z(x_hit) = 0.012-0.64 (median 0.20); 0 of 17 have Z = 0. So the walk does not miss (9,3,2) mass on
    these lines.
  - The 6.6e-11 vs 1.55e-10 gap (~2 sigma) is consistent with chance for a heavy-tailed estimator with 8 nonzero
    lines. Its s.e. is unreliable at that count.
  - The plain MC (1.55e-10 +- 0.37e-10) is the reference.

## (9,3,2) / (3,9,2) k = 10 enumeration: shard plan (2026-10-19)
- Both triples to k = 4 by `bnb.py`: (9,3,2) has 2 / 34 / 414 / 2,891 survivors at k = 1..4, and (3,9,2) has
  2 / 34 / 414 / 2,849. 0 line-walk event prefixes were pruned.
- From the k = 4 survivors, `dfs_stream.py` (sharded depth-first, checkpointed) carries each parent to k = 10.
  - Validated on (1,8,2): identical k = 10 survivors for the same parents; resume reproduces them.
- **(9,3,2):** 6 shards (`s<i>of6`), each split in halves.
  - Half 1: me (s0, s1), worker A (s2, s3), worker B (s4, s5).
  - Half 2 (`_h2`): C9 (s0, s1), C10 (s2, s3), C11 (s4, s5).
- **(3,9,2):** 8 shards (`s<i>of8`).
  - Half 1 or whole shard: W1 (s0, s1), W2 (s2 whole, s3), W3 (s4 whole, s5), W4 (s6, s7).
  - Half 2: C12 (s0, s1), C13 (s3, s5), C14 (s6, s7).
- Caps: 100 CPU-h per worker (DRI). Cost is about 50-400 CPU-s per parent, heavy-tailed. The k = 10 survivor counts are
  large: about 1e3-6e3 per 5-25 parents in early shards, so (9,3,2) projects to ~5e5 or more.
- When all shards are DONE:
  1. the union of `surv_dfs_*` files = the k = 10 survivors;
  2. the 17-hit soundness test (`hits/prob_2.5_hits.npz`) against the (9,3,2) union;
  3. hand-off to T400.
- **Rebalance log** (the final union must cover every k = 4 parent exactly once; checked by `done_parents` at the end):
  - 2026-10-19 23:30: (9,3,2) s2of6 half 1 is split.
    - Worker A keeps the head `--parents range(2,933,6)` (156 parents, tag s2of6).
    - C13 takes the tail `range(938,1443,6)` (85 parents, tag s2of6_tailC13).
