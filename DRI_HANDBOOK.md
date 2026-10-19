# DRI handbook: how this project was run, and what we learned

↑ [Project README](README.md) · [STATUS](STATUS.md) · [TASKS](TASKS.md) · [INDEX](INDEX.md) · [AGENTS rules](AGENTS.md)

Written by the outgoing DRI (Claude) on 2026-10-17 so that the next DRI (Claude, Codex or human) can pick up the
project without the chat history. Read this, then STATUS.md, then TASKS.md.
Updated by the 2026-10-18 DRI: section 10 (00:46), the struck claim in section 7 (05:15), and section 11 plus the
corrections marked "2026-10-18 pm" (afternoon). Then read [OWNER_QA.md](OWNER_QA.md) for the owner's questions and
directives.
Note (T396, 2026-10-19): the DRI directives 1-8 and the ladder protocol that used to sit at the top of TASKS.md are now
in [docs/PROTOCOLS.md](docs/PROTOCOLS.md); "TASKS Directive N" below means those. Finished board text is in
[docs/archive/TASKS_history.md](docs/archive/TASKS_history.md).

## 1. The goal, and where it stands

- **Challenge (ARC, AlgZoo):** compute the accuracy of the 432-parameter 2nd-argmax RNN M_16,10
  (`alg_zoo.example_2nd_argmax()`, accuracy 0.95313575 for x ~ N(0, I_10)) *mechanistically* (no random or
  pseudorandom inputs), with MSE at least as good as plain sampling at equal compute. Must also work on the 5
  trained seeds and on noisy weight copies. [UPDATED 2026-10-18 pm: the 300-billion-sample answer key (T192) is
  0.9531175 +- 0.0000004; the figure above and the 20M-sample 0.95317945 in `m16/truth.txt` agree with it within their
  sampling error.]
- **Status:** not solved. Every deductive estimator on M_16,10 loses by 37x to >4000x (leaderboard v3, LB3).
  [UPDATED 2026-10-18 pm: still unsolved at the strict bar, but the gap on retrains is now ~2.6x. The sampling-free
  walk estimator T317f scores F 2.63 [1.93, 3.39] on the fresh sealed panel HP4 at 3.3k FPE (RMS .75 pp, bias
  +.44 pp): lenient pass, strict fail. Scope: zoo-config retrains only; not deductive in Hilton's strict sense, because
  its approximation scheme was selected by MC on DEV2 (T324); it breaks down between 0.1% and 0.25% relative weight
  noise (T337). On the fresh open panel DEV3 it scores F 2.49. A continuity-walk bug for long lines (found 14:18)
  makes T340 variant B, T349 and T354 suspect; T317f's short window is believed unaffected, verification pending.
  See section 11 and H-L1.]
- **What works:** an understanding-based recipe (RECIPE112, `m16/approaches/understand_m43/est75.py`) beats sampling
  on every trained seed and noisy copy of M_4,3 (independently confirmed, H-U1/H-U2) and of (H,T) = (5,3), (6,3),
  (8,3) (E43, review T113 unfinished). Cells visited / linear regions fell 4.8 -> 2.0 from H=4 to 8 (H-U4) — the
  only hint so far that cost can grow slower than the region count.
- **Scaling verdict (E46, E48; unverified):** the recipe stays *accurate* as lists get longer ((8,4) passes on all
  5 seeds + noisy copies, MSE ratio <= 0.27; (8,5) seed 0 error 0.0002 pp, ratio 0.0097) but its *cost* explodes:
  1.7e5 -> 8.7e8 -> 2.2e11 ops at H=8, T=3,4,5 (cells/regions 2.8 -> 5.4 -> 33). M_16,10 projects to >=1e17 ops [CORRECTED 2026-10-18: an extrapolation from two growth steps, ~6.5e16-4.7e17 (trend ~2e23), not a bound]
  (realistic trend ~1e23). An earlier "(8,4) fails" (E45) was a 4-D orthant-mass bug, fixed in
  `m16/approaches/estT_fix/`.
- **T=4 small-H track (E47):** a fixed great-circle grid ("Hopf", exact along each circle) beats sampling at (4,4),
  (5,4), (6,4) (poorly trained, bulk errors), but it is *quadrature* (model-independent node set), not
  ARC-mechanistic, and its margin shrinks with H. T111: the T=4 failure is about T, not undertraining.
- **ARC's own position (public, 2026-08-17):** "not much closer to succeeding at the challenge". Their M_4,3
  estimators were 2.4-10x *worse* than sampling in MSE and they called that "roughly match" (E32). The private claim
  that ARC cracked M_16,10 is doubtful on public evidence (H-E5). [UPDATED 2026-10-18: H-E5 is now "reported, unverified"; see section 10]

## 2. How the project is organised (keep this)

| File | Role |
|---|---|
| [TASKS.md](TASKS.md) | Task board: In progress (owner, folder, CPU), Backlog with Ready/Blocked states, Ladder protocol, Recently finished |
| [INDEX.md](INDEX.md) | Registry (E-experiments, L-literature, R-reviews, LB-leaderboards) + hypothesis ledger (H-*) with status |
| [STATUS.md](STATUS.md) | Plain-language and technical status for the owner |
| [AGENTS.md](AGENTS.md) | Rules for every agent (CLAUDE.md only imports it — some agents are Codex) |
| `m16/leaderboard.py` | DRI-only shared harness (v3): classes, compute rule, overrides, truths, raw runs |
| `m16/approaches/<name>/` | One folder per attempt, README with the AGENTS.md sections |
| `review/<name>/` | Independent reviews (never edit the reviewed folder) |
| `lit/` | Literature scans; `lit/web_log.md` logs every post-Jan-2026 ARC source read |
| [OWNER_QA.md](OWNER_QA.md) | Every owner question with the DRI's answer and later updates; owner directives verbatim (its section 8) [ADDED 2026-10-18 pm] |
| Panels HP1-HP5 (sealed), DEV2/DEV3 (open) | Built by the T192 child (`gpu_cost_law/` on `claude/t192-gpu-cost-law`); sealed truths are read only by the scorer T281 (`claude/hp1-scorer`) [ADDED 2026-10-18 pm] |

## 3. Process lessons (what made it work)

1. **Board first.** Every agent reserves a task (commit+push the claim alone) before working; new ideas become
   backlog rows immediately. This let ~20 Claude and Codex agents work in one repo without much duplication.
   Still check for overlaps: a Codex agent once held T90 while I spawned a duplicate (fixed by splitting rungs).
2. **IDs collide.** Take max+1 from the *remote* right before pushing and push at once; IDs are never reused.
   I collided several times (E28, T42, T112) and had to renumber.
3. **Nobody confirms their own claim.** Every positive went to an independent reviewer with fresh noise seeds,
   their own truth code and adversarial tests. Reviews found real problems each time: cost under-counting (1.24x,
   2.45x), a cache bug keyed on `id()` of a temporary, non-invariance to neuron rescaling, correlated truths.
4. **Negatives are "unverified" until checked.** An exhaustion checklist (tried / not tried, with reasons) plus an
   independent re-check before "dead (verified)". The re-check of E31 found the original *reason* was wrong.
5. **Measure the measurement first.** The leaderboard had to be fixed before results meant anything:
   - charge compute *as implemented*, not as implementable (srb's "2x win" vanished);
   - exclude rows a method was tuned on ("advice"); keep DEV models for tuning;
   - group baseline = mean over runs of p_m(1-p_m)/cost, not mean(p)(1-mean(p));
   - cluster-bootstrap CIs by model;
   - truths must use **independent RNG streams per model** (shared streams made errors correlated and biased large
     budgets);
   - classes reviewed by the DRI: mechanistic / quadrature (fixed node sets, not ARC-mechanistic) / sampling;
   - plain sampling scored against itself must give ratio ~1 — use it as the calibration check.
6. **Calibrate against the challenge setter.** Reproducing ARC's own M_4,3 estimators (E32) showed their bar and
   an arithmetic slip in their comparison. Read the primary sources (owner relaxed independence rule 7).
7. **Climb slowly (owner directive).** Change one axis by one step from a passing rung; all 5 seeds + noisy copies
   at every rung; on failure stop, diagnose, fix, re-run earlier rungs. Axes: T, H, training amount, noise, task.
   **Stay in the target regime:** small-H/large-T zoo models are poorly trained (accuracy 0.32-0.60, bulk errors),
   unlike M_16,10 (0.95, thin tie bands). Main path: H up at T=3, then T up at H=8, then toward H=16. Bulk-error
   models are a second track (T116), because ARC's heavily noised copies have bulk errors too.
8. **Understanding beats brute force only if it scales.** On tiny models exact enumeration is cheap, so "cheaper
   than enumeration" is the wrong small-model bar; the right one is cost growing with H, T and boundary families,
   not with the number of linear regions (owner ruling).
9. **Report honestly to the owner in plain words**: what is solved, what is not, odds. Corrections go in the ledger
   (never delete).

## 4. Technical lessons

- **Mechanism of M_16,10 (FINDINGS):** running max, record detector, comparators, damped rotating modes (phase =
  time since max). Errors: near-ties of the top order statistics plus a "blind" all-negative region.
- **Why deductive methods fail on M_16,10:** >1e7 linear regions (H-M7) [UPDATED 2026-10-18 pm: exact prefix counts
  34 / 792 / 21,032 / 632,394 at k = 2..5 (confirmed to k = 4), >= 15.5M at k = 6, growth x23-33 per step, ~1e13
  regions at T = 10 by extrapolation; region-by-region enumeration and coverage are dead for the target (H-E13)];
  the error mass sits in rare, wide, irregular tie bands (H-E11/H-E12, heavy tails from T~5); deterministic
  summaries of low inputs are Jensen-biased (H-E10); ARC's cumulant propagation breaks because a scalar input per
  step makes hidden units perfectly correlated (E35); harmonic projection leaks ~1/D across the tie wall (E34).
- **What made the small-model recipe work (E36-E43):** the model has no biases, so correctness depends only on
  x/|x| and errors are unions of cones; organise by true boundaries x_i = x_j (half-chambers), follow gates step by
  step, resolve only gates whose switch could change the answer (impact / Lipschitz margin bound through the
  remaining layers, scale-invariant), compute exact cone masses (Girard in 3-D; Plackett/triangulation in 4-D).
  Make every quantity invariant to function-preserving neuron rescaling and permutation, and test that permanently.
- **Where it breaks:** freezing intermediate gates without a margin certificate (fixed for one step by RECIPE112;
  multi-step version T108); 4-D cone masses cost ~300x 3-D (T109); exact boundary tracing at T=3 costs ≥ the region
  count (E39) — any gap can only open along T.
- **Quadrature (fixed Sobol tie slicing) wins at short T and on mild noise** (sci_tao, sci_feynman) but ARC would
  likely not count it as mechanistic. [UPDATED 2026-10-18 pm: under Hilton's definition fixed nodes are inductive and
  gains averaged over rule scrambles are RQMC, i.e. sampling (T263; INDEX 4b). The best honest deterministic number on
  the clean target is ~1.1-1.3x (T223, quadrature class).]

## 5. Operational pitfalls (each cost hours)

- `git pull --autostash` **replaces files that running jobs have open**: the job keeps writing to a deleted inode.
  Keep live outputs in git-ignored `data/` and copy snapshots into `results/` (leaderboard v3 does this).
  Recover from `/proc/<pid>/fd/<n>` if it happens.
- `pgrep -f "<cmd>"` matches the shell running the pgrep itself; never use it in wait loops. Use the PID.
- Never kill by name pattern (rule 9). Stopped agents leave queue scripts that relaunch jobs: kill the queue PIDs
  first, then workers, checking `/proc/<pid>/cwd`.
- Background shells die at the 30-minute tool limit and on container restarts; launch long queues with
  `setsid nohup ... &` and make queues resumable (skip completed (name, budget) pairs).
- The container restarted twice: most likely memory exhaustion (15 GB, no swap) and an API session limit. Rule:
  ≤2 GB per process (`ulimit -v 2500000`), ≤2 heavy jobs on 4 cores. After a restart, resume agents with SendMessage
  (their transcripts survive), and tell them their processes died.
- Shell heredocs without quotes execute backticks inside Python strings; use `<<'EOF'` and pass values via argv.
- The stop hook demands a clean tree; commit agents' live outputs as WIP snapshots (all agents share one checkout,
  so this does not conflict with their later commits).
- Token economy: sub-agents' long transcripts are re-sent every turn; for final work spawn a fresh agent with a
  tight brief, scripts that print ≤10 lines, details to files.

## 6. Scoreboard: which sizes pass (beat sampling on all 5 seeds + noisy copies, deductive)

| (H,T) | params | method | status |
|---|---|---|---|
| (4,3) | 32 | RECIPE112 / T75 / fibre F2L2 | pass, **independently confirmed** (R2, R3) |
| (4,3), n_seqs 2^20-2^29 | 32 | T75, fibre | pass (E44; review T114 open) |
| (5,3), (6,3) | 45, 60 | RECIPE112; fibre F2L2 | pass (fibre confirmed R4; RECIPE112 review T113 open) |
| (8,3) | 96 | RECIPE112; fibre | pass (unreviewed) |
| (8,4) | 112 | RECIPE112 + multi-step certificate (estT recipe108) | pass after mass fix (E46, unreviewed); cost 6e8-1e9 ops. [RESTATED 2026-10-18 under T196 FLOPs: worst MSE ratio 0.52 as charged (2·ops), 0.36 after T192's op-completeness audit (estT = 1.37x ops); still pass, unverified] |
| (8,5) | 128 | same | accurate on seed 0, 2.2e11 ops (E48); other seeds not run. [UPDATED 2026-10-18: **fail**: recipe108 loses on seed 2 (MSE ratio 3.4 in estT ops, ~4.6 in T196 FLOPs); T192 S2, unverified] |
| (4,4), (5,4), (6,4) | 48-84 | Hopf circle grid | pass but **quadrature class** (E47) |
| (16,10) target | 432 | all | **fail**: deductive methods 37x->4000x worse than sampling (LB3); recipe projects >=1e17 ops [CORRECTED 2026-10-18: an extrapolation from two growth steps, ~6.5e16-4.7e17 (trend ~2e23), not a bound]. [UPDATED 2026-10-18 pm: on zoo-config retrains (not the target model or its noisy copies), sampling-free T317f has F 2.63 on sealed HP4: lenient pass, strict fail (section 11)] |

## 7. Theory notes (why this is hard, and what any solution must use)

- Exact accuracy of a ReLU net = Gaussian measure of a union of polyhedral cones: #P-hard in general (polytope volume,
  Dyer-Frieze). ARC only needs approximate answers.
- Black-box deterministic volume estimation is provably worse than random sampling (Elekes; Barany-Furedi): a
  deterministic method can only win by reading *structure in the weights*.
- Networks can compute pseudorandom functions; deterministic efficient estimation of circuit acceptance probability in
  general is the CAPP problem, tied to circuit lower bounds (Kabanets-Impagliazzo, Williams). If PRGs exist,
  enumerating PRG seeds works — but that is pseudorandom sampling, excluded by ARC. So adversarial networks are almost
  surely impossible under ARC's rules; ARC's bet (no-coincidence principle, Christiano-Neyman-Xu) is that *trained*
  networks have short explanations. Nothing rules out M_16,10, but its explanation may be long.
- Region counts depend on the weights, not just (H,T). Upper bound prod_{t=1..T} sum_{j<=t} C(H,j) is useless:
  825 vs 66 at (4,3); 1.3e36 vs ~1e7-1e8 at (16,10). [STRUCK 2026-10-18: refuted by T232 (H-E14), trained region counts equal norm-matched random nets; see review/fable_review A5] Trained nets use a tiny fraction of possible regions because
  their neurons switch in coordinated ways (running max, record alarm, clocks) — that coordination is the structure a
  solution must exploit.
- Any per-gate-pattern method (enumeration, fibre tracing, cell splitting) pays at least the number of distinct gate
  histories it must distinguish; at T=3 exact boundary tracking costs >= the region count (E39). The cost must be cut
  by *merging* histories that do not matter for the answer, not by cheaper bookkeeping.
- Compute is not the main blocker for a single exact answer (a GPU could plausibly do ~1e17 branchy ops in hours-days),
  but (a) the real trend is ~1e23, (b) irregular geometry runs at 1-5% of GPU peak, (c) at 1e17 ops sampling is precise
  to ~5e-6 pp so the method must be essentially exact, and (d) brute force at that scale is not what ARC is asking for.
  [CORRECTED 2026-10-18 pm: "a GPU in hours-days" does not hold for exact region enumeration. With ~2e13-1e14 regions
  at T = 10 (H-E13), 6.5 ms-0.2 s per region per core measured at k = 6, and x10-100 more at 10-D, the estimate is
  ~1e21-1e24 operations: hours to months on a 1 GW site (years in the worst corner), ~1e12-1e16x the compute sampling
  needs for 0.1 pp (OWNER_QA 6.3). The 1e17 figure was the recipe's projection. Point (d) stands.]

## 8. Ideas to try next (brainstorm, prioritised; board rows T187-T193)

All aim at the one isolated failure: gate histories multiply with every extra time step.
[UPDATED 2026-10-18 pm: outcomes (bounded phases, unverified). None showed sub-region growth along T. T187: the weight
bounds grow along T, and the oldest input's sensitivity exceeds the newest's in 97/384 (H4) and 175/384 (H8)
witnesses, so no safe merge. T188: one exact reset (all gates off), no other future-equivalent distinct states.
T189: closed for now after a bounded certificate phase. T190: no strict saving in cold counts. T191: parked. T192:
done (the GPU port reproduces E46/E48; the (8,5) recipe fails on seed 2; it then produced the H-E13 counts). T193:
blocked on saved T5 rays. The exact route itself is dead (H-E13; section 10).]
1. **Fading memory (T187).** The rotating modes are damped, so inputs far in the past may barely affect the final
   answer. Bound, from the weights, the influence of step t on the final logit margin (products of |Wh| restricted to
   active patterns, or exact per-cell Jacobians); merge cells that differ only in steps whose influence is below the
   margin. Cheap first test: measure the decay rate at (4,4) and (8,4). If influence does not decay, drop it.
2. **Same-future merging / bisimulation (T188).** After each step, merge cells whose current affine state maps lead to
   the same answers for every future input (state-machine minimisation). Cheap first test: in recipe108 at (4,4) and
   (8,4), measure how many cells are future-equivalent. Known: 33-49% of resolved splits never change the class.
3. **Mechanistic coordinates (T189).** Write accuracy as an integral over order statistics (top values, positions,
   time since max), whose joint law is known in closed form, plus a weight-derived *bound* on how much the other inputs
   can move the answer (not an average: E27 and E31 failed because averaging the low inputs is Jensen-biased).
4. **Time-shift reuse (T190).** The same Wh, Wi run at every step; a cell sub-tree computed for steps 1..k may recur
   (shifted) for steps 2..k+1. Cache and reuse; measure the hit rate.
5. **Mechanistic circles (T191).** Turn the Hopf grid (E47) into a mechanistic estimator by choosing circles from the
   weights (e.g. through the tie walls and the bulk-error chambers) instead of a fixed grid; needed for the bulk-error
   regime (heavily noised copies).
6. **GPU exact reference (T192; owner approval needed).** Port exact enumeration / the recipe to compiled+GPU code and
   time (8,5), (8,6) to measure the real cost law; possibly an exact answer key for M_16,10 (T5).
7. **Independent checks (T193)** of E46-E48 before building on them (fresh noise seeds, own truth, op recount).

**Test sizes for ideas 1-4:** develop on (4,3) -> (4,4) -> (4,5) (exact answers in seconds; (4,3) must keep passing),
confirm in-regime on (8,3) -> (8,4) -> (8,5) (compare with recipe108's cells/regions 2.8, 5.4, 33), then (16,3) ->
(16,4) -> (16,5) (same H as the target, 99% accurate) only after (8,·) shows sub-region growth. Success criterion:
cells or ops growing no faster than the region count along T, while still beating sampling on all 5 seeds.

## 9. How to continue

1. Take the priority queue at the top of TASKS.md (ideas 1-2 first: they are cheap measurements that decide whether the
   current recipe can be rescued). [SUPERSEDED 2026-10-18 pm: ideas 1-2 did not rescue the recipe (section 8); take
   the In-progress table and the DRI directives in TASKS.md, and section 11.]
2. Finish open reviews before new claims: T113 (E43), T114 (E44), T80 (E35), T57 (E34), and the E46-E48 check.
3. Keep the leaderboard honest: any new estimator for M_16,10 goes through `m16/leaderboard.py` v3; class it
   (mechanistic / quadrature / sampling) before reporting.
4. Codex agents have recorded work on the board that never reached this branch (e.g. their T131 fix). Ask every agent
   to push to `ccr-6a61f25c-wn80ii`, and check `git log` before trusting a board note.
5. Ask ARC (T41, needs owner approval) for the M_16,10 protocol if a candidate solution appears.
6. The owner's working style that worked: climb one axis one step at a time; backtrack to the last success when a step
   fails; verify negatives before calling an idea dead; keep token use low (lean agents, scripts printing <=10 lines).

## 10. 2026-10-18 update (new DRI)

- **Main route: exact enumeration made cheaper by structure.** ARC's own stated plan for M_16,10 combines feature
  analysis with "brute force after exploiting symmetries" ([lit/web_arc](lit/web_arc/README.md) S1). From T >= 4, exact enumeration costs only 1.2-1.5x
  recipe108 with ~zero error (H-U5), and recipe108 fails (8,5) on seed 2 (H-U6). ARC's bar is deductive + MSE vs
  sampling at equal compute; brute force is not excluded (QA #5). Structure to exploit: truth-cone start
  ([chamber_split](review/chamber_split/README.md), 2-5x fewer cells at H=16), future-equivalent cells (T188),
  time-shift reuse (T190), cost on noisy copies (T202).
  [CORRECTED 2026-10-18 pm: this route is dead for region-by-region enumeration and mass coverage (H-E13; DRI ruling
  after T270: x23-33 regions per step, >= 15.5M regions at k = 6; coverage still grows by a provisional >= 7.6x per
  step at delta = 1e-3, T222/T270). T188 and T190 found no saving in their bounded phases. Since the 07:03
  reframing the working route has been the cheap regime: own-cost F, sealed retrain panels, and exact 1-D walks
  combined under declared approximations (T286 -> T317 lineage). Since 13:12 the owner's top priority is rare events
  (T340). Exact computation survives as a component (the walks); cleverer exact methods are backlog T346/T352. TASKS
  DRI Directive 6 still names the old route and should be updated.]
- **The recipe is now a diagnostic** (which splits matter, T195), not the method to scale; T170 is deprioritized.
- **Protocol:** TASKS DRI directive 7 fixes the rung test (5 seeds + >= 8 noisy copies per seed at absolute sigma 0.05
  and relative 0.005, 0.01; exact or >= 1e8 independent-stream MC truth; T196 FLOPs; every budget with a CI).
  [UPDATED 2026-10-18 pm: estimator claims are now scored under Directive 8 amendment 2 by the independent scorer
  T281 on sealed retrain panels (HP2-HP5; HP1 is the noisy robustness panel). F = mean(e^2) x max(C, 1k FPE) /
  mean p(1-p); strict F <= 1, lenient F <= 10 with the upper CI <= 10 for a candidate. The written cost convention is
  in the same directive. Directive 7 still governs small-model rungs.]
- **Leaderboard caveats (repairs in T204):** LB3's summary takes the min over budgets (biased low; quote each budget
  with its CI); `table()` demotes arc_cumulants to "sampling" because its rows were stored before it entered CLASS
  (det_ok=False); srb's win rests on a hand-counted, uninstrumented cost factor 3.3/1.6 ≈ 2.06
  (`control_variate/rb9.py:158-184`), without which its ratios are ~0.4-1.1.
- **ARC:** H-E5 is now "reported, unverified": the owner's ARC contact says a newly joined ARC member has an M_16,10
  solution and that ARC is waiting for a comparison. Owner ruling 2026-10-18: "Unless we have a candidate solution, we
  should not contact ARC." The comparison package is T203; contact is T41. [UPDATED 2026-10-18 pm: the T323 dossier
  was given to the owner as a progress note, not a candidate solution; the DRI recommends no contact yet.]
- **Read:** [QA_2026-10-18](review/dri_handover/QA_2026-10-18.md) (previous DRI's answers), the DRI directives at the
  top of [TASKS.md](TASKS.md#in-progress), [T196 cost convention](review/cost_convention/README.md).

## 11. Lessons, 2026-10-18 (afternoon)

The cheap-regime lineage (T286 -> T317f), the sealed panels and the first rare-event round taught these. Numbers are
from INDEX and TASKS at ~14:21 UTC; results are unverified unless marked.

1. **A sealed panel is single-use for claims.** HP2 took five submissions of the walk lineage (R1W -> T317), so its
   later scores are forking-path numbers. HP3 (T321) and HP4 (T328) were each built for one pre-registered submission
   (T317c, T317f) and are spent; HP5 (T336, sealed, nothing scored) is the reserve. Build the next reserve before the
   current one is needed (a panel costs up to ~3 GPU units and some hours). Keep an OPEN development panel as well:
   DEV2 has served ~40 variants, so a DEV2 winner is a selected number (forking paths). DEV3 (T342, seeds 500-599,
   truths published) is for checking a development result before it is named for a sealed panel (T343, T350 and T353
   do this). Its first uses paid off at once: T339's DEV2 variance cut (T343), T353's DEV2 gain (T355) and T340's
   rare-event wins (T350) all failed to replicate on DEV3, while T317f held (DEV3 F 2.49).
2. **Recheck every F before acting on it.** F = MSE x max(C, 1k FPE) / mean p(1-p) is linear in C at fixed RMS. A
   child's "F 6.3 at 2.2k FPE" (T326) was 1.9 at the same RMS (.79 pp); the DRI and Fable r5 both caught it, and the
   slip had made the cost lane look ~3x less valuable. Recompute F from (RMS, C, mean p(1-p)) whenever a cost or an
   error changes, and quote a projected F only with its inputs.
3. **Post hoc counts inflate easily: pre-register the scoring.** The DRI's first T340 summary said "12 of 15 cases beat
   sampling" (approximate C, ties counted as wins). With each run's executed C and ties (|diff| < .05) kept apart, it
   became 10 win / 3 tie / 2 loss for the better variant per event family, and 5/15 wins for each fixed variant. Even
   the corrected count moves with the implementation: T350's own code gets 12 / 1 / 2 under the same rule on the same
   3 models (calibration, ab2ec444). Write down in advance the variant or family rule, the tie band, the cost charged,
   and the zero-answer rule for rare p: sampling's relative error is sqrt((1-p)/(pC)) only when pC >= 1; when pC < 1
   the answer 0 (relative error 1) beats sampling, so F is uninformative there and est/truth plus relative error are
   the measures. T350 fixed all of these before opening the DEV3 truths, and on 6 fresh models the family rule went
   11 win / 6 tie / 13 loss: no go (E2 by variant A held; E1 by variant B did not). T350's own walk was then
   verified exact, so the negative is not an artefact of the long-line walk bug found at 14:18.
4. **Task IDs still collide when two agents reserve in parallel.** At 13:44 a child reserved T349 (T340c); a minute
   later the DRI reserved T349 for the T340 check. The first push won and the check became T350 (c2b08a3e). Pull just
   before taking max+1, push at once, and on a rejected push re-take the number instead of rebasing the duplicate.
5. **Independent re-checks paid off.** T334 (second agent, own code) confirmed T324's two largest error-budget terms
   (line-set product: bias -.56 pp, MSE -41%; tmax: -.18 pp, -20%). T327 reproduced T319's smoothed-closure negative
   with independent code, so the DRI could mark it dead (verified, scoped). T312 re-implemented R1W V1 from its README
   (80 DEV2 within .012 pp), and T320's recount found ~9% latitude in the FLOP count. Each was a light task, and each
   either upgraded a claim or found a gap. Pair every claim that changes priorities with a check by a different agent.
6. **Exact cost cuts have reached the floor; approximations trade 1:1.** Cuts that keep the estimates identical (B2
   prefix sharing, continuity restarts, merging touching pieces) took the walk from ~12-13k to ~5.4k FPE; T317f's
   declared margin screen then reached 3.3k. The 3-line geometry's label combinatorics alone cost 2.05k FPE, and the
   oracle exact-estimate floor is ~2.2k (T326). Every declared approximation removed so far cost at least as much in C
   as it saved in MSE (Fable r5: V2 C x2 for MSE x.86; tmax .5 -> 1 C x1.27 for MSE x.80), so MSE x C does not
   improve. New components still arrive with slack until they are folded into the walk: T353 cut the spread+tilt arm
   from 6,542 to 5,079 FPE with bitwise-identical estimates (DEV2 F 2.60 vs T317f 3.14), though on DEV3 it does not
   beat T317f (F 3.32 vs 2.49; T355). Strict
   (F <= 1) needs a new representation, not tuning [inferred]; Fable r5 put the lineage's odds at ~5%.
7. **Track bias and variance separately, and check the split out of sample.** On DEV2, T339's spread-lows closure
   cut the model-to-model error sd (.667 -> .455; corr with truth .978 -> .991), yet RMS rose (.827 -> .884): the
   estimates shifted up +.27 pp, because the mean-point closure's negative bias had been cancelling part of the walk
   approximations' ~+.6 pp positive bias. A closure that lowers variance can expose bias that was cancelling. On fresh
   DEV3 (T343, another agent) the bias shift (+.22) and the corr gain replicated but the variance cut did not (sd ratio
   .912 [.76, 1.11] vs .68), so the "variance-only F ~1.06" reading was withdrawn. Report bias, error sd and corr for
   every arm, not only RMS or F. (T351's tilt then removed the bias, +.49 -> +.04 pp, at +2.8k FPE as executed.)
8. **Owner Q&A lives in [OWNER_QA.md](OWNER_QA.md).** Every owner question gets an entry with its time and the answer as
   given; when a result changes it, add a one-line "Later updates" pointer to the INDEX rows (T396: results live only in INDEX). Directives are recorded verbatim in its section 8, each
   with a one-line note of what was done. Update it in the same commit as the board change a directive causes (T345,
   T346, T352).
9. **Do not let a lane wait on a silent agent.** The Codex lanes, including the Astra reviews, have been silent since
   about 08:30; Astra round 3 never arrived (T301 closed). The Fable reviews ran as no-compute DRI sub-agents (five
   rounds, 05:16-12:13) and supplied many of the tested designs (round 3: T305-T309; round 4: T317-T319, including
   the continuity-restart saving behind T317; round 5: T326-T332). Give a silent lane's Ready rows to child sessions
   after a stated timeout, and run reviews that need no compute as sub-agents.
10. **Idle sessions lose their running jobs (10-19).** A cloud container is reclaimed once its session goes idle, and
   background processes die with it; two T398 workers lost (9,3,2) runs this way. Every long run needs a background-Bash
   waiter (`while kill -0 <pid>; do sleep 120; done`, re-armed before it times out), checkpoints with `--resume`
   (T398: every 5 parents), and a send_later self check-in that restarts dead PIDs and pushes checkpoints. Put this in
   every compute-heavy spawn prompt.

**Owner directives** are recorded once, verbatim, in [OWNER_QA.md section 8](OWNER_QA.md#directives); the ones that bind agents (no Fable, never contact ARC, parallel child sessions, CPU-first, tutorial paused) are one-liners in [AGENTS.md "Standing rules"](AGENTS.md#standing-rules). Record a new directive there only, never by copying it here or into TASKS.md.

**Tutorial update procedure (DRI process, from the 10-19 00:1x tutorial request; paused by the owner at [10-19 06:4x](OWNER_QA.md#d-1019-064x)):** keep the interactive explainer (T374, session_01XkxtNU1pcq7TYNKYqvPhSg) current: after registering notable results, send T374 one batched update so it re-publishes to the same Artifact URL; ask for a final pass when all lanes are done. If the session is gone, start a new one from claude/t374-tutorial (tutorial/README.md has the URL and rebuild steps).
