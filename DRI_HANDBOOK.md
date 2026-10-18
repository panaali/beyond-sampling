# DRI handbook: how this project was run, and what we learned

↑ [Project README](README.md) · [STATUS](STATUS.md) · [TASKS](TASKS.md) · [INDEX](INDEX.md) · [AGENTS rules](AGENTS.md)

Written by the outgoing DRI (Claude) on 2026-10-17 so that the next DRI (Claude, Codex or human) can pick up the
project without the chat history. Read this, then STATUS.md, then TASKS.md.

## 1. The goal, and where it stands

- **Challenge (ARC, AlgZoo):** compute the accuracy of the 432-parameter 2nd-argmax RNN M_16,10
  (`alg_zoo.example_2nd_argmax()`, accuracy 0.95313575 for x ~ N(0, I_10)) *mechanistically* (no random or
  pseudorandom inputs), with MSE at least as good as plain sampling at equal compute. Must also work on the 5
  trained seeds and on noisy weight copies.
- **Status:** not solved. Every deductive estimator on M_16,10 loses by 37x to >4000x (leaderboard v3, LB3).
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
- **Why deductive methods fail on M_16,10:** >1e7 linear regions (H-M7); the error mass sits in rare, wide,
  irregular tie bands (H-E11/H-E12, heavy tails from T~5); deterministic summaries of low inputs are Jensen-biased
  (H-E10); ARC's cumulant propagation breaks because a scalar input per step makes hidden units perfectly
  correlated (E35); harmonic projection leaks ~1/D across the tie wall (E34).
- **What made the small-model recipe work (E36-E43):** the model has no biases, so correctness depends only on
  x/|x| and errors are unions of cones; organise by true boundaries x_i = x_j (half-chambers), follow gates step by
  step, resolve only gates whose switch could change the answer (impact / Lipschitz margin bound through the
  remaining layers, scale-invariant), compute exact cone masses (Girard in 3-D; Plackett/triangulation in 4-D).
  Make every quantity invariant to function-preserving neuron rescaling and permutation, and test that permanently.
- **Where it breaks:** freezing intermediate gates without a margin certificate (fixed for one step by RECIPE112;
  multi-step version T108); 4-D cone masses cost ~300x 3-D (T109); exact boundary tracing at T=3 costs ≥ the region
  count (E39) — any gap can only open along T.
- **Quadrature (fixed Sobol tie slicing) wins at short T and on mild noise** (sci_tao, sci_feynman) but ARC would
  likely not count it as mechanistic.

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
| (16,10) target | 432 | all | **fail**: deductive methods 37x->4000x worse than sampling (LB3); recipe projects >=1e17 ops [CORRECTED 2026-10-18: an extrapolation from two growth steps, ~6.5e16-4.7e17 (trend ~2e23), not a bound] |

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
  825 vs 66 at (4,3); 1.3e36 vs ~1e7-1e8 at (16,10). Trained nets use a tiny fraction of possible regions because
  their neurons switch in coordinated ways (running max, record alarm, clocks) — that coordination is the structure a
  solution must exploit.
- Any per-gate-pattern method (enumeration, fibre tracing, cell splitting) pays at least the number of distinct gate
  histories it must distinguish; at T=3 exact boundary tracking costs >= the region count (E39). The cost must be cut
  by *merging* histories that do not matter for the answer, not by cheaper bookkeeping.
- Compute is not the main blocker for a single exact answer (a GPU could plausibly do ~1e17 branchy ops in hours-days),
  but (a) the real trend is ~1e23, (b) irregular geometry runs at 1-5% of GPU peak, (c) at 1e17 ops sampling is precise
  to ~5e-6 pp so the method must be essentially exact, and (d) brute force at that scale is not what ARC is asking for.

## 8. Ideas to try next (brainstorm, prioritised; board rows T187-T193)

All aim at the one isolated failure: gate histories multiply with every extra time step.
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
   current recipe can be rescued).
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
- **The recipe is now a diagnostic** (which splits matter, T195), not the method to scale; T170 is deprioritized.
- **Protocol:** TASKS DRI directive 7 fixes the rung test (5 seeds + >= 8 noisy copies per seed at absolute sigma 0.05
  and relative 0.005, 0.01; exact or >= 1e8 independent-stream MC truth; T196 FLOPs; every budget with a CI).
- **Leaderboard caveats (repairs in T204):** LB3's summary takes the min over budgets (biased low; quote each budget
  with its CI); `table()` demotes arc_cumulants to "sampling" because its rows were stored before it entered CLASS
  (det_ok=False); srb's win rests on a hand-counted, uninstrumented cost factor 3.3/1.6 ≈ 2.06
  (`control_variate/rb9.py:158-184`), without which its ratios are ~0.4-1.1.
- **ARC:** H-E5 is now "reported, unverified": the owner's ARC contact says a newly joined ARC member has an M_16,10
  solution and that ARC is waiting for a comparison. Owner ruling 2026-10-18: "Unless we have a candidate solution, we
  should not contact ARC." The comparison package is T203; contact is T41.
- **Read:** [QA_2026-10-18](review/dri_handover/QA_2026-10-18.md) (previous DRI's answers), the DRI directives at the
  top of [TASKS.md](TASKS.md#in-progress), [T196 cost convention](review/cost_convention/README.md).
