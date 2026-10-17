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
- **Decisive open question:** does that hold along T? The final experiments (`m16/approaches/final3/`): (8,4),
  (8,5), then projection/attempt at (16,10).
- **ARC's own position (public, 2026-08-17):** "not much closer to succeeding at the challenge". Their M_4,3
  estimators were 2.4-10x *worse* than sampling in MSE and they called that "roughly match" (E32). The private claim
  that ARC cracked M_16,10 is doubtful on public evidence (H-E5).

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

## 6. How to continue

1. Read `m16/approaches/final3/README.md` (the (8,4), (8,5), (16,10) experiments). If cost grows slower than the
   region count along T, continue the main ladder per TASKS.md; if not, the boundary/cell approach needs a new idea
   (e.g. merging cells across positions with shared structure, symbolic dynamics over records, T110 class-region
   families).
2. Open reviews / checks: T113 (E43), T114 (E44), T80 (E35), T57 (E34). Finish them before claiming more.
3. Keep the leaderboard honest: any new estimator for M_16,10 goes through `m16/leaderboard.py` v3.
4. Ask ARC (T41, needs owner approval) for the M_16,10 protocol if a candidate solution appears.
