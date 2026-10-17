# Working in this repo (rules for all agents: Claude, Codex, humans)

↑ [Project README](README.md) · [TASKS](TASKS.md) · [INDEX](INDEX.md) · [Approaches](m16/approaches/README.md)

Start by reading [TASKS.md](TASKS.md) (what is in progress, who holds it, and the backlog), then [`INDEX.md`](INDEX.md):
questions, finished experiments, where outputs live, and the **hypothesis ledger** (the status of every claim). [`m16/FINDINGS.md`](m16/FINDINGS.md) is the detailed narrative for the AlgZoo 432-param model.

## Rules

0. **The task board [TASKS.md](TASKS.md) is mandatory for every agent (Claude, Codex, humans).** Many agents work
   in this repo at once; the board is how duplicate work is avoided.
   - **Reserve before you start.** Before writing any code or running any experiment, `git pull`, check that the task
     is **Ready** (not Blocked) and not already *In progress*, move it there with your owner ID, date and folder, and **commit and push that edit
     alone**. If your idea is not on the board yet, add it as a new backlog row and reserve it in the same commit.
     Never work on a task someone else holds.
   - **Keep it current.** Update your row's *Notes* when the plan changes. If you stop or get interrupted, release the
     task (move it back to Backlog with a note on how far you got) so others can pick it up.
   - **Add every new idea to the Backlog.** Whenever you discover a new approach, a follow-up, a needed check or a bug in
     shared code, add a Backlog row (next free T-number; IDs are never reused) with why it matters, when it counts as
     done, and pointers, even if you will not do it yourself.
   - **IDs are allocated on push.** Before using a new T-, E-, H- or L-number, `git pull` and take max+1 of what is
     on the remote; push the row immediately so the number is reserved. Duplicates found later are renumbered by the DRI.
   - **When done,** remove your row from *In progress* and register the result in the [INDEX registry](INDEX.md)
     (or ask the DRI to), including negative results.
1. **Stay in your lane.** Write only inside your assigned folder, e.g. `m16/approaches/<name>/`. Don't
   edit shared code ([`m16/common.py`](m16/common.py), [`m16/arc_eval.py`](m16/arc_eval.py), ...) or other agents' folders. Propose changes in
   your README instead.
2. **Every approach folder has a `README.md`** with these sections:
   - **Question**: what this tries to answer.
   - **Method**: plain-language description plus the key files.
   - **Status**: running / done / abandoned (with the reason).
   - **Results**: numbers, with the exact command that reproduces each one.
   - **Compute accounting**: what counts as compute (model forward passes, FLOPs, input generation,
     LP solves) and the totals.
   - **Claims**: each claim marked `[measured]`, `[inferred]` or `[speculative]`.
   - **Alternatives not ruled out**: other explanations for your results, and checks you did not do.
   - **Failure modes**: where the method breaks.
3. **Be skeptical of your own hypotheses.** Before building on an idea, write down at least two
   alternative explanations and the cheapest test that would separate them. A hypothesis that "fits"
   is not confirmed. Prefer tests that could have failed.
4. **Nobody marks their own claim "confirmed".** Only the DRI moves a claim to confirmed in [`INDEX.md`](INDEX.md),
   after an independent check (a different script, seed, data or agent).
5. **Report negatives, and verify them before calling an idea dead.** Failed ideas and null results go in the README and
   the ledger; never delete them. A negative is only **"negative (unverified)"** until (a) the README lists every variant
   tried and every variant NOT tried with why (an exhaustion checklist), and (b) an independent agent re-checks the key
   failure with different code and tries the most promising untried variant. Only then may the DRI mark it **"dead
   (verified)"**, scoped to the method and assumptions actually tested. An implementation defect, finite sweep,
   precision limit or exhausted compute budget is not broad impossibility; unexamined plausible alternatives remain
   open/inconclusive in the backlog. A claim excluding every conceivable variant requires a proof.
   **Keep progressing:** an agent that finishes a task (positive or negative) moves on to the next eligible Ready
   backlog row. Root reserves/pushes coordinated claims before implementation, or explicitly delegates claim
   publication. Codex workers have standing delegated authority under TASKS.md to publish a TASKS-only claim
   from a fresh isolated administrative checkout of the latest shared branch, then notify root. Continue useful review or analysis
   while that handoff is arranged. Respect existing CPU/memory limits; continuity does not authorize empty polling,
   duplicate work or unbounded runs.

6. **Shared evaluation.** Estimators for the ARC challenge must use [`m16/arc_eval.py`](m16/arc_eval.py): the same noisy
   copies, ground truth and estimator interface `est(model, budget, rng) -> (estimate, evals)`. State
   whether the method is deductive (no random model inputs) or sampling-based.
7. **Independence (relaxed by the project owner on 2026-10-17).** Any source may now be read, including ARC
   material about AlgZoo published after January 2026 (ARC's M_16,10 solution is reported to be unpublished). To
   keep the comparison honest, **log every post-January-2026 ARC/AlgZoo source you read** (URL, date, what it
   says that is relevant) in [lit/web_log.md](lit/web_log.md), and mark results that depend on it.
8. Keep runs CPU-light (4 shared cores): at most 2 heavy tasks at once, tracked by the *CPU* column in [TASKS.md](TASKS.md).
   **Memory: at most 2 GB per process** (15 GB total, no swap; a container restart on 2026-10-17 was most likely memory
   exhaustion). Process samples in chunks; start long runs with `ulimit -v 2500000` (KB). Large data goes in `data/` (git-ignored).
9. **Never kill processes by name pattern** (`pkill -f`, `pgrep -f ... | xargs kill`): other agents run
   scripts with the same names. Kill only PIDs you started and have checked (e.g. by their working
   directory, `ls -l /proc/<pid>/cwd`).

## Environment
`source .venv/bin/activate` (CPU torch, transformers, alg_zoo installed editable from
`/home/user/alignment-research-center/alg-zoo`). Ground truth for M_16,10's accuracy: [`m16/truth.txt`](m16/truth.txt).
