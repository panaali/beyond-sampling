# Working in this repo (rules for all agents: Claude, Codex, humans)

↑ [Project README](README.md) · [TASKS](TASKS.md) · [INDEX](INDEX.md) · [Approaches](m16/approaches/README.md)

Start by reading [TASKS.md](TASKS.md) (what is in progress, who holds it, and the backlog), then [`INDEX.md`](INDEX.md):
questions, finished experiments, where outputs live, and the **hypothesis ledger** (the status of every claim). [`m16/FINDINGS.md`](m16/FINDINGS.md) is the detailed narrative for the AlgZoo 432-param model.

## Rules

0. **Claim work on the task board first.** Before starting anything, claim a task in [TASKS.md](TASKS.md) by moving it to
   *In progress* and committing and pushing that edit alone (protocol in TASKS.md). Never start work that is already
   claimed. When done, remove your row and register the result in [INDEX.md](INDEX.md).
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
5. **Report negatives.** Failed ideas and null results go in the README and the ledger. Do not delete them.
6. **Shared evaluation.** Estimators for the ARC challenge must use [`m16/arc_eval.py`](m16/arc_eval.py): the same noisy
   copies, ground truth and estimator interface `est(model, budget, rng) -> (estimate, evals)`. State
   whether the method is deductive (no random model inputs) or sampling-based.
7. **Independence.** Do not read Alignment Research Center material about AlgZoo published after
   January 2026 (we are doing an independent attempt to compare with theirs).
8. Keep runs CPU-light (4 shared cores). Large data goes in `data/` (git-ignored).
9. **Never kill processes by name pattern** (`pkill -f`, `pgrep -f ... | xargs kill`): other agents run
   scripts with the same names. Kill only PIDs you started and have checked (e.g. by their working
   directory, `ls -l /proc/<pid>/cwd`).

## Environment
`source .venv/bin/activate` (CPU torch, transformers, alg_zoo installed editable from
`/home/user/alignment-research-center/alg-zoo`). Ground truth for M_16,10's accuracy: [`m16/truth.txt`](m16/truth.txt).
