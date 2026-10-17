# Task board

↑ [Project README](README.md) · [STATUS](STATUS.md) · [INDEX (registry + ledger)](INDEX.md) · [Rules (AGENTS.md)](AGENTS.md)

**Read this before starting any work.** It is the single place that says what is being worked on right now and
what is waiting. Finished work lives in the [INDEX registry](INDEX.md#3-experiment-registry); claims and their status live
in the [hypothesis ledger](INDEX.md#4-hypothesis-ledger).

## How to claim a task (all agents and humans)

1. **Pull first** (`git pull`), then check **In progress** below. Do not take a task someone else holds.
2. **Claim before working:** move the task's row from Backlog to In progress, fill in *Owner* (session ID or name),
   *Started* (date) and *Folder*, then **commit and push that edit alone** before doing anything else. If the push is
   rejected because someone else pushed, pull and re-check; if they claimed it first, pick another task.
3. **Work only in your folder** (rules in [AGENTS.md](AGENTS.md)). Update *Notes* if the plan changes materially.
4. **Finish:** remove the row from In progress, add your experiment to the [INDEX registry](INDEX.md#3-experiment-registry)
   (or ask the DRI to), and list any new tasks you discovered in Backlog.
5. **Stale claims:** a claim with no commit in its folder for 24 hours may be taken over. Note the takeover in *Notes*.
6. New ideas go into Backlog with the next free ID (T-numbers are never reused).

## In progress

| Task | Owner | Started | Folder | Notes |
|---|---|---|---|---|
| *(none: all agents finished on 2026-10-17)* | | | | |

## Backlog (highest priority first)

| ID | Task | Why / done when | Pointers |
|---|---|---|---|
| T1 | **Stochastic reduction of the tie margin** (deductive): carry the low inputs as a distribution and track the exact law of the residual tie margin given the top values/positions | The one untried deductive direction (suggested by E23 and E27). Done when an estimator is scored on [leaderboard v2](m16/leaderboard.py) with a verdict | Ledger H-E9, H-E10, H-E12; [bandwidth](m16/approaches/bandwidth/README.md), [sci_poincare](m16/approaches/sci_poincare/README.md) |
| T2 | **Explain the T~5 collapse**: why the tie-band integrand becomes heavy-tailed; repeat the H x T scan with 3+ training seeds per (H, T) | Turns H-E11 from "supported, one seed per cell" into a firm result, and points at what a solution must handle | [sci_tao](m16/approaches/sci_tao/README.md) (`scan_HT.json`), ledger H-E11 |
| T3 | **Symbols from rotating-neuron gate switches** instead of records (Poincare's untried variant) | Ceiling test like E27: done when the reduced system's exact bias is measured on all 18 models | [sci_poincare](m16/approaches/sci_poincare/README.md) |
| T4 | **Re-score leaderboard v2 original-model rows** with the pooled 28M truth | v2's orig rows used the old truth (~0.025 pp low). Done when the orig rows in [results/leaderboard.json](results/leaderboard.json) are refreshed | [m16/leaderboard.py](m16/leaderboard.py), [results/leaderboard_truths.json](results/leaderboard_truths.json) |
| T5 | **Brute-force exact reference** (needs much more compute): first test whether exact 10-D cone masses can be computed fast and accurately on a sample of regions | Gives an exact answer to compare with ARC; benchmark only. Go/no-go after the cone-mass test | INDEX open questions; [cone_enum](m16/approaches/cone_enum/README.md); ledger H-M7 |
| T6 | **Qwen H2 re-run**: >=100 controls (top neurons projected out, or dense rotation), 2-3 explainers per item, n-gram dedupe for judges | Resolves the "inconclusive" verdict on the essay's claim | [RESULTS.md](RESULTS.md), [review](review/README.md), ledger H-Q1..Q4 |
| T7 | **Finish M_32,10 (E5)** and **sweep the label constant budget** (1, 3, 5, 10, 20 constants) | Tests whether "legible but not understood" scales with size or is a constant-budget artifact | [algzoo](algzoo/README.md), ledger H-A1 |
| T8 | **Causal/orthogonal test of H-M3**; check the H-M1 phase clock across other common gate sets | Firms up or withdraws remaining mechanism claims | [m16/FINDINGS.md](m16/FINDINGS.md), [m16/modes.py](m16/modes.py) |
| T9 | **Score ARC's method** on our harness when it is shared | The comparison the project was set up for. Needs ARC's estimator in `est(model, budget, rng)` form, or their numbers | [m16/arc_eval.py](m16/arc_eval.py), [m16/leaderboard.py](m16/leaderboard.py), ledger H-E4/H-E5 |

## Recently finished (details in the INDEX registry)

E10-E27, LB2 (leaderboard v2), R1 (review): see [INDEX section 3](INDEX.md#3-experiment-registry) and the
[approaches index](m16/approaches/README.md).
