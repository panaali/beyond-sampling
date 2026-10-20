# T459: DRI takeover artifact readiness, 2026-10-20

## Question
Which of the handoff's seven priority lanes can advance cheaply from published evidence while Claude service is unavailable? Original claim alias: T11598, subsequently corrected by the DRI to T459.

## Method
Read AGENTS, TASKS, STATUS, INDEX and DRI_HANDBOOK §12 at shared baseline `ab676be36f4719fcd59714782c6e5e5d5a00761d`. Listed remote heads, fetched only the relevant source branches without checkout, and inspected README/source and tree metadata with `git show`/`git ls-tree`. Inspected only record counts/keys of T370's explicitly truth-blind partial estimates. No sealed truth payloads, model calls, author imports or experiments. Commands to reproduce the inspection: `git ls-remote --heads origin 'claude/t420*'` (substitute each lane); `git ls-tree -r --name-only PIN -- m16/approaches/FOLDER`; `git show PIN:m16/approaches/FOLDER/README.md`.

## Status
Bounded review complete. Published evidence is a snapshot, not evidence that a Claude session or computation is currently alive. No held implementation lane was taken over, no claims promoted, and no branch merged wholesale.

## Results: exact inspected pins

| Lane | Branch under `origin/claude/` | Full commit |
|---|---|---|
| T370 estimator | t218-tie-tail-labels | `374d646f6b0d10a003531decbb712e83c82b442d` |
| T370 selector | t286-cheap-frontier | `4684ef59da6428a41290218ebe053c2da47efb9d` |
| T420 | t420-geom-audit | `a0133ca52b8245da58e57662a47ae8a25b393cf8` |
| T434 | t434-walk-noisy | No matching remote `claude/t434*` head at inspection |
| T442 | t442-walk-rules | `f4547a875274ef2ae39ae8e2d1bb71534b576ded` |
| T452 | t452-geom-faster | `7b5f5343dde1a1efdeb9a808c6248b2344bf16c8` |
| T455 | t455-late-triple-bound | `4e29c33861a0983535cd9bc80d660cfd9f2dd412` |
| T456 | t456-is-v4 | `a52da0d6a57d29aa2f76daa74c3360222f33aec2` |

**T370 — stage 2 incomplete.** `r1_cont/` publishes the Q16 freeze manifest, estimator, guarded stage-2 runner and partial outputs. Latest README records stage-1 completion at 28/32 E2 within 2x; board's 25/29 is an earlier checkpoint. Saved stage-2 files contain one E1 model (two thresholds) and six E2 cases. `rare_events/` publishes all sixteen K3 model files, selection code and the minimum-50-truth-hits replacement rule. Its tree has no Q16 per-model result files or `t370s2_selection.json`. Thus the required 128 candidate estimates and eight-pair selection are not published here. Preserve the freeze and truth separation; completion requires missing estimates and later independent truth custody, not simply running the selector. Source-only contract review is possible locally. Q16 is explicitly QMC-class cubature; it does not establish a deductive solution.

**T420 — audit evidence exists, final audit absent.** `geom_audit/` publishes preregistration, independent geometry code, fixed `cells.json`, partial B/C result JSONLs and crash records. README remains “running,” with Method unfilled and no written gap list or final verdict. Latest commit changes exact-volume accumulation to fit memory. Saved-record coverage/error reconciliation and source review can proceed locally; completing cell comparisons requires the referenced survivor/region inputs and bounded compute. Do not clear T416's provisional geometry status from this snapshot.

**T434 — artifact recovery needed.** The board names `walk_noisy/` and a branch that is absent from the remote head listing; its folder is also absent from the inspected shared tree. This establishes only publication unavailability at these locations. It does not prove no work happened or that another location lacks it. Recover a producer pin, preregistration and outputs before assuming a restart is needed.

**T442 — results checkpoint, not final analysis.** `walk_rules/` publishes frozen code, preregistration, gate log, configuration/analysis scripts and many run JSONLs, but no `analysis.json`/`analysis.log`. README's “scoring not started” is stale relative to the checkpoint commit. A cheap completeness/hash audit is available. Scientific analysis additionally needs the declared DEV3 truth dependency and frozen baseline rows; no full coverage or gain is inferred from filenames. Rule changes also change pilot allocation; a model bootstrap does not by itself cover rule-selection uncertainty.

**T452 — small measurements available; adoption gate missing.** `geom_faster/` publishes implementations, profile/speed/double-double summaries and gate runner, but no `results_gate_*.jsonl` or tier-2a measurement output. The board's small-sample speed report is not the required ≥10,000-cell identity gate. Tier 2a changes outputs and remains separately subject to T420 review and T400 adoption. Its README calls the reference “T420-audited”; the inspected T420 snapshot does not supply final clearance. Source and small-summary review is possible; broad adoption is premature.

**T455 — source available; certificate not established.** `late_triple_bound/` contains only README and four Python files, with no saved validation/search/cost result. `rmass.py` implements float inflation plus an arb comparison implementation; README explicitly says CROWN pruning has no outward rounding. Finite mass comparisons and sampled pruning checks cannot establish universal rounding soundness. A proof/assumption review can proceed locally; restarting long runs would require dependencies/checkpoints and still would not resolve that proof gap.

**T456 — complete author record, ready for cheap independent review.** `is_v4/` publishes preregistration, estimator/runner/scorer, panel, fifteen estimate files, fifteen already-published truth files, scored summary/table and logs. No truth payloads were opened for this readiness review. README reports the fresh-panel pass; INDEX already registers it as unverified. The heavy-tail gate never fired, and panel representativeness and repeated-run coverage remain open.

## Highest-value next action
Independent T456 saved-evidence audit: freeze the pin; check ancestry of preregistration and estimate freeze; independently recompute fifteen ratios, combined-error z scores, flags, both F bars and comparator counts from published records; inspect the scoring formula/cost contract. Cap at one small script, ≤10 CPU-minutes, no model imports, RNG, new truth generation or new experiments. Deliver discrepancy table and scoped verdict; leave calibration and generalization unconfirmed. The DRI has separately claimed this as T461.

## Compute accounting
Git/text/metadata inspection only; zero model forwards, geometry solves or scientific experiments. No experiment costs were incurred or estimated from this review.

## Claims, alternatives and failure modes
[measured] Named artifacts exist or are absent at the pinned locations. [inferred] T456 offers the shortest independent check of a completed priority result. Unpublished progress and alternate branches remain possible; recover artifacts before duplicating compute. Passing saved arithmetic would verify report consistency, not estimator unbiasedness, interval coverage, runtime provenance or ARC acceptance. Snapshot absence is not a negative scientific result.
