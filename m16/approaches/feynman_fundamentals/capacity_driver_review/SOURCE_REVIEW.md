# T246: published capacity-driver source review

Frozen author head: `3e5e09a1`. Eleven exact Git text blobs are retained. This supersedes T237's absence observation at `f02482d6`, without making that earlier time/head-scoped observation false. No author module, test, training, checkpoint, weight array, results array, MC, RNG, GPU or Colab call was executed here. Findings below describe source behavior; occurrence in a historical run is not established.

## Improvements over the earlier source inventory

The new driver supplies an explicit final evaluator with default n=100,000,000 and a dedicated generator seed777. It also saves parameter tensors, AdamW first/second moments, completed-step count and learning curves in state.pt. The original T237 description of the old alg-zoo state_dict-only helper must not be applied to this new file as if it still saved weights only. Nevertheless, the new saved state is not exact full-run continuation, for the reasons below.

The ideal recurrence and scalar learning-rate formula match the intended no-bias RNN and relative-horizon schedule. Unweighted enriched training is explicitly a different input treatment, evaluated under the plain Gaussian law. Positive/negative sign conditioning in the blind sampler is consistent with the iid Gaussian sign-law: its T+1 equally likely sign configurations consist of all-negative and each possible single-positive coordinate, with independent absolute-Gaussian magnitudes. The near-tie sampler instead edits a sorted Gaussian vector; it is not a draw from the Gaussian law conditioned on a small rank gap. That is permissible as a disclosed enrichment treatment, not a conditional-Gaussian claim.

## Actionable source findings

### 1. T2 table has an incompatible third-order-statistic access

`ht_table.py` explicitly loops T=2,...,10 and calls evaluate for every nonempty model list. `evaluate.py` unconditionally assigns `third = order[:, -3]`. An order array with two columns has no such index. Thus this evaluator cannot complete a nonempty T2 model batch as written. The T10 capacity run does not have this indexing defect. A T2-specific anatomy convention or omission of the unavailable third-rank category is needed before that table path can work. No T2 model was loaded or evaluator run in this review.

The table writes its JSON only after finishing all T values. Its broad zoo-loader exception can silently omit unavailable seeds; each reported mean is therefore over the names that actually loaded, not necessarily five seeds. `q_ht.sh` lacks fail-fast handling and appends HTDONE even after earlier command failures; that marker alone cannot prove a complete table. Per-stage exit status and expected T/seed coverage are needed for a result audit.

### 2. Resume restores optimizer moments but deliberately changes the input stream

`Trainer.state()` returns p0..p2, m0..m2, v0..v2 and step. `run.py` adds curve. It does not store generator state, full schedule/mixture configuration, source identity or seed-list binding. `load_state()` restores parameters/moments/step, then reseeds with `data_seed*1000003 + saved_step`; it does not resume the original RNG state. This is an optimizer-state continuation on a restarted data stream, not bitwise continuation of an uninterrupted run.

Automatic reuse is triggered solely by a state.pt at the chosen tag. No stored configuration is checked against current H/T/seeds, schedule, mixture, horizon or source. Same-shaped states can therefore be relabeled or continued under changed arguments without a rejection. This is a reachable source hazard, not a claim that it occurred. A later audit needs the actual configuration and interruption record. Changing the horizon also changes the future schedule; names alone cannot establish that a continuation kept the prior treatment.

The most recent periodic state is not the final state: saving is conditional on `s % 2^18 == 0 and s < n_steps`. Rerunning a completed tag can reload an earlier checkpoint, use the new resume stream, and overwrite final result/weight paths. Preserve attempts and checkpoint hashes rather than infer identical outcomes from equal tags.

### 3. Finite CPU equivalence does not cover the CUDA lifecycle

`test_equiv.py` uses graph=False, two seeds, batch64 and300 steps. It compares final parameter differences, not every intermediate step despite its docstring. Its exit tolerance is1e-4; the README's reported2.4e-7 measurement is not independently reproduced or accompanied by a raw output artifact in this frozen folder. CUDA/library/precision behavior and interrupted continuation remain outside that finite check.

Reviewer correction: a preliminary message incorrectly treated the tensor operations inside CUDA stream capture as executed updates and inferred an extra step. Capture records device operations for replay; the source does not establish that alleged step offset. That inference is withdrawn. Three warm-up bodies do execute before restoring parameters/moments and resetting the step and generator. No captured graph or training body was run in this review.

### 4. Evaluation is common-input across models and repeated across arms

The evaluator draws one x batch then evaluates every model on it. Each call resets its dedicated generator to the default seed777; run.py exposes no evaluation-seed option and calls evaluate without overriding it. Within one fixed shape/runtime, comparisons therefore use common test inputs, including repeated uses across arms. Pointwise per-model SE is ordinary conditional binomial bookkeeping under the ideal iid interpretation, not a simultaneous or selection-adjusted interval. Paired-difference uncertainty requires discordance/covariance information; it cannot silently use independent-stream variance.

The raw integer counters exist internally but are discarded from the return object in favor of proportions. Paired correctness/discordance counts, checkpoint byte hashes, exact runtime/source and actual stream state are not serialized in the final result. Rounded proportions and a nominal seed are weaker evidence than the integer counts and pair table requested by T237. No existing MC outcome or accuracy claim has been audited here.

The anatomy implemented is disjoint: all errors in the <=1-positive event are blind; outside it, prediction of the third-largest is class23, prediction of the largest is class12, and other wrong labels are other. These are error-label categories, not bins defined by a numerical near-tie distance. If the owner wants near-gap anatomy too, it requires a separately specified gap/overlap convention.

### 5. Selection, schedule and compute claims remain qualified

The README now chooses best-seed gains for the two-doubling stop and a best-seed >3SE selection rule for schedule variants. These are operational decisions. They do not by themselves control repeated/best-of-seed selection error or establish two-sided equivalence, convergence or architecture ceilings. The same fixed evaluation stream is reused; it is not an untouched final confirmation stream after all choices. Owner-approved decisions are unchanged by this review.

Longer n_train still changes relative-horizon decay as documented in T237. `trainer.py` has real-arithmetic AdamW-equivalent expressions but a different floating operation order and batched kernel from torch AdamW/nn.RNN; the finite CPU check is useful but does not prove CUDA/library/TF32 determinism or full-run equivalence. The training driver does not establish those runtime settings by source assertion. The README explicitly acknowledges different training streams.

Reported wall_sec begins after Trainer construction/capture and covers only the current invocation's loop. It excludes initialization/capture, prior interrupted invocations and final evaluation. A resumed curve retains old points with elapsed times from earlier invocations. Total Colab units and the30/55/60-unit thresholds are external operational accounting, not enforced by these Python loops. This is not a claim the owner exceeded the budget.

## Disposition and follow-up boundary

The source is materially more reviewable than the earlier static contract, but is not cleared as an exact zoo reproduction, bitwise resume or complete multi-T evaluation. The T2 path and resume provenance deserve owner attention; the remaining provenance/selection issues qualify interpretation. No shared code was edited, no training was stopped, and no new run is requested. The owner can decide which corrections and records fit the authorized study. A later bounded saved-result review should bind the executed source, final weights, integer counts, interruption records and actual settings before attributing gains or claiming a ceiling.
