# T113 recovery and independent completion audit

## Question
What does the existing E43 independent review actually establish, which saved calculations are reproducible, and what remains before the DRI can confirm the claimed smaller-case success and scaling trend?

## Method
TASKS claimc12ff831 pushed before implementation, taking the highest eligible released review in the revised DRI priority queue. Original review files remain read-only. Full source and about2.7MB saved outputs were found in `/workspace/work/admin-t110-t4/review/recipe112`; `/workspace/clean-neurons` lacks several helpers and outputs. Preserve originals with hashes, Git provenance and a local snapshot rather than rerunning completed sweeps. First independently recompute every saved fresh-panel mean-square error, original ratio, per-noise-seed ratio, own-truth/reimplementation discrepancies, rescaling changes and cell/region summary from rows. Recompute the alternative mean of per-model sampling variances at each recorded estimator cost, showing the difference from the original mean-cost convention. Reproduce the count-panel adjustment from its125 recorded rows and state clearly that this adjustment is a panel-level transfer, not a native recount of every fresh model. Recover sensitivity/adversarial/MC evidence with explicit limitations and no selection of better candidate parameters.

Alternative1: raw accuracy errors support the review while counter conventions change the conclusion (separate arithmetic/variance and cost data). Alternative2: apparent region savings are chart-counting artifacts (retain both gate regions and half-chamber pieces). Alternative3: saved truth/reimplementation agreement masks shared tie or tolerance defects (source audit and a separately frozen analytic fixture panel after inventory). Alternative4: missing provenance prevents reproduction (hash all sources/artifacts, identify historical source revisions and preserve unknowns).

No model evaluations, new noisy panels, MC or algorithm changes in this first aggregate phase. A missing artifact is not numerical failure; preserve partial state. New direct checks need a concrete frozen gap-driven scope. Independent reviewer checks aggregate/source findings before DRI confirmation. No shared source or original review edits.

## Status
Recovering committed artifacts and independently auditing raw summaries. T162 handoff complete; T154 independently reviewed. This folder owns only recovery/audit work.

## Results
Pending. Reproduce scripts with `/workspace/clean-neurons/.venv/bin/python`, one thread,2GB and review lock for<=20CPU phases.

## Compute accounting
Prior runs are research dependencies and never counted free. Inventory saved wall/CPU fields separately, with missing costs explicit. New aggregate checks record CPU but do not evaluate models. Legacy counts, scalar-source counters and complete implementation cost are distinct.

## Claims
[inferred] Existing raw rows can support an independent arithmetic audit without repeating expensive experiments. No E43/H-U4 confirmation by this author.

## Alternatives not ruled out
Additional seeds/noise, correct accounting of all model-dependent operations, rigorous floating geometry, other gate-freezing certificates, training provenance and parameter-sensitive adversarial failures.

## Failure modes
Missing model/source hashes, historical input correlation, bootstrap conditionality, counter-transfer bias, shared floating topology, positive-volume ties, duplicate gate patterns and retrospective variant selection.

## Frozen gap-driven semantic phase

TASKS continuationbaf3d099 pushed before probes. Existing `Num` has no equality methods, while `_test` checks numeric `!=0`; the saved count panel itself has output differences up to3.28215e-6. First compare wrapped/plain0 and1 equality. Then run only existingH8seed2 noise89copy3 andH4seed1 noise89copy2 (selected from already-recorded counter discrepancies, not new accuracy claims), using locally cached checkpoint hashes. Compare original float, original wrapped and in-memory equality-only repaired wrapper, all with unchanged recipe constants. Repair only `__eq__`/`__ne__`, recording their operation count; preserve unary-negation and other accounting omissions. No full cost-panel rerun or general count transfer.

Four exact analytic constant-first-label controls atH1,T3: (a) Wi0/Wh0/Wo0, (b) Wi1/Wh0/Wo0, (c) Wi1/Wh0/Wo rowsall1, (d) Wi1/Wh0/Wo[1,0,-1]. Each network always predicts firstlabel0 and therefore accuracy1/3 by exchangeability. Run both recorded exact references and original recipe/float reimplementation. Failures scope positive-volume tie ownership, not generic continuous-weight panel accuracy. No shared fixes or extra variants. This phase also records exact source/dependency and checkpoint hashes; no downloaded weights or model-input MC.
