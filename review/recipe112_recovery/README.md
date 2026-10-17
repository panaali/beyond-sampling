# T113 recovery and independent completion audit

## Question
What does the existing E43 independent review actually establish, which saved calculations are reproducible, and what remains before the DRI can confirm the claimed smaller-case success and scaling trend?

## Method
TASKS claimc12ff831 pushed before implementation, taking the highest eligible released review in the revised DRI priority queue. Original review files remain read-only. Full source and about2.7MB saved outputs were found in `/workspace/work/admin-t110-t4/review/recipe112`; `/workspace/clean-neurons` lacks several helpers and outputs. Preserve originals with hashes, Git provenance and a local snapshot rather than rerunning completed sweeps. First independently recompute every saved fresh-panel mean-square error, original ratio, per-noise-seed ratio, own-truth/reimplementation discrepancies, rescaling changes and cell/region summary from rows. Recompute the alternative mean of per-model sampling variances at each recorded estimator cost, showing the difference from the original mean-cost convention. Reproduce the count-panel adjustment from its125 recorded rows and state clearly that this adjustment is a panel-level transfer, not a native recount of every fresh model. Recover sensitivity/adversarial/MC evidence with explicit limitations and no selection of better candidate parameters.

Alternative1: raw accuracy errors support the review while counter conventions change the conclusion (separate arithmetic/variance and cost data). Alternative2: apparent region savings are chart-counting artifacts (retain both gate regions and half-chamber pieces). Alternative3: saved truth/reimplementation agreement masks shared tie or tolerance defects (source audit and a separately frozen analytic fixture panel after inventory). Alternative4: missing provenance prevents reproduction (hash all sources/artifacts, identify historical source revisions and preserve unknowns).

No model evaluations, new noisy panels, MC or algorithm changes in this first aggregate phase. A missing artifact is not numerical failure; preserve partial state. New direct checks need a concrete frozen gap-driven scope. Independent reviewer checks aggregate/source findings before DRI confirmation. No shared source or original review edits.

## Status
Raw aggregate and frozen semantic phases complete; independent audits requested. Counter repair/full recount and reference tie handling remain unresolved. T162 handoff complete; T154 independently reviewed. This folder owns only recovery/audit work.

## Results
`aggregate.py` independently verifies165 groups and10,720 saved noisy rows with zero summary discrepancies. CoreH4/5/6/8 each has1,280 noise models; worst original MSE ratios are.002578127/.012822386/.042245659/.098985793. The alternative mean of per-model sampling variances gives.002572166/.012748691/.042125109/.098775788. Original transferred count factors would give.003157287/.016526924/.056238802/.139189753, but **those factors are not validated production budgets**, because the instrumented implementation changes semantics (below) and the25-model cost sample does not recount the larger panel. Mean cells/region falls2.33768→1.67997→1.18164→.90186 on this finite panel, not an asymptotic law. Maximum core own-truth disagreement1.31e-9, float reimplementation difference4.44e-15 and rescaling change6.88e-15 are retained.

`semantics.py` confirms that original Num(0)!=0 is True. In frozenH8seed2/noise89copy3, original float estimate.9654673957873317 becomes.9654674587938789 under counting; adding only value equality/inequality restores the original float estimate exactly, with counted operations330161→326667. H4seed1/copy2 remains within2e-16 of its float reimplementation while count137720→137268. No125-model or full fresh-panel corrected recount has yet occurred. Unary negation and other declared counter omissions remain separate.

All four synthetic fixtures always predict firstlabel0 and have exact accuracy1/3. Both reference implementations return1 on the fullyzero model; for nonzerohidden zero/identicalreadout they and both recipes return2/3. The nondegenerate first-dominant control passes. These are positive-volume tie/zero-form domain defects. They show that reference agreement alone does not validate degenerate models; they do not show a bias of that size on the saved continuously trained/noised panel. ExistingT60 is the shared zero-margin repair lane.

Reproduce scripts with `/workspace/clean-neurons/.venv/bin/python`, one thread,2GB and review lock for<=20CPU phases. Original72files, source hashes, two checkpoint hashes and all original/repaired outputs are preserved. Model-dependent code has not been changed in shared sources.

## Compute accounting
Prior runs are research dependencies and never counted free. Inventory saved wall/CPU fields separately, with missing costs explicit. Aggregate CPU.267775381; frozen semantic phase1.737333006CPU including imports and two saved-model reconstructions (no random model-input evaluations). Saved fresh-group wall times total7771.997303seconds, not independently reconstructed CPU. MC dependency includes45 historical10M panels, not rerun. New aggregate checks do not evaluate models. Legacy counts, scalar-source counters and complete implementation cost are distinct.

## Claims
[inferred] Existing raw rows can support an independent arithmetic audit without repeating expensive experiments. [measured] Raw recorded accuracy statistics reproduce, but automatic cost-wrapper parity fails and analytic positive-volume ties fail. No E43/H-U4 confirmation by this author; fixes and independent review are required for stronger claims.

## Alternatives not ruled out
Additional seeds/noise, correct accounting of all model-dependent operations, rigorous floating geometry, other gate-freezing certificates, training provenance and parameter-sensitive adversarial failures.

## Failure modes
Missing model/source hashes, historical input correlation, bootstrap conditionality, counter-transfer bias, shared floating topology, positive-volume ties, duplicate gate patterns and retrospective variant selection.

## Frozen gap-driven semantic phase

TASKS continuationbaf3d099 pushed before probes. Existing `Num` has no equality methods, while `_test` checks numeric `!=0`; the saved count panel itself has output differences up to3.28215e-6. First compare wrapped/plain0 and1 equality. Then run only existingH8seed2 noise89copy3 andH4seed1 noise89copy2 (selected from already-recorded counter discrepancies, not new accuracy claims), using locally cached checkpoint hashes. Compare original float, original wrapped and in-memory equality-only repaired wrapper, all with unchanged recipe constants. Repair only `__eq__`/`__ne__`, recording their operation count; preserve unary-negation and other accounting omissions. No full cost-panel rerun or general count transfer.

Four exact analytic constant-first-label controls atH1,T3: (a) Wi0/Wh0/Wo0, (b) Wi1/Wh0/Wo0, (c) Wi1/Wh0/Wo rowsall1, (d) Wi1/Wh0/Wo[1,0,-1]. Each network always predicts firstlabel0 and therefore accuracy1/3 by exchangeability. Run both recorded exact references and original recipe/float reimplementation. Failures scope positive-volume tie ownership, not generic continuous-weight panel accuracy. No shared fixes or extra variants. This phase also records exact source/dependency and checkpoint hashes; no downloaded weights or model-input MC.

## Frozen original counter-panel repair

TASKS continuation48e8466a pushed after independent Decimal and semantic review. Revisit exactly the original125 count cases: H4,5,6,8,16; five zoo seeds; clean plus four absolute-sigma.05 noise89 copies. Apply only numeric equality/inequality to the in-memory wrapper. Per case verify original author counter against saved charge, measure original scalar float and equality-repaired counted estimates, and retain every discrepancy above1e-12. Preserve original saved wrapper values/counts without rewriting them. Capture additional equality-operation count explicitly. Five per-H chunks<=55CPU primary, no new noise or model-input probes. Download only ten missing public H5/H6 checkpoints to this owned folder, pin URL/hash, and use existing H4/H8/H16 files read-only. No full10,720-row native recount or general cost-transfer conclusion from this125-case repair.
