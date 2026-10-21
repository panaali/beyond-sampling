# T475 deadline stronger public baseline reproduction

## Question
Can an immutable public stronger estimator improve our first submission before the owner deadline?

## Method
Run unchanged pinned e30_a, logxio and V29 in isolated supervised workers, sequentially and single-threaded. Sources and hashes in sources.json. Compare predictions to the same official archived public means used by T474. First two mini rows are development; full rows are held out only from our development, not from upstream authors.

Two alternatives: published improvements may come from a different scorer version (test actual flopscope cost and residual); learned public corrections may overfit (compare distinct full rows, retain limitation). Stop any worker above 7.5 GB RSS, 120 CPU s or150 wall s. Root reports owner authorized 8 GB/process and16 GBtotal for this deadline sprint. Maximum8 CPU minutes total.

## Status
Original e30 and logxio reproduced. Original V29 deferred because both stronger descendants passed and deadline prioritizes submission verification. L6 local pilot passes but upstream server failures disqualify recommendation. Dtype-only e30 is bitwise identical on mini0, with identical FLOPs and a passing residual timer. Root coordinates submission.

## Results
Artifacts in results; root registers final numbers in INDEX.

## Compute accounting
Actual flopscope operations; setup separately metered. Supervisor wall/RSS/CPU counters retained. Four reference networks maximum per promising source.

## Claims
[inferred] Published methods are likely stronger than V25; local reproduction required. No novelty, private rank or broad generalization claim.

## Alternatives not ruled out
Public training overlap, floating-point platform effects, grader-version changes.

## Failure modes
Residual time .4s, full shape1024x16 only, memory > limit, numerical stability, learned corrections.

## Bounded follow-up preregistration
After original-source checks, test upstream X_L6=1 (sixth Strassen level) on mini0 then the existing full0/full1 only if finite and residual <.35s and score improves. Exact arithmetic is unchanged; finiteprecision and operation-count/residual overhead are alternatives. No broad search. Variants agent owns rank/threshold sweeps.

Root reviewer identified remote dtype metadata as strings; e30_l6_rpc uses str(w.dtype)=="float32" fortheexistingfloat32fastpath. Other pool dtypechecks compare stored dtype constants and remain unchanged. Test exactnativeparity against e30_l6 onmini0 before evaluating remainingrows.

Root independent reviewer found upstream WRITEUP E1 reports L6 ~20%server failures and119.3/120sworstwall, despite~1.5%FLOPs saving. FreezeL6 as not recommended; do not expand fullrowtests. e30_rpc keeps original5levels and only fixes dtype metadata equality; testmini0 bitwise equality.

## Reproduction
From repo root, using the T474 environment:
```sh
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_stronger/run.py --method e30 --width 1024 --seed 6319981554997072999 --output /tmp/e30_mini0.json
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_stronger/run.py --method logxio --width 1024 --seed 1778484118851160678 --output /tmp/logxio_full0.json
/tmp/whest-env-t473/bin/python m16/approaches/whitebox_stronger/score.py
```
Seed 7145726208641497362 is full1. Sources are immutable originals except explicitly named l6/rpc variants. The worker verifies SHA256 before import. Use a new output path; supervisor refuses overwrites. `results/scores.json` records both strict and lenient point bars relative to covariance, with no population confidence claim.

## Exhaustion checklist
Tried: original e30 and logxio, four full-sized public networks for e30 and three for logxio; upstream optionalL6 onmini0; dtype metadata checkfix. Not tried: V29 (weaker public benchmark), fullK4 (outsidecompute budget), parametersearch (separate T476agent), L6fullpanel (knownpublishedserverwallfailure), freshsealedreference (outsidedeadline scope). No estimatorfamily declareddead.

## Independent T476 review
Reviewed rank448 source against pinned e30: only provenance header and designated R_OLD literal change. Reviewed block256 diff: default environment knobs become their documented literal defaults, and both free-output-row loops plus their scratch allocations change128to256. Summed indices and estimator ranks/coefficients remain unchanged. Independently loaded all four saved complete prediction matrices and verified bitwise equality and matching model hashes against T474. See results/t476_independent_review.json. This is independent source/saved-output review, not an independent estimator rerun.

## Deadline robustness extension, frozen before fetch
Root authorized full public rows2/3, not used by our current selection, before23:57UTC. Fetchonlyprojectedreferencecolumns,max5MB,stopifnetworkstalls30s. Evaluate original e30 onboth; T476worker independently evaluates existing rank448/tau3 iftimepermits. This remains public-data reproduction with possible upstreamtrainingoverlap; nosealedgeneralizationclaim. Selectionfrozen original e30 pluslistedvariants, noadaptationtofreshresults. Stopnewcompute23:57UTC, maximum180CPU-secondsadditional.

Fresh original-e30 checks completed before23:56UTC, with outputs/provenance in results/fresh_references and scores in results/fresh_scores.json. Reproduce fetch with `python references_fresh.py --split full --output results/fresh_references`; workers use seeds in those metadatafiles. No covariance rerun onthisextension, so strict/lenientcomparisonbars are retained ontheearlierfourrowpanel; directfreshvariant-to-e30comparisons belongtoT476.
