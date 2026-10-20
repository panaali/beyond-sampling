# Literature review to experiments: 20 October 2026 (T465)

## Source and interpretation
Read the complete [novelty review v2](../lit/novelty_review.md), last changed at `a3404f51d50ee697f04139e35dc26b6a0b31b6b7`, including its seven ranked methods, references, and novelty qualifications. This is the latest review in the fetched shared branch. No new external primary sources were opened in this synthesis; prior ARC exposure is recorded in the existing web log. This is a synthesis of the review, not an independent exhaustive literature search.

The review says most components are known: line sampling/preintegration, ExactLine, elliptical slice moves, dominating-point importance sampling, probabilistic branch-and-bound, and reliability sensitivity/repair. The strongest potential novelty is the outward-rounded Gaussian mass calculation for general ten-dimensional polytope cells; the sealed equal-compute result and rare-event applications may be new results or combinations. These novelty assessments remain qualified, not priority proofs. A component's rigorous implementation does not make an entire pipeline rigorous automatically.

## All seven recommendations reconciled

| Literature recommendation | Current project evidence | Acting DRI action |
|---|---|---|
| Dominating-point importance sampling (Bai et al.) | Implemented through T436/T453/T454/T456. T461 checks the saved score but leaves full cost and general interval coverage open. | **Active T462:** independent analytic-law reliability controls and cost contract; preserve sampling-class label. |
| Halfspace charging (PROVEN inside branch-and-bound) | T437's tested variants miss the target; T460 retains the scoped negative and identifies untested stronger-bound variants and floating qualifications. | Do not extend the same failed search. Reconsider alpha-CROWN halfspaces on a saved mass-dominant frontier after soundness/source recovery. |
| GCP-CROWN cutting planes | T438 is explicitly blocked on T455; the later T418 result implicates splitting and mass charging, not only relaxation strength. | Keep blocked. No licence purchase or premature duplicated verifier integration. |
| Multilevel splitting with exact ellipse moves (LIN-ESS) | T439 is proposed; no completed target result in the current registry. It can address separated modes without finding every dominating point first. | **Next distinct method**, after a CPU slot frees: first freeze a finite Gaussian/polytope control and exact-circle transition protocol. Target performance and correlated-chain uncertainty must be tested separately. |
| Hybrid labelled certificate | T440 proposed. Needs trustworthy early-triple certificates and calibrated statistical bounds on the complement, with complete accounting of all triples. | Preserve as a deliverable goal; T420/T463 and interval work are prerequisites. Never merge deterministic and confidence guarantees into one unlabelled claim. |
| Time-split forward/backward bounds | T441 proposed, no published completed kill test found in the current registry. | Second new-method option after T439. Specify the intermediate-state law and preimage enclosure before numerical exploration. |
| QMC line sets matched to preintegration | T442 already owns this; partial outputs are published but final analysis is absent. | Recover/complete that lane rather than launch duplicate line-set search. Remains quadrature-class regardless of empirical gain. |

The final sentence of the literature review predates T449/T451's scoped negative on tested weight-derived between-line bounds. Their negative does not exclude every conceivable bound, but it prevents treating the same tested families as a newly discovered open experiment. No literature result here supplies a strictly deductive solution of the ARC target.

## CPU and experimental policy
The owner's [new instruction](../OWNER_QA.md#d-1020-cpu) permits local CPU experiments. Available tools can spawn shared-container sub-agents, but cannot allocate independent research sessions/machines. Container quota is four cores. Two owned numerical lanes run at most concurrently, each initially one thread and under 2 GB per process; the initial bounded phases allow up to 600 CPU-seconds each. These are review points, not a ban on longer useful work: extend only when the results and prerequisites justify a concrete next phase.

Before execution, publish the input/control families, seeds, thresholds, costs and stopping rule. Keep failures; distinguish analytic/synthetic controls from trained-network evidence. Cross-review the two independent lanes before adopting repairs or promoting claims. Sub-agents may now execute these explicitly authorized local experiments, superseding the older preference that they be used only for no-compute work when separate sessions are available.

T462 addresses the literature's first recommendation and the latest positive result's weakest assumptions. T463 addresses the review's floating-point warning (Jia–Rinard / safe verification) and the main certificate bottleneck. Keeping CPUs busy is useful when the work discriminates between explanations; raw compute cannot replace a missing proof or unavailable checkpoint.

## Completed bounded phases
T462 and T463 executed and cross-reviewed; [INDEX](../INDEX.md#3-experiment-registry) records the findings. T462 supplies a missing-mode reliability counterexample and a conservative fixed-n reporting alternative. T463 exposes a true floating mass underestimate, independently reproduced on another box, and supplies reviewed local interval/rational primitives. The latter is specific to T455's box-mass charge; it does not invalidate H-C1's different polytope-mass implementation. Neither phase ran a trained target or full search, and neither stopped for compute. T439 is the next Ready distinct-method pilot; T466/T467 track integration and trained validation rather than declaring the problem solved.
