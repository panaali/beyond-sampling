# Acting DRI handoff: 20 October 2026

## Authority and operational state
The owner appointed Codex root to drive the project during Claude's credit outage, expected to end Friday afternoon, 23 October (timezone unspecified). [Verbatim directive](../OWNER_QA.md#d-1020-takeover). This supersedes older Claude-only coordination/signoff wording for this interval; independent verification is still required before confirmation.

At takeover the live Codex roster contained only root. The historical agent names and task assignments did not represent active workers. No research jobs were running in this container. Claude session/process status is inaccessible through the available tools; remote branches establish publication, not liveness. Two focused Codex reviewers were started and completed T459/T460; the T459 reviewer also independently checked T461. No background research run is promised after this turn.

This runtime's sub-agents share its filesystem and compute. There is no tool here for starting separate Claude-style compute sessions. Respect the owner's separate-session preference: use sub-agents for source/reasoning reviews and saved arithmetic, and do not silently launch parallel long experiments in this container. Disk was nearly full from older worktrees; a sparse checkout avoided copying another full repository. No old work or uncommitted evidence was deleted.

## What was read
Shared baseline `2bc5ba0e`: AGENTS, DRI_HANDBOOK (especially section 12), STATUS, TASKS, INDEX, OWNER_QA, scoring/ladder protocols and COMPUTE_FRONTIER; latest tutorial source and README on `claude/t374-tutorial` at `1747cadf`. The shared tutorial is older than that branch. Tutorial publication remains paused unless requested; no tutorial edits or republication were made. The [readiness review](../review/dri_takeover_readiness/README.md) pins the current priority branches and their missing artifacts.

## Scientific assessment
The project has a confirmed equal-compute advantage on clean retrains (H-E17), scoped rare-event estimation and targeted repair (H-R1/H-R3), and promising certificate components (H-C1). It does **not** yet have an established strictly deductive, robust solution of the original challenge, nor a verified all-input rare-event certificate. The line method's transverse averaging, noisy-model scope, and certification rounding/completeness remain substantive obstacles. Current figures and verdicts remain in [INDEX](../INDEX.md); no arbitrary percent-complete estimate is meaningful.

## Decisions for the outage
1. **Continue useful bounded work rather than wait for all Claude credits.** T459 reconciled published artifacts; T460 narrowed a negative; T461 independently checked a completed positive and corrected its reporting. These required no new neural runs and no sealed-reserve access.
2. **Next: T462, IS-v4 cost and interval contract.** Resolve the remaining cost/calibration assumptions before paying for fresh replication. Source/gate controls first; the finite observed panel pass is not a general coverage guarantee. No additional model panel until a frozen replication protocol has an independent reviewer and a runtime budget.
3. **Certification: T463 is the next prerequisite, not a larger run.** T455's ordinary floating CROWN pruning and incomplete mass-rounding argument must be resolved or explicitly conditional. Sampling checks do not establish a certificate. Do not approve T416 as fully verified without T420's final audit, or adopt T452 without its required large identity gate and review. No unchanged T437 extension: its saved extrapolation is expensive and the plausible rescue is a different formulation.
4. **Preserve T370's freeze and blindness.** Stage-2 candidates/selection are incomplete in the published snapshot. Do not select on truths, tune Q16, reconstruct unavailable output by assumption, or open unused held-out panels. Resume only with the frozen artifacts and independent truth custody.
5. **Preserve existing Claude implementation ownership.** T434 has no matching published branch; T420/T442/T452/T455 have partial evidence. Recover their source/checkpoints before restarting; do not mistake an unavailable checkpoint for a scientific negative. T464 records this recovery work. Long certification, training/architecture expansion, new mini-LM streams and paid GPU work wait for concrete prerequisites. No new billion-parameter effort is authorized by this takeover.

## Completed evidence and handback
- [T459 readiness](../review/dri_takeover_readiness/README.md): artifact inventory and priorities.
- [T460 halfspace review](../review/halfspace_charge_review/README.md): finite KILL preserved; family not declared dead; floating/sampled qualifications identified.
- [T461 IS-v4 audit](../review/is_v4_saved_audit/README.md), [independent Decimal check](../review/dri_takeover_readiness/T461_FINAL_REVIEW.md): arithmetic and committed freeze pass, two reporting corrections, cost/calibration still open.

When Claude returns, read this file and T462–T464 first, recover unpublished checkpoints, then resume the handbook's queue with the above gates. None of the existing scientific claims was promoted merely because Codex became DRI. All new work is pushed to the shared branch; the old frozen cost/certificate chains remain frozen.
