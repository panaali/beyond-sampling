# T384: independent final factual verification

**PASS, 2026-10-19.** The original writer adopted all seventeen findings. I independently re-read the actual revised passages, checked the response matrix against the source, inspected the current figure implementation and compared the remaining disputed details to the pinned primary evidence. The review found four residual prose issues in the first corrected draft; all four were repaired and rechecked before signoff. There is no remaining publication-blocking mismatch within this audit's scope.

The main conclusion is unchanged: the target challenge remains unsolved. The official HP4 result remains F = 2.63, a lenient pass and strict failure; cost-audit sensitivities are identified as sensitivities. This review does not promote or retire any scientific ledger claim.

## Exact revision

- Review findings: **92416147e8618741b715d6e87a3d836c579a4c91**, against owner snapshot a3ab9caa after an independent first read of shared 7a388e22.
- Preserved owner v4 baseline: **97ee97a43d71cae28b81a2fc1ad3a1b81e45945b**, from owner 0039119b. Its reordered sections, figure repairs, styles, data and status are retained.
- Final corrected tutorial and initial response matrix: **[68ba066c6a91fb61bc91cf0f3bf839a5e79d5349](https://github.com/panaali/clean-neurons/commit/68ba066c6a91fb61bc91cf0f3bf839a5e79d5349)**.
- Source SHA256: `e90a6769b38a5385a25e34f14e6bc5d80b46e67c119366bceef38361bc7af258`.
- Generated `index.html` SHA256: `23c1022e2c57523d5dba4e546b9cde6330a6b5983f7e2d51e1903da227a5930d`.
- Generated `artifact.html` SHA256: `cd70557130a661f93077947bbe0cbfaf7b0656c143957dc82a53a98d3197b225`.

The [writer response matrix](../../tutorial/REVIEW_RESPONSE_T384.md) records adoption and rationale. Later metadata-only completion notes do not alter these source hashes. References below use line numbers in the pinned corrected `tutorial/src/page.html`; figure numbers refer to v4's ordering.

## Findings checked against the revision

| Finding | Final location | Independent resolution check |
|---|---|---|
| F01 | Figure 11, line 499; script 933–981 | **Owner repair preserved; residual claim qualified.** The strip now splits each computed gate piece at true-answer ties and all pairwise logit crossings. The walker still uses tolerances, offsets and a cap. Caption distinguishes its numerical line length from a certified exact value or Gaussian error probability. The old six-midpoint finding does not describe the final code. |
| F02 | Line 453 | **Corrected.** At most one positive coordinate has mass 11/1,024; the all-negative subset has mass 1/1,024. The approximately 0.7 pp is assigned to the broad blind set under tested training recipes. The 60% contribution is a second moment, not variance. |
| F03 | Figure 9, line 474 | **Corrected.** Matrix-RMS-scaled noise is attributed to the project, while ARC's small-model example uses absolute noise. The noise-free F3.3 is identified as a different panel from HP4 F2.63. |
| F04 | Lines 264, 327–335 | **Corrected.** E32's approximately 88 cones, 17,500 operations and 420× comparison are noisy-copy averages under historical 72-operation sampling cost, with an RMS target. A's ARC count and measured implementation count are separated; B's uncharged integral sampling is disclosed. Saved-row arithmetic below supports the rounded values. |
| F05 | Score/gain formulas; Figure 5; line 525; Figure 13; glossary | **Owner fixes retained and residuals corrected.** Gain numerator, the sampling-score floor, the thirty-point size list and rarity-cost qualifications agree with their definitions. Both Poisson and rounded-normal simulation branches are disclosed. HP3 is correctly described as T317c followed by the small numerical variant T317d, rather than an identical-estimate cost-only rescore. |
| F06 | Figure 8, line 456; density-axis string | **Corrected.** Actual bins d<0.02 and d≥0.2 are distinguished from the broader 0.05/0.1 comparison. Whole-cell Gaussian mass is assigned using the centre; the axis is not actual probability density within the band. Approximately 125, original 44.1 and independent 37.2 are no longer conflated. Full-length/causal extrapolations are qualified. |
| F07 | Figure 10, line 482; training-curve label | **Corrected.** The endpoint has 25 retrains; the other settings have 100. The curve says “best observed.” The target exceeds 81 of the 100 DEV3 retrains. Approximately 98% is an observed maximum under tested recipes, not an architecture ceiling. |
| F08 | Lines 545–557 | **Corrected.** Independent unchanged K3, the graded rule and its non-discriminating ordinary-copy test are separated. Green dots are rescaled illustrations; their visibility window is attributed to the combined deep-tail studies. Public-truth independence and the distinct E1/E2 pooled counts and costs are disclosed. The failed triple-tie test remains a failure. |
| F09 | Lines 264, 525 | **Corrected.** Official C3302/F2.63 is retained, alongside screen C3352/F2.67 and stacked C3743/F2.98. The panel mean-cost convention and arithmetic FPE unit are explicit. The audit does not replace the official result or imply measured wall-time parity. |
| F10 | Line 258; region/calculator guidance; noise readout | **Corrected.** The target's ±0.00004 pp is one standard error. The 20,000-sample uncertainty is approximately 0.15 pp near 95% and up to 0.35 pp, not a fixed confidence interval. The 74,000-run guide uses 0.1 pp RMS near p=.92; the target-specific requirement is approximately 44,700. |
| F11 | No-coincidence box, line 286; reference 3 | **Corrected against the primary post.** Eric Neyman and 2025-02-14 are correctly credited. The reversible-circuit zero-suffix property, short-advice verification and 99%-of-random-circuits condition are stated; possible false positives and the analogy's limits are explicit. |
| F12 | Theory boxes, lines 288–294 | **Corrected.** Boolean counting, general polytope volume, CAPP and oracle-access guarantees are distinct. The polynomial dependence is on circuit size and 1/ε. Related PIT/circuit results are not presented as a fixed bias-free Gaussian-network hardness theorem, a universal black-box impossibility, or a theorem about any vaguely defined sampling improvement. |
| F13 | Closure bullets, lines 505–506; line 557; glossary | **Corrected.** The fourth-largest value uses one point in the latest free position; the remaining six are pooled. The bottom-seven compression is described consistently. The later 1.04× oracle sampling diagnostic is not promoted to a deductive repair. |
| F14 | Ladder paragraph, line 370 | **Corrected.** Historical 0.27 is separated from the conservative corrected 0.523 convention. The 2.2×10¹¹ operations and 70% cost refer to seed 0. The positive conclusion remains scoped to the original small-model evaluation. |
| F15 | Quantisation discussion, lines 383–396 | **Corrected/qualified.** Measured region growth is not transferred to untested rounded weights. Finite hidden-state counts do not replace label information; input laws, ties, formats and reachable states matter. Solving a modified model is distinguished from solving the original challenge. |
| F16 | Enumeration opening; lines 422, 459 | **Corrected/qualified.** The practical enumeration ruling remains a measured/extrapolated judgment. The page does not infer a universal boundary census or exclude all exact methods/symmetries. T229's transformation class and E27's tested finite-horizon reductions are stated with their limits. |
| F17 | Exact-region discussion, glossary, references and provenance | **Corrected/qualified.** Exact mathematical decomposition is separated from numerical evaluation; one checkpoint's performance is not an architecture impossibility proof. Face/slice terminology is consistent. Generic quadrature is defined broadly, and this project's classification is attributed to its reading of ARC's heuristic. Added links lead to primary/repository evidence; weights provenance uses the new figure numbers 1,3,4,9,11. |

## Final residuals and their evidence

The first corrected draft still called HP3's second submission “cost-only” with “identical estimates.” The pinned [scorer README](https://github.com/panaali/clean-neurons/blob/5db25624f8cf7e394082141b79b3fef4920ae378/m16/approaches/hp1_scorer/README.md#L481) explicitly identifies T317d as HP3's second submission: seeded model-free transverse constants were replaced by deterministic integration, estimates changed by up to 0.0076 pp on HP3, and F changed from 5.08 to 5.07. The final text now says “small numerical variant.” This is a correction within F05, not a new claim about panel validity.

The other final changes clarify that Figure 13's probability window spans the combined deep-tail studies (F08), label the opening sampling uncertainty as one SE (F10), and repair provenance figure references after v4's reorder (F17). These are prose-only changes, individually rechecked in the final diff. No finding was rejected or deferred; “resolved” includes an explicit limitation where the available evidence cannot settle the broader scientific question.

## Verification and compute scope

`python3 review/tutorial_factcheck/verify_saved.py` passes. It reads immutable public Git objects, checks source hashes and performs scalar arithmetic. It checks the HP4 official/sensitivity aggregates, 100 DEV3 accuracies, five training rows including their sample counts, all 30 plotted gain points against their saved log, the 432-parameter/720-placement counts, Bernoulli uncertainty and gain/floor identities. The final extension also aggregates all **1,024 saved E32 rows**:

| E32 quantity | Independently recomputed value |
|---|---:|
| Mean exact counted operations | 17,506.34765625 |
| Mean leaf count | 87.8759765625 |
| Estimator A ratio, historical ARC count | 10.432509962964671 |
| Estimator A ratio, mean measured count | 1.3935086027368042 |
| Estimator B ratio, historical count with uncharged integral sampling | 2.4338418053066544 |
| Sampling-to-0.1-pp-RMS cost / mean exact cost | 421.760567 approximately |

These reproduce saved aggregate arithmetic, not the original experiments. The final script run used approximately 0.0074 process CPU seconds; JSON records the actual timing. There were zero model calls or random draws in this factual review. Earlier failed read/parser and reviewer harness attempts remain documented in the main report.

I independently compared the final tutorial against owner v4: exactly the source, two generated pages and response matrix changed. Data JSON, status.json and build script are unchanged. CSS and formula blocks are identical. The six changed script lines are the five approved display strings and one obsolete comment. The full `walkLine` body and the current wrong-strip splitting body are byte-identical to owner v4. Inspection confirms that the disclosed Poisson/normal branches and the numerical crossing algorithm match the captions.

The root and writer separately report deterministic-build and desktop/mobile browser PASS at 1280/390 pixels: 13 figures and 14 initialized SVGs, no browser errors or horizontal overflow, with live/walk controls and corrected labels checked. Root additionally checked interactive IDs/attributes and generated-source/data fidelity. Those are their integration checks; this reviewer did not rerun browser model simulations. Writer reports the structural invariant checks in its response notes.

## Remaining limits and handoff

This is a bounded evidence audit, not a reproduction of training, Monte Carlo truths, geometry enumeration, numerical roundoff guarantees, sealed-panel scoring or every external theorem. I read the two cited 2025 ARC primary posts; the complexity references were checked for appropriate claim scope rather than independently reproved. Unsettled universal claims are now qualified instead of replaced by new scientific conclusions.

Live research statuses may advance after this snapshot. T374 owns their synchronization; the review neither edits status.json nor certifies future statuses. There was no ARC contact, no new model experiment, and no scientific ledger promotion.

**T384's review, writer revision and independent verification are complete. T374's corrected-successor Artifact republication remains pending.** DRI registration is requested. T374 should preserve the existing private URL, the already-published v4/T382 history, current peer work and subsequent status updates while publishing the reviewed successor.
