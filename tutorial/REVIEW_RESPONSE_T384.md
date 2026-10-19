# T384: writer response to the independent factual review

The user asked a fresh Astra agent to critically review and fact-check the tutorial, then send feedback to the original Astra writer for improvement. This follow-up authorizes evidence-backed factual corrections beyond T382's original prose-only scope. It does not authorize changing raw research results or promoting ledger verdicts.

Review: [independent findings](../review/tutorial_factcheck/README.md), [immutable source pins](../review/tutorial_factcheck/sources.json), [saved-data checks](../review/tutorial_factcheck/checks.json). Reviewer claim `5d236123`; frozen findings published in `92416147`. The correction baseline is **tutorial v4, shared `97ee97a4` / owner `0039119b`**, including its figure repairs, section ordering and publication. T382 remains complete. Owner data and T374's status/publication responsibility are retained.

## Responses

All seventeen findings are adopted, including upstream fixes where identified. None is rejected or left for a later factual correction. Detailed evidence is in the numbered review; this table records the resulting tutorial changes.

| Finding | Response and reason |
|---|---|
| F01 — numerical walk | **Adopted; core display defect already fixed by v4.** Retain the owner's true-tie and pairwise-score splitting. Explain floating-point tolerances/offsets and distinguish the numerical length from a certified exact probability. Correct the stale six-midpoint comment; no walker changes. |
| F02 — blind inputs | **Adopted.** Define at most one positive coordinate (11/1,024) and its all-negative subset (1/1,024). Attribute the approximately 0.7 pp to the broader event under the tested training recipe. Identify the 60% quantity as a second moment, not variance. |
| F03 — relative noise | **Adopted.** Label matrix-RMS-scaled noise a project convention, distinguish ARC's absolute small-model example, and retain the actual T337 panel/curve. |
| F04 — E32 accounting | **Adopted.** Identify noisy-copy means and historical 72-operation sampling cost. Separate A's ARC-count and measured-count ratios; disclose B's uncharged integral sampling. Use 0.1 pp RMS, not a confidence bound. |
| F05 — upstream repairs | **Adopted/preserved.** Keep owner repairs to gain numerator, model-size list, score floor and rarity-cost caveat. Correct the remaining HP3 history: its second submission was a small numerical variant, not identical estimates at a different cost. Make the F floor consistent in the glossary, and disclose the rare-event simulator's rounded-normal branch at mean ≥30. |
| F06 — density bins/proxy | **Adopted.** State the actual d<0.02 and d≥0.2 bins and center-assigned whole-cell mass, with the normalized distance-to-plane definition. Distinguish their ≈125 ratio from broad-bin 44.1/37.2 results. Remove the unsupported 93% statement; retain k≤4/T≤4 scope. Axis label now names assigned mass. |
| F07 — training maxima | **Adopted.** State 100 runs except 25 at the final setting; label the curve "best observed" rather than "best of 100". Describe an observed best under tested recipes, not an architectural ceiling. State that the target exceeds 81 of the 100 DEV3 retrains. |
| F08 — rare estimators | **Adopted.** Separate unchanged K3's independent reimplementation from T366's graded rule and its non-discriminating ordinary-copy test. Explain that displayed ratios are rescaled illustrations and their visibility window covers the combined deep-tail studies. Preserve triple-tie failure status; report pooled E1/E2 outcomes with their separate family-level costs. |
| F09 — HP4 sensitivity | **Adopted.** Retain official 3,302/F2.63. Add 3,352/F2.67 and stacked 3,743/F2.98 accounting sensitivities, whose verdicts agree. Define FPE as arithmetic equivalents and C as the panel mean-cost convention. |
| F10 — uncertainty | **Adopted.** The target precision and noise readout identify one SE, approximately 0.15 pp near 95% and up to 0.35 pp. Calculator/region labels use RMS and the panel-accuracy assumption; text distinguishes the target's approximately 44,700-run requirement. |
| F11 — no-coincidence | **Adopted.** Correct author/date to Eric Neyman, 2025-02-14, and link the primary post. State the restricted reversible-circuit conjecture and its weaker-than-sound-verification guarantee. Label neural performance as motivation, not a theorem consequence. |
| F12 — complexity scope | **Adopted.** Separate Boolean counting, general polytope volume, CAPP and oracle models. No target-specific Gaussian hardness theorem or universal neural black-box impossibility is claimed. Qualify runtime/precision/access assumptions and distinguish the cited related derandomization results. |
| F13 — closure | **Adopted.** Explain the fourth value's one-node/latest-free-position representation and the remaining six pooled values. Keep the numbered simplifications and glossary consistent. Identify the later oracle sampling diagnostic as a diagnosis, not a deductive repair. |
| F14 — ladder costs | **Adopted.** Label 0.27 historical and give the documented conservative 0.523 restatement; both retain the scoped pass. Restrict 2.2×10¹¹ operations / 70% exact cost to seed 0. Use the directly pinned conservative audit rather than mixing additional accounting conventions into the lesson. |
| F15 — quantisation | **Adopted.** Remove the claim of unchanged measured region growth after weight rounding. Explain required true-label information beyond hidden state, discrete-input law/tie assumptions, format-dependent codes/reachability, and the distinction between a modified model and the original challenge. |
| F16 — exact-route scope | **Adopted.** Retain the practical DRI enumeration ruling and measured extrapolation. Remove a universal boundary census/exclusion; explain the limited symmetry class and remaining possibilities. Restrict the record-symbol negative to tested reductions and the 10-step diagnostic. |
| F17 — exactness/teaching | **Adopted.** Distinguish exact geometric formulas from numerical evaluation, trained-checkpoint accuracy from architecture limits, faces from transverse slices, and general quadrature from this project's heuristic classification. Add accessible repository and primary/evidence links, and update the weights provenance to the reordered figure numbers. |

## Implementation boundary and checks

Only tutorial source prose, generated HTML and these response notes are writer deliverables. Scientific JSON, official scores, ledger verdicts, status chips, CSS, structural blocks, controls and element IDs are unchanged. Additional reference anchors and inline notation support the factual explanations.

The root explicitly approved narrow factual display changes: five JavaScript strings (region RMS guide, calculator RMS assumption, density-axis meaning, noise uncertainty, and observed-training-maximum label) and one obsolete explanatory comment. **Executable model, sampling, plotting and walk logic are unchanged from v4**, including all of the owner's recent figure repairs. The raw numerical arrays remain byte-identical. No new research experiment or estimator was run.

The writer's scratch invariant check compares against `97ee97a4`: CSS, JSON and formulas byte-identical; all status chips, IDs, lane placeholders and structural/interactive attributes retained; existing links retained and new links restricted to References; the full script becomes byte-identical after the five declared string replacements and one comment replacement. Numerical prose changes are the evidence-backed corrections in the table, not a claim of an empty numeric diff.

`python3 tutorial/tools/build.py` regenerates both standalone pages. Final independent factual review and desktop/mobile browser results are recorded in the completion handoff and the review's final-verification section. T374 alone republishes the corrected successor at the existing private Artifact URL; the prior v4 publication and T382 closure remain historical facts.

## Completed verification

The independent reviewer signed off **PASS on correction commit `68ba066c`**, including all seventeen findings and four final residual fixes. Final source SHA-256: `e90a6769b38a5385a25e34f14e6bc5d80b46e67c119366bceef38361bc7af258`. The response matrix was checked against the actual text.

The writer's invariant check passes against v4: 545 structural blocks, 52 unchanged status chips, 19 retained existing links and seven added reference links, with the declared script substitutions only. A repeat build leaves no diff and `git diff --check` passes. Final Chromium checks at 1280×900 and 390×844 show 13 figures, 14 initialized SVGs, 17 section headings and 18 status boxes, with no JavaScript errors or horizontal overflow. Live input, sampling, slice selection, compute calculator, gate-walk, rare-event and dark-mode controls pass. Desktop/mobile screenshots were inspected. The manager also independently checked source/code and browser behaviour.

These are software/rendering checks and saved-evidence review, not new scientific experiments. The corrected successor awaits T374 republication at the existing private Artifact URL; the reviewer owns the final T384 board/DRI closure.
