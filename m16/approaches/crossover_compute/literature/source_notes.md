# Verified sources and exposure log

Read date: 2026-10-16 (America/Los_Angeles; execution host date 2026-10-17). Author: Codex literature_protocol under T19. Sources were retrieved using the connected web-search tool, not inferred from search titles alone, except exposures explicitly marked snippet-only.

## Primary sources used

| Source / version | Publication date | Exact relevant locations | Evidence versus application |
|---|---|---|---|
| [Hilton, AlgZoo challenge](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/) | 2026-01-26 | Mechanistic estimates; Conclusion challenge; footnotes 3, 8, 9 | Protocol evidence. Applications in README are our interpretation. |
| [Neyman et al., Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/) | 2025-11-18 | What makes an algorithm mechanistic?; MSP first formulation | Explicitly informal definition; representative-point counterexample; deduction-projection examples. |
| [Koyama, Gaussian simplex probability](https://arxiv.org/abs/1512.06564v2), [PDF](https://arxiv.org/pdf/1512.06564v2) | v1 2015-12-21; v2 2020-03-03 | Section 4, especially equation (18); sections 2 and 5 | Face derivative and holonomic integration machinery verified. RNN boundary cancellation/transport proposal is ours. |
| [Cunningham, Hennig, Lacoste-Julien, Gaussian Probabilities and Expectation Propagation](https://arxiv.org/abs/1111.6832v2), [PDF](https://arxiv.org/pdf/1111.6832v2) | v1 2011-11-29; v2 2013-11-28 | Equations (9)–(14), (57); sections 3.2, 7; appendix D | EP update and negative polyhedron examples verified. Earlier lit/ wording should not be read as guaranteed accurate cone mass. |
| [Gorodetsky, Karaman, Marzouk, continuous tensor train](https://arxiv.org/abs/1510.09088v3), [PDF](https://arxiv.org/pdf/1510.09088v3) | v1 2015-10-30; v3 2018-12-11; journal volume 347 (2019) | Equation (1), sections 3–5, algorithms 1–3 | Adaptive fibers/rank and integration verified. RNN face-integrand use is ours. Correct coauthors are Karaman/Marzouk, not Jakeman (a different related paper). |

Additional primary material opened but not used to select a method: [Koyama and Takemura, orthant HGM](https://arxiv.org/abs/1211.6822), 2012 preprint, full PDF opened; [Cunningham's EPMGP repository](https://github.com/cunni/epmgp), search result only. No code was downloaded or executed.

## Later-source exposure: rows for the lead to append to lit/web_log.md

The assigned lane prohibits editing lit/web_log.md directly. These exposures are disclosed even where search returned them incidentally; they do not supply the proposed methods.

| Date read | Agent | URL | Published | Relevant content | Used by |
|---|---|---|---|---|---|
| 2026-10-16 | Codex literature_protocol | https://www.alignment.org/ | Rolling page, search snippet on read date | Homepage snippet showed June 2026 white-box contest and recent random-MLP work titles. Snippet only. | None; exposure only |
| 2026-10-16 | Codex literature_protocol | https://www.alignment.org/blog/a-mikes-eye-view-of-arcs-research/ | 2026-06-09 | Search snippet described mechanistic models and rare-behavior estimation. Full page not opened. | None; exposure only |

Search also incidentally returned a later non-ARC [Shortform page](https://www.greaterwrong.com/posts/YiRsCfkJ2ERGpRpen/) whose snippet asserted progress mapping AlgZoo models. Publication date of that specific claim was not verified; full page not opened. This is an unverified claim, not evidence of a challenge solution, and was not used. A separate search result linked [a July/August-period comment](https://www.greaterwrong.com/posts/YiRsCfkJ2ERGpRpen/leogao-s-shortform/comment/guGHxetQQCEnef7k9) discussing MSP; exact author/date were not checked, so no conclusion depends on it. No unpublished ARC result was sought.

## Follow-up registration proposals for DRI

Do not allocate T numbers here; the board is shared and may have advanced.

1. Ready/light: deterministic FT rank/rare-peak ceiling test on DEV preintegrated face integrands. Done when ranks 2/4/8, held-out weighted errors and exact construction costs are reported; stop at 4096 face evaluations total. Start only after claiming a board row.
2. Ready/light: decision-facet cancellation audit on existing small cone complexes. Done when flux finite-difference agreement and retained patch counts support or refute useful compression. No global M_16,10 enumeration.
3. Extend T28 notes (do not duplicate T1): rank/gate-respecting joint Gaussian probability reference plus EP representation-invariance test. Done when union/chamber versus frozen-gate effects are separated on predeclared DEV surrogate cases.

None has been started by this agent. The caps are suggested experiment designs; DRI may change them before preregistration.

## T19 follow-up exposures and primary formulas

The follow-up read the current local `lit/web_methods` and `lit/web_arc` reports first. This is secondary exposure to their logged sources; only the following additional primary pages/PDFs were opened directly. Read date remains 2026-10-16 client time (2026-10-17 host). Append these rows to the shared log via the lead/DRI.

| Date read | Agent | URL | Published | Relevant content | Used by |
|---|---|---|---|---|---|
| 2026-10-16 | Codex literature_protocol | https://arxiv.org/pdf/2605.05179v2 | 2026-05-14 v2 | Full PDF, especially S.4.3 and Table 1: factorization preserves estimator; depth-dependent arithmetic cost. | T19 supported_projection.md cost caution; no imported implementation |
| 2026-10-16 | Codex literature_protocol | https://www.alignment.org/content/files/2026/05/deductive_estimation_for_random_halfspace_intersections.pdf | Dated 2025-05-02, posted May 2026 | Sections 3–5, projection recurrence and average-case sketch guarantee. | T19 supported_projection.md, extension for DRI-held T38 |
| 2026-10-16 | Codex literature_protocol | https://www.alignment.org/blog/mechanistic-estimation-for-expectations-of-random-products/ | 2026-05-15 | Opened accompanying announcement. | Context only, formula taken from PDF |
| 2026-10-16 | Codex literature_protocol | https://discourse.aicrowd.com/t/phase-1-write-ups-team-puffi/18175 | 2026-08-16 | Public third-party challenge report; separate analytical and QMC entries, not AlgZoo. | Source discovery only |
| 2026-10-16 | Codex literature_protocol | https://discourse.aicrowd.com/uploads/short-url/yQsl71y2a3zDgb4SpYQFnt9EpoE.pdf | Linked 2026-08-16 | Separate-Universe Response Closure, equation (2), fitted response modes; reported loss to sampling at greatest depth. | T19 supported_projection.md: distinct closure considered and not selected |

Pre-2026 primary reads: [Wu thesis](https://gabrieldwu.com/assets/thesis.pdf), March 2025 title page / May 16 Harvard deposit, section 6.4 and algorithm 4; [Formalizing the presumption of independence](https://arxiv.org/pdf/2211.06738), appendix D. These overlap existing background and were used to avoid repeating E13. Harvard PDF endpoint returned 429; the author's public copy was successfully read.

No source was found establishing a public M_16,10 solution, but this bounded search cannot establish its absence. No private solution was sought. The new supported-projection formula is our mathematical application of the published DPE framework, not a claimed ARC algorithm or a result verified on trained models.
