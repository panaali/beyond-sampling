# Approaches to ARC's challenge on M_16,10

↑ [Project README](../../README.md) · [STATUS](../../STATUS.md) · [INDEX (registry + ledger)](../../INDEX.md) ·
[FINDINGS (mechanism)](../FINDINGS.md) · [Leaderboard code](../leaderboard.py)

Each folder is one attempt, run by one agent, with a `README.md` in the template from [AGENTS.md](../../AGENTS.md).
The **ID** is the experiment's row in the [INDEX registry](../../INDEX.md#3-experiment-registry). "Deductive" means no
random inputs are fed to the model. MSE ratios are relative to plain sampling at equal compute (below 1 is better).
Only the leaderboard (LB2) numbers were run by the DRI; the rest are the agents' own.

**Updated 2026-10-18 (afternoon).** The first three tables are the original index. The tables after them add every
folder created since: on the shared branch `ccr-6a61f25c-wn80ii`, and on the child branches listed below. Status comes
from the folder's INDEX row where it has one. Rows marked "(README)" have no INDEX row; their status is taken from the
folder's own README or HANDOFF.md. Nothing here upgrades a claim, and most rows are unverified. "F" is the own-cost
figure of merit of TASKS Directive 8 (amendment 2): F = MSE x max(C, 1k FPE) / mean p(1-p), with strict F <= 1 and
lenient F <= 10. A folder that exists only on a child branch does not count as evidence on the shared branch until it is
pushed there (TASKS DRI directive 1).

| Key | Child branch (folder only there, not on the shared branch) |
|---|---|
| b192 | `claude/t192-gpu-cost-law` |
| b194 | `claude/t194-width16-probe` |
| b218 | `claude/t218-tie-tail-labels` |
| b241 | `claude/t241-neuron-account` |
| b286 | `claude/t286-cheap-frontier` |
| b292 | `claude/t292-oracle-floor` |
| hp1 | `claude/hp1-scorer` |

## Deductive attempts (ARC's actual bar): all lose to sampling

| Folder | ID | Idea | Result |
|---|---|---|---|
| [rank_exact](rank_exact/README.md) | E10 | Freeze gates per rank-order class; integrate exactly | 8-13x worse; gates not fixed by rank order (H-M8) |
| [propagation](propagation/README.md) | E13 | Propagate Gaussian mixtures / cumulants through the RNN | Off by 10-69 pp; cumulant ceiling (H-E6) |
| [cone_enum](cone_enum/README.md) | E14 | Enumerate error cones; model-free cone masses | 56-91x worse; >=444k error cones (H-M7) |
| [reliability](reliability/README.md) | E17 | FORM/SORM per rank chamber | 20-50x worse; lowest deductive bias (+0.15 pp) |
| [branch_bound](branch_bound/README.md) | E19 | Certified probabilistic branch-and-bound | Certified nothing at 1e11 FLOPs |
| [observer](observer/README.md) | E21 | Signal-detection "observer" quadrature | ~10x worse |
| [sci_laplace](sci_laplace/README.md) | E26 | Laplace: asymptotics of tie bands | Oracle additive model misses by 4% (H-E8); fallback fails on seeds |
| [sci_poincare](sci_poincare/README.md) | E27 | Poincare: symbolic dynamics over records | Ceiling +0.07-0.75 pp, always positive (H-E10) |
| [sci_feynman](sci_feynman/README.md) | E24 | Feynman: exact partition + segment walks | Deterministic quadrature ~1-3x; not deductive in ARC's sense |
| [stoch_margin](stoch_margin/README.md) | E31 | T1: low inputs as a distribution given a top-3 skeleton | Best reduction -0.13 to -0.45 pp; fidelity needs top-5, which leaves <=1.2x to gain |
| [sci_tao](sci_tao/README.md) | E25 | Tao: exact structure + averaged pseudorandom part | 507x at T=3, collapses at T~5 (H-E11); ~1.5x at T=10 |

## Sampling-based and quadrature methods (baselines, not solutions)

| Folder | ID | Idea | Result (LB2 at 64k: orig / seeds / sigma 0.0025 / sigma 0.005) |
|---|---|---|---|
| [control_variate](control_variate/README.md) | E12 | **srb**: exact last step + near-tie strata | **0.42 / 0.45 / 0.50 / 0.61: best, CI < 1 everywhere** |
| [tie_lines](tie_lines/README.md) | E18 | "joint": gap strata + tie-swap antithetics | 0.34 / 0.72 / 0.51 / 0.53 |
| [boundary](boundary/README.md) | E11 | Boundary zones + exact 1-D band integrals | 0.61 / 0.68 / 0.52 / 0.68 |
| [tube](tube/README.md) | E15 | Deterministic tie-slicing quadrature | 0.14 / **1.23** / 0.46 / 0.75 (fails on seeds; bias floor) |
| [debiased_series](debiased_series/README.md) | E20 | Exact E[surrogate] + sampled residual | Loses to stratified below 128k |
| [great_circle](great_circle/README.md) | E22 | Exact integration around random great circles | 3-10x worse (H-E7) |
| [array_rqmc](array_rqmc/README.md) | E16 | Array-RQMC with an exact last step | ~1.0-1.6x |

[UPDATED 2026-10-18: LB3 supersedes these LB2 ratios. With cost charged as implemented and tuned rows excluded, srb
does not beat sampling on the untuned groups (INDEX section 4b, LB3). Its remaining edge rests on a hand-counted cost
factor (DRI_HANDBOOK section 10).]

## Diagnostics

| Folder | ID | Question | Answer |
|---|---|---|---|
| [bandwidth](bandwidth/README.md) | E23 | Is the tie band width compactly describable? | No: learnable but not compact; uncorrelated across seeds |

Earlier DRI-run estimators live directly in [`m16/`](../README.md): probe v2-v4 (`estimate.py`), exact lines
(`lineint.py`), RQMC (`qmc_test.py`), stratified sampling (`stratified.py`).

## Added 2026-10-18: small-model ladder, recipe and exact enumeration (ARC protocol, M_4,3 up to (8,5))

| Folder | ID | Idea | Result |
|---|---|---|---|
| [understand_m43](understand_m43/README.md) | E36, E37, E40, E41 (T63) | Accuracy estimate for M_4,3 derived from its mechanism (RECIPE112, T75) | Done. Beats sampling on ARC's protocol (MSE ratio .019) and on fresh noise of seeds 0-4 (.0002-.25); H-U1/H-U2 confirmed (scoped). T=4 fails (E41) |
| [understand_ladder](understand_ladder/README.md) | E38, E39 (T73/T88/T93); R4 | Fibre-traced error wedges, H up at T=3 | Done. Passes every T=3 rung to (8,3); ops ~H^2.0 vs regions ~H^1.84, so H-U3 is refuted at T=3 (R4) |
| [ladder_training](ladder_training/README.md) | T86 | Train the missing ladder sizes (H=7, T=7) with the zoo's own code | Paused 2026-10-17 by the owner's stop; TASKS row Ready (README) |
| [ladder_training_axis](ladder_training_axis/README.md) | E44 (T101) | Training-amount axis of M_4,3, n_seqs 2^28 down to 2^20 | Done. Passes on every rung (unverified; review T114) |
| [training_axis_ladder](training_axis_ladder/README.md) | T90 | The 2^29 neighbour of the passing M_4,3 recipe | Bounded phase complete; the DRI classified the rung as passed |
| [training_axis_review](training_axis_review/README.md) | T114 | Independent review of E44 | Review complete, DRI sign-off pending. Absolute MSE ratio .765 [.58, .95] |
| [margin_training_recovery](margin_training_recovery/README.md) | T115 | Recovery of the saved margin training-axis runs (k28, k30) | Raw recovery complete and independently reviewed; largest saved p20 ratio .103 (README) |
| [final3](final3/README.md) | E45 | RECIPE112 + multi-step certificate along T at H=8, and the M_16,10 projection | Done. The (8,4) part is superseded by E46 (leaf-mass bug) |
| [final3_review](final3_review/README.md) | T133 | Independent audit of E45's accounting and scope | Done as a bounded review (README) |
| [estT_fix](estT_fix/README.md) | E46, E48 (T131, T164, T185) | Leaf-mass bug fix, (8,4) re-score, (8,5) with 5-D masses | Done. (8,4) passes on 5 seeds + noisy copies (unverified); (8,5) seed 0 accurate at 2.2e11 ops; seed 2 later fails (T192 S2) |
| [e46_e48_review](e46_e48_review/README.md) | T193 | Independent E46-E48 evidence and cost gate; seed-2 exact reference | Saved 9-case audit passes independent review; TASKS row blocked on saved T5 rays (README) |
| [t185_evidence_review](t185_evidence_review/README.md) | T185 review | Independent review of the saved T185 (8,5) evidence | Saved-data phase complete and independently reviewed (README) |
| [frozen_recipe_holdout](frozen_recipe_holdout/README.md) | T148 | Frozen recipe108 on held-out H8,T4 seed 1 | Bounded held-out test complete; aggregate review passes (README) |
| [recipe_cost_review](recipe_cost_review/README.md) | T153 | Cost of the frozen recipe108 calculation | Trace complete (28,000 rays, 6,182 cells reproduce); independent review pending (README) |
| [recipe_mass_separation](recipe_mass_separation/README.md) | T145 | Recipe approximation error vs numerical mass error | Decomposition complete; independent review passes (README) |
| [certificate_review](certificate_review/README.md) | T108 | Independent review of the gate-freezing certificate | Finds a strict-interior failure of the individual bounds; a safe cumulative primitive passes 768 exact checks; adoption (T170) deprioritized (README) |
| [exact_ladder](exact_ladder/README.md) | T46 | Exact cone enumeration + exact cone measure at H=16, T=2..6 | Exact on all 29 leaderboard models at T=2-4 and on seeds at T=5; T=6 extrapolated. TASKS row paused by the owner's stop (README) |
| [small_h4_4](small_h4_4/README.md) | T51, T64 | H4T4 accuracy by bounded cone integration | T64 done: values match within 2.8e-16, integration 1.6-2.1x faster; DRI acceptance pending |
| [small_ladder_review](small_ladder_review/README.md) | T52, T72 | Independent review and compute audit of the small-model ladder | Base phases done (H2T4 and H4T4 enclosures; H2T6 p .2481524); DRI acceptance pending |
| [bulk44](bulk44/README.md) | E47 (T165 bulk44) | (4,4) bulk errors, the Hopf great-circle estimator, the T=4 H ladder, T111 | Done. Beats sampling at (4,4), (5,4), (6,4), but quadrature class (fixed node set); margin shrinks with H |
| [bulk_hopf_review](bulk_hopf_review/README.md) | T178 | Independent review of the bulk44 Hopf evidence | Bounded H4T4 phase complete and reviewed; shared repair and cost closure are T182/T183 (README) |
| [hopf_native_cost](hopf_native_cost/README.md) | T183 | Native numerical accounting of the Hopf estimator | Analytic, h8 and h64 phases complete with independent audits (README) |
| [hopf_reference_precision](hopf_reference_precision/README.md) | T183 continuation | Unchanged Hopf h64 on 25 fixed H4T4 models with source costs and tighter truths | All 60 group and 50 individual upper ratios below 1 under the declared budgets; quadrature class (README) |
| [bulk_error_mechanism](bulk_error_mechanism/README.md) | T116 (bulk) | Bulk-error mechanism attribution | Constant-output gate regions explain ~30% of H4T4 error; simple certificates cover .006 of the mass; rescues open |
| [bulk_joint_feasibility](bulk_joint_feasibility/README.md) | T120 | Exact simultaneous bulk-feasibility discriminator | Protocol frozen (README) |
| [bulk_rank_mass](bulk_rank_mass/README.md) | T122 | Mass of exact rank-conditioned bulk certificates | Selected-panel phase complete; independent review pending (README) |
| [rank_chart_ambiguity](rank_chart_ambiguity/README.md) | T125 | Whole rank-chart bulk ambiguity | Exact granularity obstruction; 47 neighbourhoods verified independently (README) |
| [class_region_families](class_region_families/README.md) | T110 | Ordered top-two class families as simplicial charts | Initial algebra and T3 phase reviewed; the approximate quadratic recipe worsens 4/5 cases |
| [class_region_t4](class_region_t4/README.md) | T110 (T4) | Trained T4 class-region partition | Six frozen cases: families 96 -> 24/72, exact terminal cells 2,595 -> 1,191/2,338; review pending |
| [fading_influence](fading_influence/README.md) | T187 | Fading-memory influence diagnostic | Weight bounds grow along T; the oldest input is more sensitive than the newest in 97/384 (H4) and 175/384 (H8) witnesses; no safe merge (README) |
| [future_equivalence](future_equivalence/README.md) | T188 | Same-future margin signatures (state-machine merging) | Controls pass; the trained H4 pilot finds one exact reset and no other future-equivalent states (README) |
| [conditional_order_bounds](conditional_order_bounds/README.md) | T189 | Weight-derived uncertainty bounds for omitted inputs | Scalar and domain controls pass, audit passes; T189 closed for now (README) |
| [weight_circles](weight_circles/README.md) | T191 | Weight-derived circles (a mechanistic Hopf grid) | Bounded analytic phase complete and reviewed; parked by DRI directive (README) |

## Added 2026-10-18: two-neuron (H2) and small exact mechanisms (Codex lanes, 2026-10-17)

| Folder | ID | Idea | Result |
|---|---|---|---|
| [feynman_fundamentals](feynman_fundamentals/README.md) | T55 | First-principles lane and standing independent reviewer for the Codex folders | Standing lane; reviews most folders in this and the next table (README) |
| [small_h2_34](small_h2_34/README.md) | T47, T51 | Two-neuron T3 and T4 recovery cycles | Done: H2T4 .3077307430, H4T4 .4770131260; independent geometry agrees within 7.75e-13; DRI acceptance pending |
| [small_h2_56](small_h2_56/README.md) | T48, T62, T71 | H2T5 and H2T6 deductive accuracy and bounded quadrature | T5/T6 phases complete (T6 p .248150); counted ops cut ~20-24x (T71); no equal-cost win yet; DRI acceptance pending |
| [small_h2_78](small_h2_78/README.md) | T49, T53/T54, T61/T65 | H2T7 and H2T8 reset decomposition | Open: no certified full estimator; T61 negative; the approximation stays biased |
| [small_h2_910](small_h2_910/README.md) | T50, T59 | Trained H2 T9/T10 applications | T59 cache port done; accuracy work at cycle 1 |
| [global_reset_words](global_reset_words/README.md) | T50 (reset) | Exact global reset words for mixed-sign H2 | Bounded phase complete; a scoped reset obstruction is recorded |
| [reset_wedge](reset_wedge/README.md) | T49 (reset) | Magnitude-dependent H2 reset wedge | Three strata contribute .1726 of H2T8 accuracy; .1429 of event mass unresolved |
| [reset_root_factor](reset_root_factor/README.md) | T66 (reset) | Exact reset-root factorisation | Bounded phase complete; repairs the H2 gate/readout count discrepancies |
| [three_suffix_reset](three_suffix_reset/README.md) | T49 (three) | Three-input suffix after a certified reset | Adds .0151 to H2T8 partial accuracy; cost and coverage insufficient |
| [prefix_wedge_remainder](prefix_wedge_remainder/README.md) | T55 lane | Bound the chance that dropping an early hidden-state amplitude changes correctness | Two bounded phases complete, independently reviewed (README) |
| [h2_boundary_mechanism](h2_boundary_mechanism/README.md) | T76 | Boundary-family mechanisms for H2T4 and H2T5 | Three-input rescue passes 11/28 own-cost cases; no general winner |
| [history_distribution](history_distribution/README.md) | T76 (history) | Positive conditional hidden-history distributions | Bounded phases complete and reviewed; convergence and cost open |
| [nonzero_history](nonzero_history/README.md) | T76 (rescue) | Nonzero-history rescue | Protocol frozen; controls and pilot only (README) |
| [output_forgetting](output_forgetting/README.md) | T105 | Output forgetting with the full rank law (a partial-event contribution) | Disjoint events cover 26.8% of H2T10 input mass; the rest is unresolved |
| [output_forgetting_complement](output_forgetting_complement/README.md) | T105 | Exact complement ambiguity under four-input output forgetting | Claim and protocol only (README) |
| [stratum_transport](stratum_transport/README.md) | T58, T83, T91 | Stratum-preserving H2 forward transport | Prototypes done (T8 four-input marginal error .00054 -> .00014); no full estimator |
| [gaussian_pair_terminal](gaussian_pair_terminal/README.md) | T117 | Fixed-state Gaussian two-input terminal operator | Primitive agrees within 4.45e-16; a trained probe exposes a 1.64 pp interpolation defect; T118 pilot next |
| [middle_target_pair](middle_target_pair/README.md) | T127 | Two-Gaussian terminal with the target in the middle | Bounded phase complete; source review (README) |
| [real_arc_tail](real_arc_tail/README.md) | T102 | Real even-arc weighted-tail recurrence | Bounded phase complete; GL8 is 43.6% slower than the endpoint formula |
| [directed_chebyshev](directed_chebyshev/README.md) | T78 | Directed positive Chebyshev tails | Primitives complete and reviewed; no geometry or accuracy certificate |
| [endpoint_cumulant_review](endpoint_cumulant_review/README.md) | T40 | Exact-law argmax endpoint review | Bounded phase complete; support repairs remain inaccurate |
| [rank_conditioned_transport](rank_conditioned_transport/README.md) | T67 | Fixed target rank and two-dimensional hidden transport | Bounded rescue negative (fails the 1e-7 gate); representation open |
| [hidden_reference_review](hidden_reference_review/README.md) | T37 (hidden) | Hidden-reference gate-remainder ceiling | First-order and final-step rescues fail on both DEV cases |
| [reference_error](reference_error/README.md) | T37 (P1) | Reference-error independence ceiling | Negative: independent-residual laws miss by 22.6-48.7 pp |
| [third_step_means](third_step_means/README.md) | T69 lane | Exact third-step hidden means without Gaussian closure | All controls pass; independent saved review (README) |
| [third_step_pairs](third_step_pairs/README.md) | T55 lane | Exact third-step pair moments | Controls pass, independent audit; frozen six-network pair phase (README) |
| [structured_prefix_mixture](structured_prefix_mixture/README.md) | T69 | Structured prefix mixture | Single-model and 24-cell phases complete, independently audited (README) |

## Added 2026-10-18: cone masses, certificates and native-cost audits (Codex lanes; frozen by TASKS DRI Directive 3)

These refine costs and numerics on fixed small panels (mostly the 6,347-cone H4T4 corpus and H8T4). Directive 3 lets
each finish its current phase and start no new ones; none makes an accuracy or sampling-win claim.

| Folder | ID | Idea | Result |
|---|---|---|---|
| [cone_mass4](cone_mass4/README.md) | T109 | Cheap exact Gaussian masses of simplicial cones in R^4 (goal < 300 ops, error < 1e-7) | Continued in the T109 folders below; TASKS row Ready (README) |
| [cone_mass4_review](cone_mass4_review/README.md) | T119 | Independent check of T109's trained-cone masses | 72 controls pass; corpus .4770137 vs independent .4770131, one negative cone mass (README) |
| [positive_t4_corpus](positive_t4_corpus/README.md) | T123 | Positive-reference coverage of the fixed T4 geometry | Covered mass low by 5.8e-7 (2.2e-7 after T119's fallback) (README) |
| [acute_t4_reference](acute_t4_reference/README.md) | T126 | Acute-ray positive reference | 27 controls pass; 432 of the 488 uncovered cones are acute (README) |
| [radial_t4_reference](radial_t4_reference/README.md) | T129 | Exact radial bounds for positive cone references | With T123/T126 covers all 6,347 error simplices; accuracy agrees with an independent value within 3.9e-16 (README) |
| [positive_mass_budget](positive_mass_budget/README.md) | T130 | Cost-first positive cone masses | 15 of 54 analytic controls return within bounds; 39 hit the order limit (README) |
| [analytic_inner_mass](analytic_inner_mass/README.md) | T143 | Positive analytic inner mass | Named work down 65% / 74%; 181 controls pass (README) |
| [moment_mass_budget](moment_mass_budget/README.md) | T135 | Analytic moment intervals for positive cone masses | Replaces about half the quadratures, mostly cheap ones; accuracy within 8.8e-6 (README) |
| [moment_h8_cost](moment_h8_cost/README.md) | T168 | Original moment shortcut on H8T4 | 64-case pilot complete; review requested (README) |
| [positive_h8_full](positive_h8_full/README.md) | T109 | Positive analytic-inner full H8T4 phase | Misses the < 300 ops/cone goal: 4,071 known ops/cone (README) |
| [positive_h8_cost](positive_h8_cost/README.md) | T161 | Positive analytic-inner H8T4 cost discriminator | Protocol frozen (README) |
| [positive_h8_holdout](positive_h8_holdout/README.md) | T167 | Frozen positive-mass transfer to H8T4 seed 1 | Full transfer complete; independent audit passes (README) |
| [positive_inner5](positive_inner5/README.md) | T109 lane | Positive analytic inner integral in dimension five | Primitive and fixed-product phases done (README) |
| [positive_inner5_batch](positive_inner5_batch/README.md) | T11589 | Batch primitive parity | 30 controls complete; audit passes (README) |
| [positive_inner5_cost_compare](positive_inner5_cost_compare/README.md) | T11589 | Fixed expression-cost comparison | 66 rows complete; audit passes (README) |
| [positive_model_adapter](positive_model_adapter/README.md) | T171 | Model-facing positive-mass audit | Both checkpoints reproduce every original ray bitwise (README) |
| [stable_moment_full](stable_moment_full/README.md) | T109 | Full frozen stable-moment mass phase | 35,098-cone phase complete; saved-data audit requested (README) |
| [stable_moment_h8](stable_moment_h8/README.md) | T181 | Stable Dirichlet quadratic variance | Controls and 64-cone pilot complete (README) |
| [thin_duffy_bound](thin_duffy_bound/README.md) | T109 | Thin Duffy truncation verification | 17,955-cone range/bound phase complete; no mass integration (README) |
| [gauss_legendre_enclosure](gauss_legendre_enclosure/README.md) | T11588 | Exact enclosures of the T5 Gauss-Legendre constants | 32 nodes/weights and 64 moments verified independently (README) |
| [wedge5_controls](wedge5_controls/README.md) | T11583 | Five-dimensional wedge-product controls | 66 evaluations complete and reviewed (README) |
| [orth5_cost_review](orth5_cost_review/README.md) | T185 lane | Five-dimensional mass cost inventory | Adapter and synthetic phases complete and reviewed (README) |
| [mass_reduction_cost](mass_reduction_cost/README.md) | T11579 | Fixed mass norm and reduction coverage | 172 parity controls pass; TASKS row Ready (README) |
| [cofactor3_guard](cofactor3_guard/README.md) | T109 / T11580 | Guarded 3x3 cofactors (a cheap determinant with an error enclosure) | 25 analytic controls pass; audit passes (README) |
| [cofactor_angle_bounds](cofactor_angle_bounds/README.md) | T109 | Fixed-denominator cofactor angle bounds | 21 controls pass; 64-case continuation complete (README) |
| [joint_cofactor_angle](joint_cofactor_angle/README.md) | T184 | Joint stored-matrix cofactor angle | 12 controls pass independently (README) |
| [log_cofactor_cost](log_cofactor_cost/README.md) | T179 | Log/cofactor mass source-cost audit | Complete; final review passes (README) |
| [log_cofactor_review](log_cofactor_review/README.md) | T179 | Log/cofactor rule audit | Full 35,098-cone phase fails one frozen tolerance case (scoped) (README) |
| [determinant4_interval](determinant4_interval/README.md) | T144 | Counted determinant and outward interval | Fixed-corpus comparison complete and reviewed (README) |
| [determinant_native_cost](determinant_native_cost/README.md) | T158 | Installed determinant cost | Complete and reviewed (README) |
| [inverse4_cost_review](inverse4_cost_review/README.md) | T150 | Fixed 4x4 inverse cost audit | Floating-expression envelope 188 (56 LU + 132 solves); no full cost (README) |
| [inverse_numeric_cost](inverse_numeric_cost/README.md) | T157 | Inverse numerical branch categories | Adds 102 numerical units (290 in all); no complete cost (README) |
| [rank_cost_review](rank_cost_review/README.md) | T156 | Values-only SVD/rank kernel cost | Dispatch/control phase complete (README) |
| [rank_numeric_envelope](rank_numeric_envelope/README.md) | T156 | Native rank numeric envelopes | Selected-path audit complete and reviewed (README) |
| [rank_runtime_census](rank_runtime_census/README.md) | T165 | Frozen rank-kernel runtime census | 16 chunks: 275,897 rank calls counted, no failure observed (TASKS note) |
| [dqds_retry_review](dqds_retry_review/README.md) | T163 | dqds retry review | Source review complete and reviewed (README) |
| [plackett_cost_review](plackett_cost_review/README.md) | T146 | Cost inventory of the repaired T4 calculation | Full trace complete and reviewed; TASKS row blocked (README) |
| [plackett_endpoint_review](plackett_endpoint_review/README.md) | T142 | Locate the Plackett endpoint sensitivity | Diagnostic complete and reviewed (README) |
| [plackett_precision_review](plackett_precision_review/README.md) | T137 | High-precision cause of one negative Gaussian mass | Complete and reviewed (README) |
| [plackett_rule_doubling](plackett_rule_doubling/README.md) | T140 | Fixed uniform Plackett rule doubling | Complete and reviewed (README) |
| [leaf_mass_review](leaf_mass_review/README.md) | T131 | Independent diagnosis of the estT leaf-mass overcount (the bug behind E45) | Reference repair complete and audited (README) |
| [leaf_triangulation_review](leaf_triangulation_review/README.md) | T136 | Raw-leaf triangulation review | Four-parent capture complete; no full certification (README) |
| [fixed_t6_triangle](fixed_t6_triangle/README.md) | T104 | Fixed tangent frames for six-input triangle masses | Bounded phase complete and reviewed |
| [t6_edge_algebra](t6_edge_algebra/README.md) | T106 | Degree-three spherical edge algebra | Removes 13 transcendental evaluations; pilot CPU gain only 2-3% |
| [vector_triangle_selection](vector_triangle_selection/README.md) | T124 | Guarded vectorised triangle-rule selection | Initial and rescue pilots complete (README) |
| [batched_triangle_fallback](batched_triangle_fallback/README.md) | T116 (triangle) | Grouped positive quadrature fallback | Implementation and pilot comparison done and reviewed (README) |
| [cubic_wide_rule](cubic_wide_rule/README.md) | T154 | Fixed cubic wide-rule discriminator | Eight batches complete; review requested (README) |
| [schur_polynomial_kernel](schur_polynomial_kernel/README.md) | T160 | Fixed-rule Schur polynomial kernel | Trained panel withheld after severe numerical instability (README) |
| [schur_bernstein_rescue](schur_bernstein_rescue/README.md) | T162 | Stable Schur representation rescue | Controls and the 256-cone panel complete (README) |
| [facet_mean_float](facet_mean_float/README.md) | T175 | binary64 facet mean evaluator | Baseline complete and audited; TASKS row Ready (README) |
| [constant_output_shortcut](constant_output_shortcut/README.md) | T177 | Structural constant-first-label shortcut | Controls pass; saved-data audit requested (README) |
| [radial_bound_rescue](radial_bound_rescue/README.md) | T152 | Radial-bound discriminator on 32 unresolved cones | Complete and reviewed (README) |
| [error_cone_semantics](error_cone_semantics/README.md) | T139 | Output-error cone semantics | Point checks complete and reviewed (README) |
| [heldout_error_semantics](heldout_error_semantics/README.md) | T151 | Held-out error-cone semantics | Complete and reviewed (README) |
| [gate_symbol_witness](gate_symbol_witness/README.md) | T138 | Exact gate-symbol sufficiency | Bounded exact structural result, reviewed (README) |
| [penultimate_continuity](penultimate_continuity/README.md) | T132 | Continuity and rank-side localisation of penultimate error | Trained phase complete (README) |
| [face_strip_continuity](face_strip_continuity/README.md) | T159 | Correctness-crossing faces and Gaussian density | Coefficient phase complete; all 88 faces rebuilt independently (README) |
| [axis_strip_continuity](axis_strip_continuity/README.md) | T155 | Support-axis strip continuity with exact word feasibility | Trained phase complete and reviewed (README) |
| [joint_strip_continuity](joint_strip_continuity/README.md) | T147 | Joint-Gaussian strip continuity bound | Bounded theorem done and reviewed (README) |
| [gaussian_margin_slice](gaussian_margin_slice/README.md) | T141 | Conditional Gaussian retained-margin coordinate | Bounded component done and reviewed (README) |
| [prefix_face_projection](prefix_face_projection/README.md) | T164 (prefix) | Unbounded crossing-face projection | Controls pass; 198 projections reviewed (README) |
| [fibre_constraint_review](fibre_constraint_review/README.md) | T99 | Approximate confirmation-order constraints | Bounded phase complete; corrections retained |
| [fibre_cost_review](fibre_cost_review/README.md) | T121 | As-implemented cost audit of the frozen F2L3 fibre recipe | Done and reviewed (README) |
| [fibre_fast_review](fibre_fast_review/README.md) | T128 | Audit of the frozen fast-fibre recipe | Done and reviewed (README) |
| [fibre_suite_cost](fibre_suite_cost/README.md) | T134 | Complete fast-fibre cost recount | All 471 frozen cases reproduce exactly (README) |
| [fibre_t4_recovery](fibre_t4_recovery/README.md) | T100 | Recovery of the frozen H8T4 fibre evidence | Recovery and repair reviewed; TASKS row blocked (README) |
| [fibre_t4_native_cost](fibre_t4_native_cost/README.md) | T180 | Native cost of the T4 fibre recipe | Discriminator complete (README) |
| [fibre_cold_native](fibre_cold_native/README.md) | T180 | Fibre cold numerical closure | Done; full native cost not certified (README) |
| [compiled_geometry](compiled_geometry/README.md) | T66 (geometry) | Compiled geometry kernels with checked semantics | Bounded phase complete; the H16 circle timing target is met |
| [compiled_scalar_transition](compiled_scalar_transition/README.md) | T66 (scalar) | Native scalar Gaussian transition kernel | Bit-identical values; runtime 1.4-1.5x |
| [native_zero_gate_review](native_zero_gate_review/README.md) | T107 | Native exact-zero gate branch | Local repair done |

## Added 2026-10-18: M_16,10 estimators and diagnostics of 2026-10-17 (ARC reproductions, quadrature, tensor, harmonic)

| Folder | ID | Idea | Result |
|---|---|---|---|
| [arc_m43_repro](arc_m43_repro/README.md) | E32 (T36) | Reproduce ARC's two M_4,3 estimators (calibrates "roughly match") | Done. RMS 3.3% / 4.8%; MSE vs sampling ~10.4 / ~2.4 |
| [arc_cumulants](arc_cumulants/README.md) | E35 (T35); T42-R1 | ARC's Hermite-corrected cumulant propagation | Negative: 41-100 pp off at every (H,T,K); review T42-R1 done |
| [harmonic_sphere](harmonic_sphere/README.md) | E34 (T38) | ARC's harmonic deduction-projection, adapted to the RNN | Negative: loses by 1e2-1e7 in MSE (a ~1/D leak across the tie wall) |
| [harmonic_negative_review](harmonic_negative_review/README.md) | T57 | Independent review of the harmonic negative | Reproduces within 5e-14; corrections do not rescue it; the family stays open |
| [anisotropic_harmonic](anisotropic_harmonic/README.md) | T56 | Harmonic sketch on the near-tie hemisphere | H4 noisy RMSE .0116 vs isotropic .0401; no sampling win |
| [ht_seeds](ht_seeds/README.md) | E33 (T2) | Does the H x T tie-slicing collapse replicate across training seeds? | Done. H=16 gain 1,260x at T=3 down to 1.6x at T=10 (H-E11 confirmed, revised; quadrature class) |
| [skeleton_quadrature](skeleton_quadrature/README.md) | E42 (T103) | Deterministic quadrature of the top-3 skeleton integral | Negative: best 3.5-4.1x over MC (target 20x) |
| [sci_tao_v2](sci_tao_v2/README.md) | T20 (P1) | Incremental exact segment walker (sci_tao port) | First port done; review fixes verified |
| [feynman_incremental](feynman_incremental/README.md) | T20 | Incremental Feynman, tube and Laplace ports | Ports implemented and checked; DRI acceptance pending (README) |
| [tube_importance_port](tube_importance_port/README.md) | T20 (cold) | Cold per-model importance weighting | Neither proposal is a validated MSE winner |
| [walker_tolerance_review](walker_tolerance_review/README.md) | T30 | Legacy walker tolerance and completion review | Fixtures expose three defects; corrected v2 matches; DRI fixes open |
| [tensor_mass](tensor_mass/README.md) | T29 | Exact face-mass tensor compression | Mixed negative: low rank present, cross reconstruction unreliable |
| [tensor_cross](tensor_cross/README.md) | T32 | Global tensor-cross face integration | Negative: +2.7 to +23.5 pp bias |
| [tensor_negative_review](tensor_negative_review/README.md) | T81 | Independent review of the TT-cross negative | Grid bias separated from cross error; no family-wide impossibility |
| [tensor_enrichment](tensor_enrichment/README.md) | T81 (enrichment) | Weighted residual support enrichment | Mixed accuracy; no general winner |
| [face_surplus](face_surplus/README.md) | T34 | Adaptive face-quadrature premise check | Negative for the pre-registered smoothness screen |
| [face_surplus_review](face_surplus_review/README.md) | T82 | Complete-stencil review of face smoothness | Rescues do not clear the screen; structural rescue open |
| [network_switch_slices](network_switch_slices/README.md) | T82 (switches) | Actual network-switch partitions on fixed face slices | Mixed; outer quadrature bias and full cost open |
| [facet_transport](facet_transport/README.md) | T31 | Tiny spherical decision-facet transport | DEV screen negative (1.4-1.9x, below 4x) |
| [proxy_face](proxy_face/README.md) | T33 | DEV-only proxy importance density | Negative versus sampling |
| [budget_collapse_review](budget_collapse_review/README.md) | T44 | Budget-matched tie-slicing diagnostic | Bounded; cost semantics audited, outer bias remains |
| [walkfree_gain_review](walkfree_gain_review/README.md) | T45 | Walk-free structural gain predictors | Bounded; correlations only, no estimator |
| [crossover_compute](crossover_compute/README.md) | E30, T19 | Crossover compute per method; negative-result scope audit; Codex-DRI handoff file | E30: all 18 group ratios above 1 (3.1-4,707); T19 audit open |
| [mechanism_checks](mechanism_checks/README.md) | E29 (T8) | Phase clocks and orthogonal mode checks | Done: H-M1 supported (qualified); H-M3 weak |
| [mechanism_independent_review](mechanism_independent_review/README.md) | T39 | Independent mechanism comparison | Bounded phase done; T27 follow-up |
| [phase_history_review](phase_history_review/README.md) | T27 | Exact phase-history decomposition | Bounded; not a weight-only predictor |
| [position_patch_review](position_patch_review/README.md) | T26 | Controlled position patching | Bounded; no unrestricted causal claim |
| [rank_gate_tail_review](rank_gate_tail_review/README.md) | T28 | Rank law versus gate map | Bounded; diagnostic only |

## Added 2026-10-18: M_16,10 structure diagnostics (mostly child branches)

| Folder | ID | Idea | Result |
|---|---|---|---|
| `width16_probe` [b194] | T194, T216, T217, T222 | recipe108 at the target width ((16,3), (16,4)); weight-symmetry quotient; modal gate words; prefix mass coverage | Done (unverified): at (16,4) cells ≈ regions; T216 and T217 KILL; T222's coverage counts feed H-E13 |
| `random_control` [b194] | T232 (H-E14) | Trained M_16,10 vs norm-matched random weights | H-E14 CONFIRMED (scoped k <= 4) after T280: trained prefix regions are generic |
| `shatter_test` [b194] | T239 (H-E15) | Is the error part shattered while the correct part is simple? | H-E15 CONFIRMED (scoped k <= 4, T <= 4): 26-27x more cells per unit mass near ties at T=4 |
| `tie_blind` [b194] | T223 | Tie-slicing with the blind region split off | Done (unverified; quadrature class): the best honest deterministic number, ~1.1-1.3x on the clean target (README) |
| `rate_T` [b194] | T277 | Dimension factor as the cause of the H-E11 collapse | Refuted (unverified): rate .49-.67 at every T from 3 to 10 |
| `rate_alpha` [b194] | T271 | Convergence rate per deterministic family; activity-matched random control | Mixed (unverified): probe biases do not move with budget |
| `tie_tail_labels` [b218] | T218, T272 | Mechanism-labelled tie-slicing tails; kink density; in-band-flip freezing | KILL (unverified): the tail is a continuum of rare events |
| `neuron_account` [b241] | T241 | Quantitative per-neuron account of M_16,10 | Target missed; core CONFIRMED by an independent review (DRI) |
| `gap_formula` [b241] | T269 | Logistic error probability given the gaps, integrated over the exact order-statistic law | Negative (unverified): transfer across models off by 0.4-19.5 pp |
| `crown_tree` [b241] | T279 | CROWN-style linear-relaxation certification tree on S^9 | Negative (unverified): uncertified mass .35 at 1e5 cones at (16,4) |
| `rival_bounds` [b194] | T289 | Joint rival-event Gaussian cone bounds (Astra r2 C1) | Negative (unverified): valid but vacuous enclosures |
| `shadow` [b194] | T290 | Algebraically coarsened shadow network (Astra r2 C2) | KILL (unverified) |
| `stationarity` [b194] | T310 | Training-stationarity identities as family knowledge (Fable r3 B4) | STOP (unverified) |
| `capacity_ceiling` [b192] | T234 (and T261's GPU check) | Accuracy ceiling of the architecture: more data, schedules, hard cases | Running (interim): the best retrains reach ~.98; M_16,10 sits near the 75th-85th percentile |

## Added 2026-10-18: cheap regime and the sampling-free lineage (child branches)

Frozen estimators are scored by F on the open DEV2 panel and, once, on a sealed panel by the independent scorer T281.

| Folder | ID | Idea | Result |
|---|---|---|---|
| `cheap_frontier` [b286] | T286 | Every existing deductive estimator at its cheapest settings, scored by F | Negative (unverified): nothing deductive within 7x of the lenient bar on DEV2 |
| `lenient_bb` [b218] | T274b | Lenient push, exact lane: branch_bound per budget; rank_exact + exact last m steps | Negative (unverified) |
| `probe_v5` [b241] | T274a | Probe v5a/v5b and a QMC control v5q | Negative for the mechanistic lane; v5q (inductive) passes lenient on HP1 |
| `frozen_check` [b194] | T282 | Independent check of T274b's frozen-gate kill | As written the agreement failed; post hoc, with matched classes, it reproduces T274b within 25% |
| `frozen_cheap` [b194] | T287 | Witness / frozen-gate class estimators at F <= 10 | Negative, NO GO (unverified) |
| `blind11` [b194] | T293 | Deductive blind-region rate (Fable r2 C3) | Inconclusive; uninformative in substance |
| `meanfield` [b194] | T294, T294b, T295 | Class-conditioned mean-field ReLU closure; readout diagnostic | T294/T294b inconclusive; T295 dead (verified, scoped) after T299 |
| `mf_verify` [b292] | T299 | Independent check of the mean-field readout negative | Reproduces T295 in substance |
| `oracle_floor` [b292] | T292, T296 | Oracle bias floor of the linear closure; ANAT-80 error anatomy; R0 on LB3 | T292 DEAD by rule (floor 4.77 pp); T296 R0 NO-GO (RMS 1.39 pp on LB3) |
| `r0_variants` [b241] | T298 | Oracle floors of R0-family point closures | GO by top-4 exact (.389 pp); post hoc top-4 + spread lows .244 |
| `closure_recheck` [b241] | T303 | Re-check of top-4 + spread lows, and its integration cost | Reproduced within 3%; integration needs >= 10k FPE |
| `class_sharing` [b241] | T305 | Share per-class closure errors across position classes | Negative: no scheme >= 4x at <= .25 pp |
| `r0_integral` [b218] | T297 | Deterministic skeleton integral of the R0 closure within 4k FPE | DEAD for threshold designs (unverified) |
| `r0_shared` [b218] | T300 | R0 boundaries shared across the max's position (lenient, retrains) | Stage 0 lenient pass on DEV2; HP2 F 8.89 [6.42, 11.86] (upper CI above 10; inductive step); stage 1 stopped |
| `r0_walk` [b218] | T302 | R0 integral by exact segment walks along the gap lines | FAIL by the pre-registered line; the best deductive F at the time |
| `r1_walk` [b218] | T304, T306 | R1W: walks plus a v4-aware closure | Narrow FAIL; frozen V1 on HP2 F 14.92 [10.74, 19.38]; estimates VERIFIED (T312) |
| `r1_b2b5` [b218] | T307 | R1W V1 + exact prefix sharing (B2) + exact last-input marginalisation (B5) | Stopped at the pre-registered B5 check |
| `r1_skip` [b218] | T311 | Skip walks that provably cannot flip | Negative: killed by arithmetic before building |
| `r1_v1b2` [b218] | T313 | R1W V1 with executed B2 sharing and instrumented FLOPs (cost only) | HP2 recount 16.83 without B2, 13.51 with B2; ruling: B2 counts only once executed, so the official V1 F stays 14.92 (all fail) |
| `r1w_repro` [b241] | T312 | Independent re-implementation of R1W V1 from its README | Done: all 80 DEV2 estimates within .012 pp; DRI marked the estimates VERIFIED |
| `r1_capped` [b218] | T315 | Capped exact walk + declared linear extrapolation | FAIL (DEV2 F 900-1,700) |
| `r1_cont` [b218] | T317 (b-f), T325, T326, T330, T331, T335, T337, T344, T349, T354 | Continuity restarts + executed B2, then the frozen T317c-f family; noise curve; rare-event bounds; T340c | T317c HP3 F 5.08; T317f HP4 F 2.63 [1.93, 3.39], lenient pass, strict fail; T326 knob cuts exhausted; T337 breakdown at .1-.25% noise; T344 scoped negative; T349 accurate (K5: 9/9 E1 within 2x) but 1.8-4.5M FPE; T354 running. A long-line walk bug (14:18) makes T349/T354 suspect |
| `t317c_confirm` [b241] | T320 | Independent confirmation of T317c (FLOPs, deductive audit, estimates) | Estimates within .0077 pp; FLOPs +8.9% (benign); DRI: confirmed lenient candidate on HP2 (scoped) |
| `budget_t317d` [b292] | T324, T333, T347, T351, T353 | Error budget and deductive audit of T317d; product overstatement; T317h combined attempt; cost-only spread+tilt | T324's two largest terms CONFIRMED (T334); T347 NO GO; T351 NO FREEZE; T353 GO on DEV2 (F 2.60 at 5,079 FPE) but does not confirm on DEV3 (T355) |
| `budget_check` [b241] | T334 | Independent check of T324's two largest error-budget terms | Both CONFIRMED (README) |
| `smooth_verdict` [b292] | T319 | Gaussian-smoothed closure (Fable r4 D3) | STOP; dead (verified, scoped) after T327 |
| `closure_floor` [b286] | T327, T339 | Re-check of T319 and its untried variants; T317g with the spread-lows closure | T327 reproduces T319; T339 NO FREEZE (DEV2 error sd .667 -> .455, but bias +.27 pp; the sd cut does not replicate on DEV3, T343) |
| `kink_surrogate` [b241] | T318 | Kinked-tangent surrogate (Fable r4 D2) | STOP: integration RMS 6-13 pp vs .72 for the exact walk |
| `mix_sketch` [b292] | T309 | Future-aware mixture propagation (Fable r3 B3) | STOP: dead in the cheap regime (scoped) |
| `info_ceiling` [b241] | T308 | Information ceiling (Fable r3 B1) | GO with a scope caveat (.477 pp, uses rank_exact); inductive diagnostic |
| `info_ceiling2` [b241] | T314 | Strict information ceiling with <= 4k FPE of features | Information-infeasible (scoped; best .855 pp) |
| `info_ceiling3` [b241] | T332 | Paradigm-level ceiling from T317e's own per-row outputs (Fable r5) | NEITHER go nor stop (.577 pp) |
| `mech_closure` [b241] | T338 | Lows chosen from the weights treated exactly, the rest pooled | NO GO: no better than simple recency |
| `lows_orderstat` [b241] | T341 | Order-statistic test of T338's running-max reading | NO GO; the running-max reading is refuted |
| `t339_check` [b194] | T343 | Independent check of T339's decomposition on DEV3 | Done: does NOT replicate (sd ratio .912 [.76, 1.11] vs .68 on DEV2); the variance-only reading is withdrawn |
| `t353_score` [b194] | T355 | Second scorer of T353 vs T317f on all 100 DEV3 models (scoring only) | Done (unverified): T353 does NOT beat T317f on DEV3 (F 3.32 vs 2.49; +53% C) |
| `candidate_dossier` [b241] | T323 | Progress note: how far the sampling-free estimator gets on retrains | Done: HP3 F 5.07; a progress note, not a candidate solution |

## Added 2026-10-18: rare events, exact methods, infrastructure

| Folder | ID | Idea | Result |
|---|---|---|---|
| `rare_events` [b286] | T340 | Can the exact walks estimate error probabilities of 1e-4 to 1e-7 better than sampling at equal compute? | Partial positive (unverified, post hoc): best variant per event family 10 win / 3 tie / 2 loss of 15; each fixed variant 5/15. Not replicated on DEV3 (T350); variant B suspect (walk bug) |
| `rare_check` [b241] | T350 (filed as T349) | Independent check of T340 on 6 fresh DEV3 models with its own code | Done: NO GO on 6 fresh DEV3 models; family rule 11 win / 6 tie / 13 loss. Its own walk is verified exact, so the negative stands; the E2/A win replicates, E1/B fails |
| [exact_brainstorm](exact_brainstorm/README.md) | T346 | Clever exact methods, and exact computation as a component | Proposed, low priority: seed ideas A-J with kill tests; no runs (README) |
| `gpu_cost_law` [b192] | T192, T193 check, T222, T270, T276, T284/T321/T328/T336, T342, T348 | Compiled/GPU exact reference and cost law; prefix counts; sealed panels HP1-HP5; open DEV2/DEV3; GPU census | T192 running per INDEX (the port reproduces E46/E48; (8,5) fails seed 2); H-E13 counts; HP1-HP5 sealed (HP5 the reserve); DEV3 done (T342; T317f F 2.49 there); census T348 queued |
| `hp1_scorer` [hp1] | T281 | Independent scorer of frozen estimators, the only reader of sealed truths | Standing service (running): HP1-HP4 scored |
