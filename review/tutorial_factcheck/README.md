# T384: independent factual review of the interactive tutorial

## Question

Does the current tutorial accurately teach the project’s mathematical problem, measured results, evidence limits and references to an ML-literate reader with no prior background?

## Method

Fresh Astra reviewer, separate from the original Astra writer. Claim **5d236123**, 2026-10-19. I read the shared tutorial at **7a388e22** before opening the earlier T382 critique/handoff, then compared the newer owner snapshot **a3ab9caa**. Required workflow reading included tutorial/README, which contains a historical review summary; I did not use the earlier T382 findings to form my first assessment. The owner snapshot already repairs several first-pass issues (F05). Findings below are against a3ab9caa unless explicitly identified as historical. Later owner integration 97ee97a4 is retained during revision. The original writer owns edits and the response matrix; this reviewer owns the independent findings and final verification.

Review covers source prose, embedded JSON, figure code and generated-source fidelity. It traces selected claims through INDEX, actual README/source artifacts and published aggregate JSON; it does not rerun networks, training, geometry, large Monte Carlo or sealed truths. Small saved-data arithmetic is in [verify_saved.py](verify_saved.py), with its output in [checks.json](checks.json). Immutable source pins and direct links are in [sources.json](sources.json). Root separately verifies rendering and interaction of the final page.

Two competing explanations were considered before calling a mismatch a defect: a figure may deliberately use a different estimator/panel/bin, or a newer owner update may already repair stale wording. The cheapest discriminating checks were the embedded arrays, the original method definitions and the owner-source diff. A missing reproduction does not make a scientific result false.

## Status

**Done: revised tutorial independently rechecked, factual review PASS.** Seventeen findings were sent directly in actionable batches and resolved in writer revision **68ba066c**, including preserved upstream owner fixes and explicit qualifications. See [final verification](FINAL_VERIFICATION.md) and the [writer response matrix](../../tutorial/REVIEW_RESPONSE_T384.md). This is an editorial evidence audit, not a change to scientific ledger verdicts. T374 retains status.json authorship and publication at the existing private Artifact URL; republication of this corrected successor is a separate pending action.

## Results

**The main measured results are supported by the cited records.** The official HP4 score is F = 2.63149, RMS 0.748829 pp and bias +0.436364 pp at 3,302.05 FPE. The important corrections concern interpretation, figure labels, numerical exactness, accounting conventions and scope. No new experiment is needed to repair them.

Severity: **high** changes a central interpretation, mathematical claim or displayed evidence; **moderate** is a material qualification, provenance or teaching issue. Confidence is confidence in the finding, not a fresh scientific confirmation of the underlying experiment. Locations below use source line numbers in a3ab9caa (same line numbering as the reviewed shared draft for these passages).

### F01 — Figure 10’s displayed wrong length is approximate

**High; confirmed defect; high confidence.** At source line 497 the caption says the walk measures the wrong stretch “exactly.” In the same file, lines 963–965 classify six midpoint subintervals per gate piece; they do not split at every readout crossing or true-rank tie. Lines 937–947 also use tolerances, a 4,000-piece cap and a 1e-9 advance. These are direct implementation facts, not a speculative numerical failure.

**Replacement:** distinguish the mathematically exact piecewise-linear method from the numerical browser demonstration. Say the red strip and reported wrong length use six midpoint checks per piece and are approximate. Do not change the model or walker to claim a new certified computation.

**Resolution update:** owner v4 (shared 97ee97a4 / owner 0039119b) replaced the six-midpoint strip with splits at true-answer ties and every pairwise output-score crossing before the writer's T384 correction. The final page preserves that algorithm and qualifies its numerical tolerances. The six-midpoint description above applies only to the pinned finding baseline, not the final implementation.

### F02 — “Blind” inputs are not the all-negative orthant

**High; ambiguity with a false implied bound; high confidence.** Line 402 moves from “almost all” negative to probability 1/1,024, then attributes about 0.7 percentage points to “these inputs.” The actual blind event is **at most one positive coordinate**, with probability (1+10)/2^10 = 11/1,024. The all-negative subset has probability 1/1,024 and therefore can contribute at most 0.09765625 pp of error, not 0.7 pp. The T234 primary grid table defines the event explicitly and reports .0065–.0074 error across the ordinary training configurations.

**Replacement:** name both events and their probabilities. Attribute the approximately 0.7 pp to the broader blind event in the tested training recipe. More training did not remove it in those runs; other training distributions were tested and reduced it at a cost elsewhere. The quoted 60% quantity is a face **second-moment** share, not automatically the same quantity as variance.

### F03 — The relative-noise widget uses a project convention

**Moderate; confirmed attribution defect; high confidence.** Figure 8, line 423, calls sigma times each matrix’s RMS “ARC’s evaluation rule.” The widget implements exactly that relative-noise rule (lines 895–896), but INDEX H-E4 explicitly says the rule and levels are project choices. The primary E32 reproduction uses **absolute** sigma .05 for M4,3; ARC did not specify a transferable M16,10 noise level.

**Replacement:** “our relative-noise convention,” with the distinction from ARC’s M4,3 example. Preserve the T337 curve and its stated panel.

### F04 — E32’s small-model comparison needs its original accounting

**High; confirmed classification/context defect; high confidence.** Lines 327–335 treat both reproduced ARC estimators as mechanistic and attach 17,500 operations, approximately 88 cones and 420x to the clean checkpoint without panel qualification. The primary arc_m43_repro README reports these as **means over noisy copies** under its historical convention (multiply-add = one operation; sampling = 72 operations). Estimator B samples its six-dimensional integrals, and that sampling cost was uncharged in the quoted comparison. A’s approximately 10.4 ratio uses ARC’s crude volume-operation estimate; A’s measured clipping cost gives approximately 1.4. The 420x comparison targets **0.1 pp RMSE**, not a specified-confidence interval of ±0.1 pp.

**Replacement:** retain the historical results with their panel and cost convention, separate A from B in the mechanistic column, and disclose the uncharged sampling in B. Do not translate historical operations into the modern 5,948-FLOP target FPE without an explicit conversion.

### F05 — Owner corrections already resolve several first-pass defects

**Moderate; historical defects, owner-resolved; high confidence.** The a3ab9caa owner diff fixes the gain numerator, the sampling F-floor sentence, the 30-model sequence-length list, HP2’s reuse, the Poisson-versus-binomial caption, rarity-independent cost wording and “any puzzle” wording. These are not new failures of the adopted baseline.

**Verification:** with q = 1−p, n independent face contributions of mean q and CV, cost C = nc, the gain is [pq/C]/[q²CV²/n] = **p/(q CV² c)**. The old formula used 1−p in the numerator even though p elsewhere means accuracy. Plain sampling’s expected score under the stated floor is max(C,1000)/C, so it equals one only for C≥1000. HP2 served five submissions; HP3 also evaluated a later numerical/cost variant. Preserve these qualifications consistently in the glossary. Figure 12’s function additionally uses a rounded normal approximation when the Poisson mean is at least 30; do not describe all settings as exact Poisson draws.

### F06 — Figure 7b labels the wrong bins and omits the proxy’s definition

**High; confirmed display defect and missing scope; high confidence.** Line 405 says near <.05 and far >.1, but source line 884 plots 358k/2.9k and random 32–170k/11–25k. The original shatter_test table (3) puts those values in **d<.02** and **d≥.2**. Their trained ratio is approximately 125. The broader .05/.1 contrast is 44.1 in the original prefix calculation and 37.2 in its independent check, not 120. Moreover, each cell’s whole mass is assigned to the bin containing its center; this is not the input probability actually inside the angular band. INDEX H-E15 and SCALING_OBSTRUCTIONS section 7 explicitly preserve that caveat.

**Replacement:** relabel the plotted bins, explain the center-assigned density proxy, separate broad-bin from extreme-bin ratios, and restrict the evidence to k≤4/T≤4. The distance formula is a normalized distance-to-plane proxy (sin of the angular distance), so “radians” is at best a small-angle shorthand. Do not infer literal relocation of the same individual regions from a trained-versus-random ensemble comparison. The unsupported “93% of all regions” sentence should be removed or sourced to an exact matching definition.

### F07 — Figure 9 hides a smaller endpoint sample and overstates “ceiling”

**High; confirmed display defect plus unsupported generalization; high confidence.** Line 431 says 100 retrains per setting, and line 929 labels the curve “best of 100.” Embedded panels.json and the primary T234 grid table both say **25** at 2^34, versus 100 at the other plotted settings. Maxima from unequal sample sizes need that disclosure. Finite training runs show an observed best near 98% under tested recipes, not an architectural ceiling. The same embedded DEV3 data put the target above 81 of 100 retrains, so “typical, not unusually good” is misleading without specifying the training-data comparison.

**Replacement:** name the 25-model endpoint and label the maxima accordingly; say “best observed under these training settings.” State the target’s position in DEV3 or explicitly restrict any typicality comparison to the longer-trained panels. No new population claim follows from these 100 models.

### F08 — The rare-event narrative blends K3 and the graded-g12 fix

**High; confirmed estimator/provenance mismatch; high confidence.** Lines 543/553 imply the finer graded rule is what T368 independently reran. Primary rare_deep/README explicitly uses its own **unchanged K3** implementation, 10/10 within 2x with ratios .88–1.39. T366’s graded rule is 8/8 at 1.10–1.30 on a different, fresh ordinary-case test, but plain K3 is also 8/8 there; the source says this test does **not discriminate the fix**. The independent recheck saw public truths and checks implementation, not fresh selection.

At line 555, 21 wins/22 ties/67 losses pool **110 E1 and E2 cases**. The primary T360 report charges 1.0–1.5M FPE for the three E1 thresholds together and 3.7–6.7M for the four E2 thresholds together. It is not a 110-case comparison all at 1–1.5M FPE. Figure 12 rescales the same eight reported ratios to arbitrary slider p; those dots are illustrative relative errors, not fresh estimator runs at every p or selected budget.

**Replacement:** explicitly distinguish K3, the graded fix, its non-discriminating ordinary-case test, and the triple-tie failures. State the pooled comparison and family-level costs accurately. Keep the recorded confirmed/unverified scopes; do not independently promote a status.

### F09 — The official headline score is right, but the cost caveat matters

**Moderate; omitted documented accounting sensitivity; high confidence.** Official HP4 aggregate JSON confirms F=2.6314938, C=3302.0487, RMS=.7488285 pp and bias=.4363636 pp. The same scorer’s README and public summaries identify an omitted screen charge: C≈3352/F≈2.67, or C≈3743/F≈2.98 under stacked alternative counts. Both leave the strict failure/lenient pass unchanged.

**Replacement:** keep the official F=2.63, clearly labeled, and add one short cost-sensitivity sentence. “Network runs” here means arithmetic **equivalents**, not that the estimator literally runs 3,302 independent inputs or has matching wall time. The F formula’s C is the stated mean-cost panel convention; model-varying costs need a declared aggregation rule.

### F10 — Uncertainty labels conflate a standard error with a fixed guarantee

**Moderate; confirmed display/statistical defect; high confidence.** Line 915 gives every noisy copy “±0.15 pp from 20,000 samples.” Binomial SE is 100 sqrt(p(1−p)/20000): approximately .15 pp near the clean target, but up to .354 pp at p=.5. The displayed ± figure also does not identify its confidence level. The 74,000-run guide is approximately .1 pp **RMS** for panel variance near .074; the target’s variance needs about 44,684 runs. A 95% half-width would cost about 1.96² times the corresponding RMS count.

**Replacement:** label one standard error and its dependence on p (or compute the displayed estimate’s SE), identify the guide’s panel/variance assumption, and use RMS language for the calculator. Root authorizes narrow label/readout changes only; preserve model/sampling logic.

### F11 — The no-coincidence reference and formal scope are misstated

**High; confirmed reference defect and overbroad summary; high confidence.** Lines 286/591 attribute the cited 2025 post to Christiano, Neyman and Xu. The primary [A computational no-coincidence principle](https://www.alignment.org/blog/a-computational-no-coincidence-principle/) is **Eric Neyman, 14 February 2025**. Its concrete conjecture concerns reversible circuits on 3n bits and the property that no input ending in n zero bits maps to an output ending in n zero bits. A polynomial-time verifier accepts some polynomial-length advice for every satisfying circuit, while accepting none for 99% of random circuits. It may accept individual circuits without the property: this is weaker than sound proof verification.

**Replacement:** separate the broad motivating intuition from that restricted formal conjecture. Label 10%-versus-95% neural accuracy as an analogy motivating ARC’s research, not a consequence of the conjecture or a measured probability of an “outrageous coincidence.” Correct the author/date and link the primary reference. Both newly consulted ARC posts are from 2025, so no new post-January-2026 source log entry is required.

### F12 — Complexity results need the problem and error model specified

**High; unsupported transfer/overbroad inference; high confidence.** Lines 288–294 blend #CircuitSAT, exact polytope volume, general ReLU accuracy, fixed Gaussian M16,10 accuracy, deterministic CAPP and membership-oracle volume approximation. These are different computational problems. Boolean-gate simulation supports general-network hardness with an appropriate **Boolean input distribution and label task**; it does not by itself prove hardness for this fixed bias-free Gaussian architecture. An arbitrary exact Gaussian probability is not generally a finite rational count. Likewise, classical convex-body oracle lower bounds concern worst-case approximation factors under their oracle/geometric assumptions, not exponentially large absolute error in a probability bounded by one. They do not prove that every useful estimator must read weights.

**Replacement:** present the counting and volume results as worst-case context. State that an efficient deterministic uniform worst-case CAPP algorithm under specified precision/runtime guarantees would have major derandomization consequences. Do not claim any vaguely defined method that “beats sampling on every network” automatically resolves those problems. Kabanets–Impagliazzo is specifically about polynomial identity testing; Williams’s cited algorithms concern related circuit problems. Valiant’s permanent paper establishes a canonical #P-complete problem; #CircuitSAT is the standard accepting-witness count. Make that citation distinction rather than implying the permanent paper is a theorem about this RNN. The oracle paragraph should say black-box guarantees can be weaker, not establish a universal neural-network impossibility theorem.

### F13 — The closure compresses the bottom seven; it does not equate all seven

**Moderate; confirmed method-description defect; high confidence.** Lines 503–504 and the glossary say seven smaller inputs become one value, followed by a separate fourth-largest point. The actual T335 source retains the top four, represents v4 at one quadrature node in the latest free position, and pools the **six** smaller values with a linearized conditional mean. The current wording conceals the v4 placement and double-counts its status.

**Replacement:** “compress the bottom seven: represent the fourth-largest by one point and fixed placement, and pool the remaining six.” The simplified bullet list may remain pedagogical but must not contradict the actual closure.

### F14 — The ladder quote uses an obsolete mixed-unit cost ratio

**High; confirmed stale accounting/context defect; high confidence.** Line 370 gives E46’s worst ratio .27 without saying it predates T196. Primary review/cost_convention documents an estimator-operation/sampling-FLOP mismatch. Its conservative two-FLOPs-per-operation restatement gives .523; the later implementation audit’s 1.37 multiplier gives approximately .36. Both remain below one, but .27 is not the corrected equal-FLOP result. The 2.2e11 operations and 70% exact cost in E48 are **seed 0 only**, not a cost established for all the winning T5 seeds. The old small noisy panel is not the later full rung protocol.

**Replacement:** identify historical versus corrected conventions, retain the unchanged pass/fail conclusions within their original panel scope, and restrict the E48 cost to seed 0. Do not suggest the mostly-exact recipe literally merges all equivalent cells: it also decides which splits can be safely skipped under its margin rules.

### F15 — Quantization expectations currently read like results

**Moderate; unsupported inference/omitted state; high confidence.** Line 441 asserts that quantized weights still produce the measured 23–33x region growth. That number belongs to the full-precision model; discrete/zero weights can alter partitions. Binary hidden activations do give at most 2^16 hidden states, but hidden state alone does not retain the true second-largest input’s rank/value information. TASKS T352 explicitly proposes a state/input-interval/**label-summary** computation. Discretized inputs also require a probability law and a tie convention. FP4/FP8 bit patterns are not all distinct ordinary real activation values, and equal bit count does not establish equal reachability.

**Replacement:** keep these as hypotheses, describe the augmented state required for accuracy, and say solving a quantized variant solves that modified model—not automatically ARC’s original real-valued challenge. Do not infer unchanged region growth or exact-DP practicality without its experiment.

### F16 — Exact-route and symbolic negatives exceed their proved scope

**High; unsupported universal exclusion; high confidence.** Line 408’s “Consequence for any exact method,” approximately 10^12 wrong-boundary pieces and universal “scaling is the only exact symmetry” do not follow from the k≤4/T≤4 density measurements. The independent T229 theorem only excludes nonidentity invertible global transformations commuting with ReLU everywhere and preserving the actual weights/logits. Its TARGET_RESULTS explicitly leaves reachable subsets, argmax-only invariance, noninvertible quotients and other representations open. A reviewed extrapolation is still not a boundary census or universal lower bound.

Line 472’s “every small input ... however long ago” concerns a finite **10-step diagnostic**; it does not exclude every finite-state/symbolic representation. The relevant sci_poincare source tested particular record-symbol and deterministic-low-input reductions.

**Replacement:** retain the DRI’s scoped practical judgment against tested region-by-region enumeration and measured growth/extrapolation. State unknown boundary counts/aggregation opportunities rather than a universal exact-method obstruction. Restrict the symbolic negative to the tested reductions and horizon. No ledger status change is needed.

### F17 — Mathematical exactness, architecture claims and evidence access

**Moderate; ambiguous teaching claims; high confidence.** Lines 303/327/354 describe “exact arithmetic,” “zero error” and exact enumeration without distinguishing an exact decomposition from its floating-point solid-angle/quadrature implementation. The E32 reference computation is the scorer’s own enumerator, so zero error against itself is not an independent roundoff certificate. Line 324’s two-neuron “cannot do the task” turns one trained checkpoint’s approximately 41% accuracy into an architectural impossibility claim. “Face” is also used for the line segment itself rather than the tie-wall point/patch parameterizing a transverse slice. The original standalone page provides many record IDs but few actionable source links.

**Replacement:** describe exact geometric formulas evaluated numerically, preserve measured precision qualifications, say “this trained two-neuron model,” explain face/slice terminology consistently, and add a direct repository/primary-source path for readers. Define quadrature generally as weighted numerical integration; qualify this project's fixed/Sobol-rule classification as its reading of ARC's heuristic, rather than attribute a categorical ban on every quadrature rule to ARC. These are pedagogical corrections, not new negative architecture results.

## Verified-claim checklist

“Verified” below means traced and arithmetically checked at the specified level; it is not DRI confirmation of a new scientific claim.

| Claim | Independent check | Scope/result |
|---|---|---|
| 432 weights, h0=0 recurrence, 720 top-three placements | Embedded weight dimensions; 16+16²+10·16; 10·9·8 | Correct parameter/placement arithmetic; no new forward run |
| Target .9531175; 300B sampling SE | Tutorial constant and T192 provenance; sqrt(pq/3e11) | .00003859 pp is **one SE**, consistent with .00004 pp; raw 300B run not reproduced |
| HP4 F2.63, CI[1.93,3.39], RMS.75, bias.44, cost3302 | Public immutable scorer aggregate JSON; C/F=n*; RMS/sqrt(F) | Values correct; sampling RMS≈.4616 pp; F09 cost caveat required |
| Panel score and sampling variance | Bernoulli variance and stated project formula | Correct for stated common/mean-cost convention, with floor qualification; no CI bootstrap rerun |
| Gain law | Independent variance derivation | p/[(1−p)CV²c], repaired by owner; iid-face diagnostic, not arbitrary deterministic quadrature law |
| T3–10 Figure 5 points | Parse all 30 embedded triples; compare original d5_law.log | Values trace to saved log; lengths3,4,5,6,8,10; mixed individual gains around1 atT6/T8 need geometric-mean wording |
| E32, E36/H-U1 small-model wins | Primary E32 README/results and independent R2/R3 records | Positive remains scoped; historical conventions and B sampling disclosed; no retraining/reproduction |
| E46/E48 ladder | Primary cost-convention audit and INDEX’s corrected scope | Historical.27 cannot stand as corrected equal-FLOP metric; T5 cost seed0 only |
| Prefix2,34,792,21032; k5=632394; k6≥19740596 | Published exact recount, H-E13/T270 record | Prefix≤4 independently confirmed in ledger; k6 floating geometry lower-bound qualification preserved; no enumeration rerun |
| Entropy-growth guide5.1,4.6,4.1,3.6 | T348 primary census README | Correct k5–8 rates; graph is deliberately anchored to total k4 count, not actual entropy count; later census sampling limits matter |
| Figure7b densities | Original T239 table and independent H-E15 record | Numbers real, bins mislabeled and proxy definition omitted; F06 |
| DEV3 100, range85.1–97.1%, mean92.5% | All100 embedded scalars | Correct rounded summaries; target above81/100 |
| T234 means/maxima | Embedded rows versus primary grid3 table | Values agree; n25 endpoint and finite-recipe scope required |
| Rare E1 eighteen-of-eighteen; deep ten-of-ten | Primary T360/T368 READMEs and saved reported ratios | Correct scoped results; K3/graded distinction, known-public truth and mixed costs required |
| ReLU cones and spherical mass | Positive homogeneity; rotational invariance | Exact mathematical facts under no-bias/isotropic-Gaussian assumptions; numerical implementation separately qualified |
| No-coincidence and matching-sampling primary posts | Read original2025 pages | Proposal/conjecture, not proof of target tractability; corrected author and formal scope |
| Complexity references | Check stated problem against mathematical definitions and cited scope | General counting/volume/derandomization context; no target-specific hardness theorem established |

## Compute accounting

No model forward passes, training, random inputs, geometry solves, large-array replays or numerical quadrature. Only repository reads, small JSON/text parsing, exact integer arithmetic and scalar square roots. The saved-check script records its process CPU time. HTTP reads were limited to the two cited 2025 ARC posts; no messages or contact to ARC. One initial sandboxed read and one optional parser import failed; neither produced scientific data, and the successful read/parser is recorded transparently in sources.json. The first saved-check attempt incorrectly counted nested weight-array rows instead of scalar entries; its assertion failed before output. The check was corrected to count scalars recursively. This was a reviewer harness error, not a model or tutorial defect.

## Claims

- [measured] The source/JSON mismatches above exist in the pinned baseline; saved headline aggregate arithmetic agrees with the published values.
- [inferred] Correcting these issues materially improves factual and pedagogical reliability without changing the main unsolved verdict or the scoped positive results.
- [speculative] None of the edits establishes a new estimator, architectural ceiling, general impossibility or future performance guarantee.

## Alternatives not ruled out

The underlying research may have newer results not integrated into this snapshot. T374’s live statuses can change independently; those remain its responsibility. Unreviewed raw experiments may contain additional numerical issues. Counts, geometry and Monte Carlo truths were not newly reproduced. General complexity or symmetry claims could be established under additional assumptions, but this tutorial does not supply them. The reviewer did not open sealed HP5 truth data.

## Failure modes

Treating this audit as a full scientific replication; applying a source’s status to a different estimator or panel; publishing an old generated file; overwriting T374 status updates; or using “inferred” to shelter a universal mathematical assertion without evidence. Final factual verification must compare the writer’s actual revision and response matrix, not merely this recommendation list.
