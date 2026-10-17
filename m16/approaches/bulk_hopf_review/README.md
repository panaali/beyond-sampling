# T178: independent DRI bulk44 Hopf evidence review

## Question
What do the saved H4T4 Hopf panels establish, which arithmetic/inference/cost qualifications remain, and is the claimed exact circle partition correct on coefficient-zero controls?

## Method
Claim068300d4 was pushed before implementation, after root direction and explicit h4/910 ownership checks. Freeze DRI ac321d97 source/raw files with hashes in manifest.json. Initial numerical scope is the existing25 H4T4 models (five zoo seeds, each clean plus four noise50 copies), not a new model panel. Reproduce report ratios by exact rational contractions, preserving pooled clean/noise and reporting those populations separately. Retain raw discrepancies and the clamped MC-variance subtraction; a finite Hopf128 reference is a comparator, not independently certified truth.

Freeze 95% simultaneous marginal binomial intervals for these25 saved MC estimates, using Bonferroni .05/25. They remain valid without independence across models sharing input streams. Bound each estimator's squared discrepancy and p(1-p) on the resulting truth box, then aggregate; inconclusive upper bounds are not method failures. Numerical beta inversions and recorded MC provenance must be checked independently before claiming certified coverage. No new MC, exact reference, checkpoint load or input evaluation is authorized in this initial phase.

Source inventory found `_cuts` concatenates plus roots for all columns followed by minus roots for all columns, while its nonzero mask repeats each column twice. Freeze two-column coefficient cases [1,0], [0,1], [1,-1], [0,0] with B=0, on (0,2pi) and (pi,2pi), plus a mixed-row fixture. Analytic expected roots are multiples of pi/2. Verify root membership and classification between returned cuts. Mixed zero/nonzero cases discriminate this mask defect; all-zero/all-nonzero alone cannot. A separately identified local tile-mask rescue may be tested only after preserving original failures, with all other operations unchanged. This does not establish any saved trained case contains the defect.

## Status
Sources and protocols frozen; arithmetic and controls pending. T115 completed handoff71d71050/reviewa558c8a9; k29 continuation remains Ready.

## Results
None yet. Source-level concerns are hypotheses awaiting the frozen checks. The source's unqualified continuity and approximately N^-1 convergence claim needs separate nondegeneracy/rate assumptions.

## Compute accounting
Read-only source/data inventory, finite arithmetic, beta inversion and small analytic circle-root controls only. One thread,2GB, <=20CPU review lane. No new model/reference/MC runs. Inherited native96 counts stay labeled; inspect source expressions and sampler convention before any same-unit conclusion.

## Claims
[measured from source] Saved report pools each clean model with four noisy models. [open] Statistical strength, circle partition correctness in mixed coefficient-zero cases, quadrature accuracy and complete cost accounting.

## Alternatives not ruled out
1. Good apparent agreement could reflect a genuinely accurate grid or a correlated finite-reference error; compare raw MC data and simultaneous marginal uncertainty without treating larger-grid agreement as a certificate.
2. A counterexample on coefficient-zero controls might not affect generic trained rows; preserve that scope and do not invalidate saved outputs without a witness.
3. Cost omissions or different sampling conventions may explain ratios; inventory source categories without transferring another estimator's multipliers.

## Failure modes
Clamping a noise-subtracted discrepancy to zero does not imply zero error. Five clean/noise observations per seed do not establish a distributional guarantee. Shared RNG streams prevent an independent-model chi-square argument. Repeated quadrature levels, source agreement and asymptotic claims are not finite error certificates. No broad negative or source adoption follows from this bounded review.

## Frozen continuity counterexample extension
T55 independently checked the angular-order derivation before implementation. Use the legal all-zero H1T4 model, whose prediction is always first index0. Let a=(r,0,0,1)/sqrt(1+r²), b=(0,r,-1,0)/sqrt(1+r²), so a,b are orthonormal Hopf circles at exact phase gap pi/2. For r<1, coordinate0 is never an extreme rank; antipodal symmetry gives second-largest fraction1/2. For r>1, it is second exactly on two cosine-positive sectors of total length3pi/4−atan(r), so F=(3pi/4−atan(r))/(2pi). The one-sided limits at r=1 are1/2 and1/4. No choice of tie ownership at r=1 repairs this discontinuity. The true global Gaussian accuracy remains1/4.

Freeze r=1/2,2 and r=1±2^-k for k=2,4,8,16; compare unchanged source circle integral with these analytic angular lengths, without random inputs, dense angular sampling or a trained checkpoint. These cases have no identically equal input pair because r!=1. All readout differences are zero, so they do not exercise the mixed-zero mask defect; this isolates the continuity claim. The synthetic result does not refute generic trained convergence or provide a general rate bound. Continuity alone would not prove a specific quadrature rate in any case.

## Frozen full-circle root-defect application
Independent T55 angular derivation checked before execution. H2T4 model Wi=(1,0), Wh=0, Wo has only row1,column0 equal1. Its prediction is1 for x3>0 and0 otherwise, including first-index ties. Freeze a=(2,0,0,1)/sqrt5 and b=(0,2,-1,0)/sqrt5. Correctness holds exactly for atan(1/2)<phi<pi/4: in quadrantI input1 must lie between input3 and input0; quadrantIV input1 is negative; for cos<0 input0 lies below both input3 and a positive sine coordinate. Exact fraction is (pi/4−atan(1/2))/(2pi). Compare original source with analytic value, preserve the result, then apply only the reviewed tile-mask change and compare again. This is one synthetic circle, no random input grid, trained model or whole-base quadrature. Source-returned fraction .125 is a pre-run hypothesis; no conclusion rests on that predicted numerical result.

## Bounded findings
Independent review96522579 reproduces all10 H4T4 report rows, all30 clean/noise/pooled groups and25 truth boxes; it verifies the mixed-zero root-mask defect and scope. Report arithmetic is correct under its stated inherited96 units. The pooled h64 raw ratios are .103897,.019970,.055307,.058806,.025306; clamped noise-subtracted ratios are .037066,0,0,0,0. A zero here is a truncated noisy statistic, not zero bias. Numerically inverted simultaneous95% marginal binomial truth boxes give pooled ratio upper bounds1.20349,.81879,.96762,1.06844,.83566, all with lower0. Thus this particular conservative check is inconclusive for two seeds; it is not evidence those seeds fail. Beta-tail residual is below3.28e-15, but no outward-rounded numerical enclosure is claimed. Saved source/MC/recipe/counter provenance matches run87 in all25 rows. The source uses shared RNG7 for noisy truths, so cross-model independence was not assumed.

The primitive has four mismatches in eight frozen coefficient-zero controls and three strict constant-sign-partition violations. The tile-only local mask correction passes all eight. A full synthetic H2T4 circle subsequently gives exact fraction .05120819117478336, original source .125, corrected source .051208191174783355; the error is7.37918 percentage points on this circle. This is a concrete implementation defect with a verified bounded rescue, not a negative against the Hopf method or any particular saved trained model.

The frozen allzero-H1T4 rank-degenerate example has one-sided correct-fraction limits1/2 and1/4. All ten source-circle checks match the derived angular formulas within1.12e-16. Hence the universal continuity assertion is false without additional assumptions. Generic trained convergence and useful finite quadrature bounds remain open.

`COST_SCOPE.md` inventories unresolved operation sites and distinguishes native96, dense-sampler192 and optimized-sampler160 conventions. No corrected global cost ratio is asserted. Independent review of the final full-circle and cost-inventory phase is pending.

Reproduce with `python root_controls.py`, `python raw_audit.py`, `python continuity_control.py`, `python full_circle_control.py`, each under the <=20CPU review lock. No existing trained numerical result is rerun. Primitive CPU.004016341, raw arithmetic.007995607 then.007994986 after adding provenance guards, continuity.003986728, full circle.004005075. The first arithmetic run and its repeated accounting are preserved in run_accounting.json. Ten continuity circles and two original/rescued synthetic full-circle calls are validation work; no new trained checkpoint, whole-base grid, MC input or dense angular grid is used.
