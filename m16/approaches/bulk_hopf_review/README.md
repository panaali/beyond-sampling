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
