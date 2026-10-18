# T133 fixed implementation evidence and scaling

`check_final3_review.py` independently recomputes all nine raw/corrected
ratios and observed-exact gaps using rational arithmetic. The stated
nominal sample-equivalent denominator is2000/3. Saved Bonferroni
Clopper-Pearson endpoints satisfy independent incomplete-beta tail
checks within8e-15. Over each simultaneous95% truth interval the ratio
n(e-p)^2/[p(1-p)] is minimized at the nearest endpoint unless e lies
inside. Its minimum across the nine cases is1.09707648094. This supports
the fixed panel's failure under its declared nominal units, without
certifying source operation completeness, noisy-model population behavior
or a newly measured truth. Correlated truth samples do not invalidate the
Bonferroni union bound. Error-gap calculations using numerically unreliable
'exact' source values remain descriptive; T131 independently diagnoses that
primitive on unchanged geometry.

T3 mean466 cells divided by actual235.6 regions is1.97792869, whereas
dividing by sampled lower bound165.4 gives only upper bound2.81741233.
Increasing cell/lower-region ratios do not prove increasing true ratios:
true-region counts235.6,6000,200000 satisfy all reported lower bounds and
give decreasing ratios. Mixed means and seed-specific counts are distinct
comparisons; neither justifies a constant growth-factor extrapolation.

The unchanged chart enumeration has4*T! full-dimensional open charts for
T>=3: all permutations, either x0 sign, and either top-gap comparison.
Strict gaps1,2 or2,1 give either comparison; a common translation gives
either x0 sign without altering gaps. Rank-order cone lineality is exactly
the all-ones direction; adding sign(x0) eliminates nonzero lines. A
full-dimensional pointed cone needs at leastT extreme rays.

The source charges H for every first gate and H*active(g1) for the first
recurrence. Summing signs givesT!*(4H+2H*r), with r nonzero Wi entries.
H16,T10,r16 yields2,090,188,800 nominal operations before deeper work.
Even caching repeated first-recurrence maps leaves, in recipe108/exact,
the first _margin_split matrix evaluation: at least4*T!*T*H*(T+1)
=25,546,752,000 nominal operations. These are complete-processing lower
bounds for unchanged source, not actual completed T10 runs. Free declared
model-independent preprocessing is excluded. Redesigned predicates,
shared cross-chart computations, sparse kernels and coarser top-two
families remain open; no estimator-family impossibility is implied.
