# T45 walk-free structural gain predictors

## Question
Can quantities computed from weights and model-free rank moments predict the
historical tie-slicing gain beyond the obvious dependence on sequence length?
Claim48d0d9f was pushed before implementation. Historical T2 summary gains have
already been read; this is not a blinded holdout claim. Feature definitions and
statistics below are frozen before any feature/gain associations are computed.

## Method and frozen feature panel
Inventory the complete T2 grid: H in{8,16,32}, T in{3,4,5,6,8,10}, seeds0..4,
trained2^30 sequences. Fetch public checkpoints by their explicit seed, never
best-of-five. Record URL/SHA256 and any missing file; no substitution. Downloads
are read-only public resources, capped4 simultaneous requests and bounded total
bytes. No external source code is executed. Numerical phase remains<=110CPU,
one thread/2GB and shared lock. No walker or stored F values are used to build
features; table gains are joined only after a separate feature file is saved.

Use one exact-in-real-arithmetic conditional moment cell per ordered top-two
position pair, from E30's deterministic numerical moment integration generalized
to each T. At the conditional mean, build the complete frozen-gate input map
for every preactivation and final logit. Two features only, no fitted combination:

1. **Mean Cantelli gate-uncertainty proxy** (hypothesized lower gain as it rises):
   for each frozen preactivation linear form a·X, mean m and variance v under
   its actual rank-cell law, use v/(v+m²), then average over cells/steps/neurons.
   This is a per-form one-sided moment bound when m is nonzero; summing/averaging
   it is a diagnostic feature, not a probability that all gates agree. Identically
   zero forms have proxy0, deterministic nonzero forms0, m0/vpositive gives1.
   Moment integration/roundoff is numerical, not a certified probability bound.
2. **Mean true-boundary gradient alignment** (hypothesized higher gain as it rises):
   final frozen logit-margin gradient g=G_second−G_competitor is compared to
   n=e_max−e_second for the maximum competitor, or e_second−e_competitor for a
   lower competitor. Average signed cosine(g,n) over all competitors/cells.
   Zero gradients contribute0 and are counted. This tests whether local decision
   directions align with the true rank-boundary normals, without computing
   errors integrated along any face. A high alignment need not imply a good
   margin offset or a cheap global gate representation.

Both features are invariant in real arithmetic to hidden-unit permutation and
positive hidden-coordinate changes with compensated weights, and to common
positive output scaling. Test those properties on fixed models H8T3seed0,
H16T6seed0,H32T10seed0 before accepting feature extraction. Charge all controls.

## Frozen statistical comparisons
After committing/saving features, join the90 historical T2 rows by H,T,seed.
Report each feature against log Sobol gain and log iid-face gain, with no feature
selection. Primary associations: global Spearman; pooled within-T rank correlation;
pooled within-(H,T) rank correlation. Re-rank within each fixed group and center
the ranks before pooling. Global association alone can be an H/T confound.
Use2000 fixed-seed45001 model bootstrap resamples within the18(H,T) cells,
recomputing ranks each time. These CIs condition on historical gain estimates;
they do not include rule/truth estimation error or establish causal prediction.
Compare the T-only direction explicitly. Two predeclared sensitivity subsets:
excludeT3 (whose historical metric often uses variance-only gain), and historical
accuracy>.97 (to compare the earlier tail-share report). Accuracy is an audit
filter, never a predictor input. No fitted regression, tuned cutoff or best-model
selection. Missing/nonpositive gains remain explicit; do not silently discard.

## Alternatives before evaluation
1. A seemingly strong predictor merely tracks T: check both within-T and within-
   (H,T) ranks, not just global correlation.
2. Canonical mean gates are unrepresentative (T28 already shows closure bias):
   report both distinct features and do not mistake correlation for an estimator.
3. Training failure/low accuracy creates a different mechanism: preserve the
   full-grid result and frozen high-accuracy sensitivity rather than selecting it.
4. Historical cost/gain uncertainty and T3 truth limits may dominate: retain
   source hashes/metric flags and the no-T3 comparison; no portable-cost claim.

## Status
Claimed; protocol frozen before feature computation/associations. Claude remains
DRI. Only this folder is edited; independently reviewed controls/results required.

## Results
Initial full90 feature panel saved before gain join. Sobol pooled within-(H,T)
rho is−.07778 for gate uncertainty (CI[−.3204,.1722]) and−.01111 for boundary
alignment (CI[−.2485,.2337]). Both confidence intervals include0. Within-T
alignment rho.3262 disappears after also controlling H. Global−T baseline is
.8912. All subsets/features/metrics are retained in `associations.json`;
no favorable feature is selected. Features1.97344CPU, association1.23403CPU;
all90 explicit public checkpoints total404290bytes. No walker/F calls.
This diagnostic cannot pass a ladder rung or certify an estimator win.

## Frozen distinct boundary-point rescue (after initial panel, before new run)
The interior canonical mean may miss the geometry where tie-slicing integrates.
For each fixed ordered top-two chamber and each competitor, project its mean
onto the corresponding true rank facet: y=mean−n*(n·mean)/2, with the same
inward norm-sqrt2 normal n as above. Evaluate the actual gate/Jacobian map at
y+2^-20*||y||*n, a fixed small displacement inside the chamber; do not choose
the side or epsilon from feature/gain outcomes. Define its signed cosine between
network logit-margin gradient and n. Preserve zero gradients as0/counts.
Freeze only two summaries of this new boundary representation: mean cosine and
10th-percentile cosine over all cells/competitors. The latter tests whether a
fixed low-alignment tail, rather than the average, matters. Neither summary is
selected as a winner after the result.

Use all90 unchanged models, the same source moment table and the SAME association
script, subset definitions, bootstrap recipe and metric flags. Save boundary
features before their gain join. Repeat positive hidden-coordinate/permutation/
output-scaling checks on the same3 models. This is an explicitly post-initial-
result mechanistic rescue, not a fresh blinded validation. <=20CPU review lane
if feasible, else<=110CPU primary lock; no new face walks, widths, noise/model
sweep, or fitted feature coefficients. Failure remains scoped to these finite
representations, not all walk-free predictors.

## Compute accounting
Charge all checkpoint loads, numerical rank-moment integration, canonical gate
and Jacobian propagation, margin gradients, covariance products, invariance
controls, table join and bootstrap. Record bytes/URLs/hashes separately from
CPU. Feature generation must not read gain/F columns; association is a separate
stage. Elapsed CPU is not a portable scalar-FPE certificate.

## Claims
[speculative] Rank-boundary alignment and gate uncertainty may predict variation
in historical tie-slicing gain beyond T. [inferred] They do not use face-walk F.
No correlation, estimator advantage or causal explanation is claimed in advance.

## Alternatives not ruled out
Different structural features, exact support/cell summaries, mixtures of gates,
other task families and genuinely held-out predictors. A failed two-feature
panel would not establish that walk-free prediction is impossible.

## Failure modes
Historical gains use legacy walker/cost conventions and finite truth/rule samples.
Canonical means omit nonlinear histories. Bootstrap on only5seeds/cell can be
unstable; conditional CIs omit gain-estimation uncertainty. Finite positive-
coordinate invariance checks are controls, not exact symbolic certification.
