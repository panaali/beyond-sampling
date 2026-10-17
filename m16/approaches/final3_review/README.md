# T133 independent E45 accounting and scope audit

## Question
Which E45 conclusions follow from saved measurements, which need a cost/geometry qualification, and which extrapolations can be replaced by a scoped proof? Claim85e768b preceded implementation. T131 owns clipping/mass attribution; this review does not diagnose or repair its source.

## Method
Pin E45 raw outputs, summary/source scripts, estT, its MC routine, T3 comparator data and the nominal leaderboard sampling formula at85e768b. Recompute every saved ratio, error, comparison and aggregate from raw numbers without importing author summary code or executing neural inputs. Preserve current units; separately state any lack of a complete estimator/sampler common-unit audit.

Frozen uncertainty check: all nine T4 cases, M=4,000,000 Bernoulli truth draws each, conditional on these fixed models. Derive simultaneous95% Bonferroni Clopper-Pearson intervals (alpha .05/9 per model, no independence across models needed) and the minimum/maximum true squared-bias/sampling-variance ratio over each interval. Same seed99 across models creates correlation; do not treat nine truths as independent observations. Separate raw ratios from the author's finite-truth-variance subtraction. No new MC, noise or model selection.

Frozen scaling check: show both five-seed means and the matched seed0 sequence; preserve the T4-total versus T5-nonmass distinction and compare nonmass to nonmass only when available. Sampled distinct gate counts are lower bounds; ratios with these denominators are upper bounds, not actual region ratios. Construct a numerical counterexample to the logical inference of increasing true ratios, consistent with every reported lower bound, without claiming those hypothetical counts are real.

Source-derived alternative: halves(T) has4*T! nonempty open strict-order/sign/top-gap charts for T>=3. Construct exact integer interior witnesses for every chart at T3/4/5 using gaps1,2 versus2,1 and common shifts fixing sign(x0). Count the unchanged declared first-gate and first-recurrence work, excluding all deeper work and model-free chart setup. With H hidden units and r nonzero input weights, the completion lower bound is T!*(4H+2Hr). At H16,T10,r16 it is2,090,188,800. Verify the target input-weight support from the installed public model without running inputs; otherwise retain r explicitly. This applies to completion of the unchanged factorial-chart implementation, not a successfully executed T10 estimator (its masses are unsupported) or an entire method family. T110 top-two reparameterization can remove the factorial and remains an alternative.

Competing explanations: a summary arithmetic/transcription issue; finite MC uncertainty explains the apparent failures; missing common cost units change comparisons; sampled region undercount drives apparent super-region growth; or the fixed chart representation has an independent combinatorial cost barrier. The listed raw arithmetic/statistical/source checks separate these without duplicating T131 geometry work.

## Status
Running bounded raw/source checks. One thread,2GB, review lock<=20CPU; independent T55 verification required. No new neural sampling or geometry enumeration.

## Results
Pending. Shared E45 tables and claims are not edited here; DRI receives explicit proposed qualifications.

## Compute accounting
This audit reads saved outputs and performs small arithmetic/statistics/integer witness checks. Public weights are read only for support; zero model forward passes. Existing reported ops are nominal source counters, not newly certified hardware operations. Every new audit runtime and source hash is recorded.

## Claims
[inferred] Sample lower bounds alone cannot determine ratios to true region counts or their growth. [inferred, review pending] The unchanged factorial-chart prefix already exceeds1e9 nominal charged operations at the target whenr16. No family-wide impossibility.

## Alternatives not ruled out
T131 mass/geometry repair; certified mass integration; coarser top-two or class-region parameterizations; symbolic/repeated-time structure; alternative gate certificates; matched units and faster executed kernels. Recipe-minus-biased-exact differences are not known repaired errors without a shared-error proof.

## Failure modes
Treating Monte Carlo truth as exact, assuming independence of reused input streams, comparing seed averages to one seed without disclosure, equating lower bounds with exact region counts, propagating two observed growth factors as a theorem, or calling a fixed implementation's cost barrier a method-family negative.

## Further scoped source lower bound and cheap rescue caveat
The two first recurrent maps are identical across many charts and could be memoized by sign(x0), so the first2.09B bound alone does not exclude that implementation change. Independently, each nonempty initial cone is full-dimensional and pointed, hence any correct extreme-ray representation has at leastT rays. Its first _margin_split always charges len(c.R)*H*(T+1) for ray/gate dot products, before its gate decisions. This gives4*T!*T*H*(T+1) additional nominal charged operations,25.547B atH16,T10, even if the two recurrent maps are cached. It still applies only to the unchanged ray-testing/chart architecture; sparse predicate evaluation or a different chart representation can evade it. No general lower bound on accuracy estimation is claimed.
