# Source inventory, before any saved numerical contraction

Frozen stage1 commit60967f89, preregistration9dc776b0. All claims below concern
read source, not newly executed author code. The primary eight-model finite
coverage result is not yet independently contracted here.

1. **Primary metric is a finite fixed-rule metric.** tail.py selects ceil(.05*2n)
   largest stored g² values from each fixed scrambled Sobol rule, then labels only
   those selected points. analyse.py selects top-L label sums on rule700's tail
   and computes their share of rule701's tail sum. The two finite rules are
   distinct; there is no numerical integration-error or population-coverage CI.
   Only points with a truthy kind enter analysis. Saved contraction must report
   missing-kind count/mass and all raw tail/summary totals, not silently discard.

2. **The n-check is not a new half-sized face rule.** analyse.py draws a fresh
   Bernoulli half-subset of the already-selected ruleA tail separately for every
   scheme and L. It does not evaluate2^13 faces or recompute their top5% tail.
   Pooling A+B similarly pools their separately selected tails, not necessarily
   the top5% of their combined face values. This is useful conditional thinning,
   but is not a literal sample-size convergence experiment. Random choices are
   not archived, so saved arithmetic alone cannot reconstruct their assignments.

3. **The null does not preserve the actual label-size profile.** The preregistered
   control says labels are randomly reassigned with the same size profile.
   analyse.py instead assigns each retained point an independent uniform integer
   in range(K), where K is the observed A+B mechanism-label count. This preserves
   the number of possible IDs, not the observed occupancy counts or profile.
   The singleton null is explicitly in-sample. Neither null substitutes for
   a population confidence statement about the primary out-of-sample metric.

4. **Within-label heaviness is measured inside the global tail only.**
   within_tail(A,f) receives only A's top5% points. It computes the top5% share
   among those tail points of each sufficiently populated label, and compares
   with the pooled tail points. It does not estimate the complete conditional
   second moment/tail fraction of that label over all faces. Therefore this
   statistic alone cannot prove that a dedicated full-family rule fails.

5. **Feature semantics need literal wording.** The README calls the in-band
   flips a set. tail.py stores the first three ordered break events; analyse.py
   uses the resulting tuple directly. Repeated neuron/step events or a different
   order would remain distinct labels. Round(hi,12) collapses coincident break
   keys, and only one minimizing gate is retained per boundary. The saved audit
   can count duplicates/order differences in the retained lists, but cannot
   recover discarded later events or certify a complete gate-event set.

6. **Face-sign and actual-input blind regions differ.** blind_split.py groups
   fibres by their tied face value u<=0, then assigns the whole fibre's g² to that
   group. This is a legitimate face stratum, not the exact input event K<=1
   throughout every segment. For kind12, y=(1,1,-4) has u>0 and
   tau=5sqrt(2)/3; t=3sqrt(2)/2 gives x=(5/2,-1/2,-4), K=1. For kind23,
   y=(4,-1,-1,-5) has u<0 and the same tau; shifting the tied pair by that t
   gives x=(4,1/2,-5/2,-5), K=2. Both t are strictly inside their windows.
   Thus a '60% blind face second-moment share' is not an input-event mass/error
   statement. T226 independently reports T223 clips fibres by the actual event;
   no defect is transferred from this naming distinction to that method.

7. **Walk agreement is finite numerical consistency, not exact certification.**
   core.py uses a[-8,8]160001-point numerical CDF table, inverse interpolation,
   tau cap8.5, eps1e-10, at most2000 walk iterations and a scaled1e-12 zero-margin
   tie condition. scalar walk_one uses the same eps, at most4000 cells and
   numerical roots/CDFs. Neither returns an explicit exhaustion/cap flag here.
   Source tail.py saves Fcheck, but not the full scalar intervals, vector walk
   states, inputs, or a complete all-face g array. Fcheck agreement can be
   audited as a saved diagnostic, not independently recomputed without new
   author/model work. No occurrence of a cap or trained error is asserted.

8. **Prefix counts are floating LP gate-history counts.** prefix_blind.py uses
   normalized floating coefficients, HiGHS LPs and TOL1e-9. Failed LP statuses
   become NaN; the caller does not explicitly reject them before split logic.
   No saved exact feasibility witnesses or raw histories accompany its summary.
   Reproducing two other floating counts does not certify exact cardinalities.
   'sec' is wall-clock elapsed time. Count-growth extrapolation is conditional;
   no universal future growth lower bound follows from a few completed prefixes.
   T219 independent exact counting remains separate; this review will not rerun it.

9. **Negative scope.** A low finite-rule coverage score can support rejection of
   the frozen label/selection/threshold test, once raw contraction is checked.
   'The tail is a continuum' and exclusion of all few-family decompositions
   require more than these finite records. Coarse labels, omitted features,
   different rank-aware conditioning and dedicated rules remain alternatives.

10. **Runtime binding is incomplete.** The saved model JSON identifies names/T/n,
    but carries no model-weight bytes, library versions or executed-source hash.
    The frozen driver prints `err_est` and writes `err_note`, whereas the retained
    run log prints `err` and the JSON omits `err_note`. This establishes a visible
    logging/serialization version difference, not a model-result defect. The
    blind launcher requests kmax10 and its log ends `Terminated` after completed
    k4; the final JSON explicitly reduces kmax to4 and notes k5 stopped. Preserve
    that incomplete continuation and unknown extra runtime, rather than calling
    the whole original launch a completed k4-only experiment.

## Proposed next saved-only phase (not execution clearance)

Bind all8 models/two rules; independently contract stored tail masses, deterministic
label counts/selections/cross-copy coverage and within-tail statistics; report
all missing-label and duplicate-retained-flip diagnostics. Preserve half/null
outputs as source-qualified summaries when their random assignments are absent.
Contract blind-prefix summary ratios and completeness flags without new LPs.
No model/face/RNG generation, fits, mass integrations, new label rules or author
imports. Freeze exact formulas, tie ownership, tolerance/rounding interpretation,
all output fields and20CPU/2GiB resource cap before independent gate and one run.
