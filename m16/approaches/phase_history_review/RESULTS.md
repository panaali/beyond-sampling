# Observed-history decomposition of the E29 phase slopes

All ten seed/mask selections reproduce E29's counts exactly. The six supported
phase slopes reproduce to1.65e−12 degrees/step; the other four selections still
have fewer than three supported ages and are not assigned a slope. The complete
modal reconstruction has maximum absolute residual8.994e−13 and maximum
relative residual1.710e−11 (small amplitudes amplify relative error).

This checks an exact identity using observed hidden states and gates, not a
new explanation that predicts those gates without running the network.

| Final mask | Seed | Eigenangle | Initial phase slope | Base | Base+gate | Base+forcing | Full/observed |
|---|---:|---:|---:|---:|---:|---:|---:|
|64975|82026|42.748|−26.833|15.914|44.339|64.310|41.467|
|64975|82027|42.748|−27.087|15.660|44.386|64.142|41.710|
|63951|82026|46.571|−12.042|34.530|45.805|26.755|38.367|
|63951|82027|46.571|−10.320|36.251|45.183|26.457|38.859|
|65007|82026|42.838|9.449|−6.092|49.713|39.406|47.035|
|65007|82027|42.838|9.051|−6.489|49.557|39.283|47.020|

All entries are degrees per maximum age, using the original unweighted linear
regression of unwrapped circular bin means. They are **not additive causal
effects**. Adding complex vectors and taking phase is nonlinear; removing a
term does not recompute gates or create a realizable network trajectory.

For mask63951, adding the observed gate correction brings the base phase
slope to45.2–45.8, while then including input forcing brings it to38.4–38.9.
For64975 and65007, forcing also lowers the base+gate slope. But these finite
cohorts do not support the statement that forcing alone explains every
eigenangle discrepancy: initial-phase/selection and operator-history effects
are substantial, and base+forcing alone is not a good reconstruction.

## Initial phase and unwrap ambiguity

The initial post-maximum phase depends on the maximum age within the final-mask
selection. Even the unforced fixed-operator base is therefore not expected to
have slope equal to the eigenangle. For64975/63951, the independently unwrapped
initial slope plus eigenangle equals the reported base slope to rounding.

For65007, supported ages are1,3,4,7,8,9. Multiplying circular bin means by
exp(i*age*theta) is exact, but unwrapping separately need not commute with that
operation across gaps. At seed82026, the initial mean moves from−170.1deg at
age4 to−19.2deg at age7; multiplying by the eigenmode produces base means1.2deg
and−79.3deg. The two least-change unwraps choose different branches. Thus
9.449+42.838 is not the independently unwrapped−6.092. Both original outputs
are preserved. A different prior on the missing-age phase could choose a
different slope; none is selected after seeing these results.

Minimum supported-bin concentrations for observed phase are.873–.890 at64975,
.949–.958 at63951 and.586–.653 at65007. Initial/base concentrations fall to
.259–.324 at65007. No low-concentration bin was silently removed beyond the
predeclared original count rule.

## Relative contribution magnitudes and matched histories

For the three supported masks, the median base amplitude divided by final
amplitude ranges.038–.107; total gate residual1.451–1.837; forcing.326–1.014.
Large gate/forcing terms partly cancel. The median support-leakage component
alone is1.191–1.332. It cannot be dropped when using the zero-extended
principal-block left eigenvector. The selected-output-gate component has
median.235–.779. These are ratios per sequence, not fractions of explained
variance or additive shares of accuracy.

Across both cohorts,14970 distinct post-maximum gate histories were observed.
For64975,68/71 histories with at least20 examples cover4448/4574 sequences;
for63951,22/18 histories cover1072/986; for65007,12/12 cover460/479. Exact
history-conditioned phase correction means/concentrations and component
ratios are retained in `results.json`. Most histories are too sparse for a
stable separate empirical description. Equal history includes equal age;
these within-history summaries do not independently estimate an across-age
phase slope or prove sufficiency of a gate-only state.

## Algebra and representation caveat

The left row is taken from the inverse of the right-eigenvector matrix, without
complex conjugation, selecting the positive-imaginary dominant complex mode.
With l supported on D*, l(D*WhD*)=lambda*l. The exact decomposition is

    g_t = l(D_t−D*)Wh h_(t−1) + lD*Wh(I−D*)h_(t−1),
    f_t = lD_tWi x_t.

Telescoping z_t=lambda*z_(t−1)+g_t+f_t yields the retained reconstruction.
Independent T55 review checked the operator, eigenvector convention, finite
convolution and noncausal interpretation before the numerical result.

The decomposition is representation-dependent. Since lambda is nonzero, a
left eigenvector of D*Wh can instead be extended by

    l_full[inactive] = l_active Wh[active,inactive]/lambda.

It agrees with the original observable on final states supported in D*, but
differs on earlier states and reorganizes the initial/gate/forcing terms.
Thus the measured support-leakage component is necessary for **this** E29
observable; it is not an invariant causal mechanism or impossibility proof.
No alternate full-eigenrow numerical scan was run in this bounded phase.

## Scope, alternatives and compute

The primary hypotheses all remain relevant: input forcing and gate changes
matter, and final-mask conditioning already changes the starting phase. The
exact identity closes the numerical discrepancy once actual histories are
supplied. It does not derive their distribution from weights, estimate accuracy,
show that one eigenmode causally stores the maximum, or establish a speedup.

Alternatives retained: a full eigenrow representation, explicit missing-age
phase priors, stronger conditioning that predicts histories cheaply, other
training seeds and causal interventions with recomputed gates. None is rejected
because this finite accounting does not implement it. No new Gaussian cohort
was selected: the same two E29 streams were regenerated, including discarded
pilot draws, and no parameters were fitted beyond the original phase summaries.

Total CPU including imports/loading/setup1.41881sec, peak RSS328528KiB.
Counts:1,000,000 Gaussian variates,80,000 diagnostic sequences/800,000 hidden
steps;5 eigendecompositions/inverses;45,717 selected sequence-mask pairs;
457,170 recomputed recurrent vectors;1,874,397 complex projected vectors.
Also charged in CPU: gate masks, component recurrences, summary regressions,
quantiles, history dictionaries, reference reads and controls. JSON serialization
follows the timing snapshot. This is a diagnostic cost inventory, not an
instrumented scalar FPE count or benchmark estimator cost.
