# What conditional correct/error coverage can establish

This written contract supports owner T239; it does not alter the owner's
experiment or choose support/disproof thresholds after observing results.
All formulas below refer to a fixed model, input law and declared partition.
No new statistics or curves are calculated here.

## 1. Specify the counted object and denominator

Let C_i be a finite disjoint partition up to null boundaries, with mass m_i.
Let E be model error under the declared first-index prediction and task tie
rules. Define e_i=P(C_i intersect E), c_i=m_i-e_i, P_E=sum e_i and P_C=sum c_i.
For either class A with positive P_A, sort its contributions a_i descending
with a deterministic tie rule. Define

    N_A(q)=min{k: sum of the k largest a_i >= q P_A}, 0<q<=1.

Report P_A, the full curve or preselected q, N_A, positive-contribution count,
fraction of all declared cells, and absolute omitted mass P_A*(1-q). Zero-class
mass makes conditional coverage undefined, not zero complexity. Missing cells
or unknown contributions require an explicit unresolved category, not zero fill.

The same activation leaf may contribute to both classes. Such N_A describes
mass coverage by that leaf partition. Calling it a count of uniformly correct
or wrong decision regions instead requires a verified output-label refinement
with unique parent/child identities and no overlap/omission. Comparisons across
activation leaves, clipped output pieces, simplices, gate histories and grouped
affine maps must keep separate denominators. Center labels are not uniformity.

Equal conditional targets do not equal equal global error budgets: at q=.999,
the omitted masses are .001*P_C and .001*P_E. Report both and optionally compare
at a predeclared common absolute omitted-mass tolerance as a separate question.
Do not infer complexity solely from cells divided by class probability: a rare
event makes that ratio large even before any geometric explanation is supplied.

## 2. Partition dependence prevents an intrinsic-complexity conclusion

A constant-label cell of mass p can be arbitrarily subdivided into n equal-mass
pieces without changing the function. Its q-coverage count becomes ceil(q*n).
Thus large cell count, count per probability, entropy or coverage count is not
by itself a lower bound on representation complexity or the work needed by every
algorithm. The experiment concerns a specified enumeration/partition rule.

Freeze start charts, input-rank refinement, gate split policy, output clipping,
triangulation, ordering, tolerances and grouping. Preserve same function under
different start-chart diagnostics if already available; do not generate new
partitions merely to obtain a favorable comparison. A coarser grouping requires
accounting for its unions/domains and construction cost, not just duplicate
coefficient deletion. This preserves the meaningful claim that one fixed engine
may require more pieces to cover error mass than correct mass.

Finite fragmentation cannot establish that the error function is random-like,
that learning exhausted the architecture, or that region aggregation is
impossible. State which operational statistic is meant by 'simple'/'shattered'.
Preregister a plausible falsifier as well as a support criterion, keep mixed
outcomes and separate these descriptive findings from the owner's intuition.

## 3. Correctness and numerical masses need separate verification

Retain original returned m_i,e_i and all negative, nonfinite, e_i>m_i and tiny
remainder cases. Do not silently clip them or use a numerical kind tag as an
exact correctness certificate. If mass intervals are available, combine their
uncertainty with coverage; aggregate normalization alone does not validate
individual masses or cancellation in c_i=m_i-e_i. Bounds for individual pieces
must also cover their partition/provenance assumptions. A confidence interval
from MC is not a deterministic quadrature enclosure. If the contribution bounds
are probabilistic, joint coverage must account for every interval used and any
data-dependent selection; pointwise intervals do not automatically survive sorting.

For verified nonnegative class contributions a_i in [l_i,u_i], a fixed selected
set S has conditional coverage at least

    sum_S l_i / (sum_S l_i + sum_notS u_i),

when the denominator is positive and the partition is complete. Any unresolved
class mass upper bound must be added to the denominator's complement. The
analogous upper ratio uses sum_S u_i and sum_notS l_i (and unresolved lower
mass). A zero denominator with possible zero class mass stays undefined.
These are conservative bounds for a FIXED S, not proof that a floating sorted
selection is optimal. They give a certified upper bound on N_A when a particular
S reaches q. If total class mass has a positive lower bound L and the sum of
the k largest u_i is less than qL, then no k represented cells suffice;
unknown unrepresented cells prevent that lower-bound conclusion unless included.

Conditional normalization can amplify integration error on a small error set.
For growth ratios, carry ambiguity in N_A through the ratio; do not compare
point counts as if threshold crossings were certain. If no rigorous intervals
exist, explicitly call the curves numerical diagnostics. No new integrations
are authorized by this contract.

## 4. Tie-distance and blind strata must be probabilities of intersections

Choose raw or normalized gaps and define which order statistics they use.
Raw gap thresholds are not invariant along a ray of a homogeneous cone, and
normalized gaps generally vary across directions within it. A center, one ray
or average representative cannot label an entire cell as 'near tie'. Required
quantities are P(C_i intersect A intersect gap-bin), or certified homogeneous
membership with explicit unresolved mass. Actual intersections may require
extra geometry/integration; it is not free postprocessing of ray COUNTS.

Predefine joint/disjoint bins for first-second gap, second-third gap and blind
event. Their overlap must not be counted twice; report both class-conditional
and joint probabilities. The blind probability (T+1)/2^T uses the actual input
length under independent symmetric signs, not 11/1024 for every smaller model.
Continuous-input rank ties are null in the ideal law, but floating task/readout
ties and coefficient-identical logits retain explicit first-index ownership.

Across T, order-statistic distributions change. A fixed raw gap or quantile gap
answers different questions; neither should be silently substituted after
seeing coverage. Prefix regions are not full target decision regions unless
future inputs and labels have been accounted for.

## 5. Comparisons, provenance and finite inference

Pair trained/random/scrambled models with the same engine, start/stop rules,
quadrature/cost convention, partition unit and coverage targets. Keep all seeds,
partial failures and mass residuals. Global positive normalization can preserve
prefix-history counts while changing Gaussian masses (T236); row normalization,
relabeling and sign shuffling are different interventions. An exact neuron
permutation with corresponding incoming/outgoing transformations preserves the
function; arbitrary row permutation alone generally does not.

Correct/error sets themselves change with model accuracy. Conditional coverage
differences can reflect different error locations, error rarity, training or
partition geometry; counts alone do not identify a unique cause. Compare
absolute probabilities and the specified strata as well. A handful of models
does not provide a universal architecture claim or asymptotic law. Growth from
T4 to T5 and extrapolation to T10 must retain measured versus inferred status;
different-T zoo checkpoints need not be the target network's prefixes.

Freeze source/checkpoint/runtime hashes, mode, chart identities, mass-kernel
settings, completeness/stopping and original data schema. Preserve selected
threshold and support/disproof rules before new outputs. Reusing already viewed
archives is a descriptive follow-up, not a fresh holdout by renaming it. If MC
or resampling is later used, specify the sampling unit, its dependence and
selection procedure; individual cells of one deterministic network are not iid
training-seed replicates. This phase has no numerical execution authority.
