# T67: fixed target rank and two-dimensional hidden transport

## Question
Does conditioning on the second-largest target eliminate the dominant target-coordinate interpolation error while preserving honest cost and boundary treatment?

## Method
For iid continuous inputs, each target index j is second largest with probability1/T. Conditional on that event, q=F(x_j) has Beta(T-1,2) density. The unique larger index is uniform among the T-1 other positions. Until it occurs, its probability at the next non-target position is1/m, where m is the number of remaining non-target positions. The other inputs are independent conditional draws below or above the fixed target. Target insertion does not decrease m.

We use the normal law truncated symmetrically to[-L,L], with F its normalized CDF. The rank statements remain exact. Original accuracy lies between Pbox*A_trunc and Pbox*A_trunc+1-Pbox, where Pbox=(1-2*Phi(-L))^T, conditional on exact evaluation of A_trunc. DefaultL6 gives omitted mass below2e-8 for T<=10. Numerical quadrature/interpolation errors are separate and uncertified.

The final implementation splits outer target integration at q=.5 and uses positive Gauss-Legendre weights times the Beta density. The original Gauss-Jacobi failure is preserved below. Backward transport uses only two hidden coordinates plus the binary remaining-above state. Each time has its own finite box B_(t+1)=Wh_positive*B_t+L*abs(Wi), startingB0=0, ensuring all reachable transitions are in the next box even if Wh_positive expands. Conditional transitions support both quantile quadrature and guarded analytic Gaussian integration of the bilinear interpolant; the latter caches only charged within-estimate geometry and sparse kernels.

Final fresh scalar-input integration explicitly splits ReLU gate intervals and intersects affine output margins; identically equal margins obey first-index argmax ties. When the target is last, that fixed target step is included in the scalar interval construction. No discontinuous argmax labels are silently interpolated as terminal data.

## Status
Bounded T67 implementation/validation phases complete; root/DRI acceptance and future scope remain separate. Root pushed claim42f60e1 and expanded scope inb7209a1. T55 supplies independent derivation/stability review and T50 read-only mathematical review; this folder owns the distinct engine. No global-challenge success claim.

## Results
First cycle (checks.py, checks_results.json): six constant-readout controls on T3,T4,T7 have maximum error2.78e-17; six scalar terminal partitions have maximum defect2.22e-16, with duplicate logits correctly assigned to the first index. CPU.380seconds. The deterministic toy Wi=(1,0),Wh=0 selects the last label iff final input is positive, otherwise label0. Exact accuracy is1/T+1/[2(T-1)]-(T+1)/[(T-1)*2^T], unchanged by symmetric truncation. Global Gauss-Jacobi fails this sharp-boundary control: T4 errors at outer orders8/16/32 are-.01334/-.00773/-.00420; T7 errors+.002583/+.000906/+.000274. T3 happens to be exact and would have hidden the problem alone.

Before any trained run, freeze this structural repair: split target quantiles at q=.5 (the deterministic zero-input boundary under the symmetric law), using positive Gauss-Legendre on each half with explicit Beta density. Report actual2*n target nodes. This split depends on the known zero/ReLU boundary, not observed accuracy. Repeat all controls; preserve the failed unsplit results. Then bounded H2T7 LOCALPILOT tests, initial grid9,outer8 per half,input8 and one-axis refinements grid17/outer16/input16, without fitting settings to heldout truth.

Split-boundary checks (checks.py --target-rule split_sign, checks_split_results.json) pass: all nine sign-toy cases have absolute error<=1.11e-16; constants<=8.33e-17; terminal partitions<=2.22e-16. CPU.909seconds. T55 independently reviewed the engine's chronology, remaining-slot count, conditional recurrence, scalar margins and normalization, finding no algebraic discrepancy. Trained convergence remains untested at this checkpoint.

First trained cycle (pilot.py --seconds18, pilot_results.json), H2T7 LOCALPILOT checkpoint SHA09ac1bc640e3b15641a12606d91a4c28e699aa47fd14aff202bac52bfa0444e0:

| hidden nodes per axis | target nodes per half | input nodes | estimate | CPU |
|---:|---:|---:|---:|---:|
|9|8|8|.21248113|.308|
|17|8|8|.22475549|.811|
|9|16|8|.21254897|.543|
|9|8|16|.21243939|.322|

Existing heldout estimate.23706055 has SE.00234936; it is not a precise reference. No new samples were generated. The input-tail bound is1.38123e-8, but hidden interpolation is plainly unresolved: doubling hidden resolution changes the estimate by.01227, whereas target/input refinements change it by less than.00007. This is a negative result for the coarse uniform hidden grid. Finite stage bounds grow to approximately(25.24,28.35), making those uniform cells large.

Next bounded cycle is frozen before measurements: preserve the proved boxes but use fixed quadratic hidden coordinates B_t*(i/(n-1))^2 to resolve near-origin states. Compare hidden9/17/33 at target8/input8, plus uniform33 as a control. This changes mesh spacing only and uses no sampled states or truth-fitted allocation. Report all four outcomes and convergence rather than selecting whichever is nearest heldout accuracy.

Quadratic mesh cycle (mesh_cycle.py, mesh_results.json): h9=.22578459(.304CPU),h17=.23215131(.908CPU),h33=.23427521(3.183CPU). Uniformh33=.23162007(2.788CPU). Concentrating near zero improves this convergence sequence, but h17-to-h33 still moves.002124 and the coarse heldout reference cannot resolve small bias. No out-of-domain query occurred. Next prespecified checks: at quadratich33 separately double target and input rules; compare the same engine against the known H2T4 cone reference at h17/h33; no extra Monte Carlo.

Refinement checks (refine.py t7/t4): at T7 quadratich33, target doubling gives.23430653(6.580CPU), input doubling gives.23417594(3.743CPU). Their changes+.0000313 and-.0000993 are much smaller than the preceding hidden refinement, but remain nonzero. The same unmodified engine on H2T4 gives.30822479 at h17(.556CPU) and.30775292 at h33(1.874CPU), versus the independently validated cone reference.3077307430. Errors decrease from.0004941 to.0000222. The last small error may reflect cancellation between interpolation and quadrature errors, so it is not a convergence certificate.

Next frozen bounded check: T7 quadratich65, keeping target8/input8, to measure the remaining spatial change. This is the only next trained run; retain its cost and result regardless of whether it approaches heldout accuracy.

Fine check (fine_mesh.py, fine_mesh_results.json): quadratich65 gives.23477246 in12.357CPU, a+.00049725 change fromh33. Spatial changes shrink, but this alone neither bounds the remaining bias nor resolves the existing heldout uncertainty. There were18,523,184 interpolation queries and270,400 analytic scalar-terminal queries; no new samples.

Comparison with the existing three-coordinate T65 transport is descriptive, not a matched-compute MSE claim: its saved h5/b9/input16 result is.22550079 with24.1465CPU terminal construction plus.0401CPU transport (small_h2_78/anisotropic_results.json). T67h65 uses12.357CPU including all construction and gives.23477246. Both use the same checkpoint; the old result is from an earlier run, and the shared heldout reference remains coarse. The new method is promising at lower measured CPU, but its total numerical error and sampling competitiveness remain unresolved.

The T65 owner subsequently supplied the stronger final h9/b17/input16 result.23193864 (interaction_results.json). That final cycle records38.5489CPU excluding imports and reuses earlier terminal caches whose construction is additional; it must not be represented as a free warm-cache estimate. This newer result improves the old baseline and is retained here to avoid comparing only against its weaker setting.

## Polynomial transition phase: frozen protocol

For fixed h, the next hidden state is ReLU(Wh*h+Wi*x). Split x at every ReLU root, next-grid coordinate crossing, and the target u. Within an interval the hidden map is affine and bilinear grid continuation is a polynomial of degree<=2 in x. Its Gaussian integral uses M0=Phi(r)-Phi(l), M1=phi(l)-phi(r), M2=M0+l*phi(l)-r*phi(r). This eliminates conditional-input quadrature error for the supplied interpolant, not hidden interpolation or outer-target error.

Use centered coefficients derived from endpoint/midpoint continuation values to avoid large global polynomial coefficients. Stable tail CDF differences supply M0. Narrow intervals, invalid centered moment bounds, or invalid integrated probability trigger positive12-point Gaussian-density quadrature on the original bilinear continuation. With x=c+h*t and M=abs(c)*h+h^2/2, its real-arithmetic error is at most4*h*phi(c)*Pmax*exp(M)*M^11/11!, because degree10 exponential truncation times quadratic continuation is integrated exactly. Divide by conditional support probability, sum interval bounds, and propagate stage maxima. This is a guarded analytic/numerical transition, with explicit fallback counts; rounding remains uncertified. T55 independently checked these moment and fallback formulas.

Before trained comparisons: partition/constant continuation, a bilinear polynomial with direct analytic expectation, and thin/tail intervals checked against independent high-precision integration. Failures are retained. Then compare the same T7 mesh17/33, target8 rule against saved input8/16 results; no newMC, no h129, no changes to the previous quadrature baseline. Count crossings, sorts, intervals, moments, fallback nodes and CPU. Stop explicitly if the cycle exceeds its cap or probability guards fail.

Primitive controls (polynomial_checks.py, polynomial_checks_results.json) pass in.232CPU: eight intervals checked against70-digit integration have maximum relative error1.19e-15, including widths1e-12 and tails near±6. The known bilinear conditional expectation agrees within2.78e-17; T3/T7 constant-readout errors are5.55e-17. Two narrow-interval moment guards triggered and used the positive fallback. Some float discrepancies exceed tiny quadrature-only bounds, as expected from the explicitly excluded rounding error. T55 independently checked the bound in quadratic_gaussian.md with high-precision arithmetic.

First trained polynomial cycle (polynomial_cycle.py, polynomial_results.json) records a cost failure: h17 gives.23202739 in2.968CPU, compared with input8=.23215131 in.908CPU. The numerical scalar-input bias removed is about.000124, while the hidden-grid error is much larger. The h33 row hit the18CPU cycle cap and is retained as a failure. The h17 fallback-only integration bound6.09e-18 excludes floating moments, grid and outer-target errors; it is not a total accuracy bound.

Next optimization is frozen before timing: the conditional transition is linear in the continuation grid and does not depend on the target label. Build its sparse four-corner interpolation weights once per(stage,target quantile), charging all interval/moment construction, and reuse only within a single estimate across labels. Preserve the direct polynomial implementation as a control. Verify constants, row sums, no materially negative weights, and same-mesh estimates before any cost conclusion. No free cross-model or cross-run caches.

Within-run label reuse (cached_cycle.py, cached_results.json) passes: direct/sparse primitive agreement and constant controls are at rounding scale; h17=.2320273938422007 differs from direct by2.78e-17 and takes1.672CPU rather than2.968. h33 completes at.2341528809641711 in8.031CPU. Matrix row defects<=7.78e-15, with no negative retained weights. Conditional-input GL16 on this sameh33 mesh was.23417594017, only.0000231 away and cheaper at3.743CPU. Thus the analytic operator is a useful conditional-integration reference, but not yet a cost improvement over GL16. All192 sparse kernels and their6.25M nonzero entries at h33 are freshly built inside each estimate; fallback-only bound2.57e-17 remains separate from rounding/grid/outer-q errors.

Next frozen reuse step: scalar ReLU/grid crossings themselves are independent of target q. Build full-input interval geometry and basis moments once per stage; for each q reuse complete intervals below/above u and recompute only the single interval crossing u per hidden state. Assemble lower and upper kernels independently, avoiding subtraction of nearly equal total/cumulative masses. Validate against the current sparse implementation on the same controls and h17 before h33. All full-piece and partial-piece construction remains charged within the estimate.

Full-piece reuse (cached_cycle.py --reuse-pieces, piece_cached_results.json) preserves h17/h33 estimates to2.78e-17/zero and reduces CPU to1.003/3.691. At h33 it computes386807 interval moment triples instead of3487696, retaining the same sparse kernel entries and row checks. This is now comparable in measured cost to GL16, while removing its conditional-input quadrature error for the fixed interpolant. It does not remove the larger hidden-grid error.

Next frozen terminal reuse: when the target is not the final index, accepted scalar-input intervals depend on hidden node and label but not target value u. Cache those intervals once per label within the estimate, then evaluate their truncated CDF intersections in batches for each q. The target-last suffix still depends on u and will be rebuilt. Preserve root/tie logic and compare unchanged h17/h33 outputs; count both interval construction and all CDF queries.
Apply the same terminal reuse to a GL16h33 baseline so any generic caching gain is not misattributed to analytic transitions.

Terminal reuse (cached_cycle.py --reuse-pieces --reuse-terminal, terminal_cached_results.json) passes scalar/batched equivalence and constant controls. Analytic estimates remain.2320273938422007(h17) and.2341528809641711(h33), now taking.496/1.949CPU. The same terminal reuse gives GL16h33=.2341759401739373 in2.098CPU. The two methods now have comparable measured cost; the analytic one supplies a conditional-integration reference without a tunable input-node count. This single timing comparison does not establish a portable speed advantage. Spatial and target quadrature errors still dominate overall uncertainty.

Next frozen validation: repeat the known H2T4 h17/h33 comparison with analytic transitions to expose any cancellation previously masked by input8 quadrature, and run T7h65 once with the same analytic operator/target8 rule. No h129 or new samples. Retain any resource-limit failure rather than silently weakening the check.

Final phase checks (refine.py t4 --analytic; fine_mesh.py --analytic): T4 analytic h17/h33 errors are+.00045804/+.00013648 against the cone reference, at.136/.396CPU. Thus the previous GL8h33 error+.0000222 benefited from cancellation; increasing integration accuracy did not improve its total error. This validates the warning against selecting an apparently best setting by reference agreement.

T7 analytic h65=.2346504539637787 in10.542CPU, peak RSS995076KB. Its h33-to-h65 change+.00049757 is similar to the GL8 sequence, demonstrating unresolved spatial error independently of conditional quadrature. All matrix row defects remain below8e-15. The1.38e-8 input-tail bound and2.43e-17 fallback-quadrature bound do not include this spatial error, outer-target error or floating rounding. The original challenge is not solved by this result.

## Compute accounting
Charge all target nodes, terminal gate/margin interval operations, conditional quantile transforms, hidden-grid construction, interpolation and rule setup; report CPU separately from operation counts. No new Monte Carlo or training. Each cycle<=120CPU with one thread,2GB cap and shared compute lock; checkpoints after coherent results. Large query arrays chunk below65536 points.

The experimental estimator.py exposes est(model,budget,rng). It fixes hidden17/target8-per-half, analytic cached transitions and terminal reuse. Budget is explicitly advisory; actual charged CPU-calibrated forward equivalents are returned, including deterministic-zero timing calibration. Imports are excluded, calibration outputs never enter the accuracy estimate, and this is not a portable FLOP ledger. interface_check.py passes the constantT3 control with5.55e-17 error and returns a positive finite measured cost. The returned cost, which can exceed the requested budget, must be used in comparisons.

## Claims
[inferred] Conditioning removes recurrent target-coordinate interpolation. [measured] Exact toy/constant/tie controls pass; the optimized analytic engine gives T7 LOCALPILOT.23465045 at hidden65, while successive spatial refinement still moves.00049757. [measured] Charged reuse preserves finite-grid estimates and reduces cost. This is deterministic numerical transport, not an exact accuracy certificate or an equal-compute sampling win.

## Alternatives not ruled out
1. An incorrect1/m law or missing1/T factor could make plausible probabilities. Constant readout, constant continuation and toy analytic controls test this before trained runs.
2. The apparent gain could hide terminal label smoothing. Explicit scalar decision intervals and first-index tie controls isolate this source.
3. Hidden-grid interpolation or target quadrature may still dominate bias. Refine one axis at a time and retain all results, including nonmonotone or negative findings.
4. Finite hidden boxes might clip expanding dynamics. Use stage-specific proved boxes and fail on out-of-domain queries; account separately for discarded input-tail mass.

## Failure modes
No certified bound for grid interpolation or outer quadrature. The optional quantile transition also has unbounded quadrature error; the guarded analytic transition bounds only its numerical fallback in real arithmetic. Gate and moment arithmetic use floats. Rare conditional tails need stable CDF differences. Positive-mass output ties cannot be discarded. Truncation guarantees alone do not certify total accuracy. Only one LOCALPILOT T7 and one zoo T4 checkpoint were tested, not an independent noisy ensemble or a full shared benchmark. The original challenge remains open.

## Spatial diagnosis protocol (T67 next phase, frozen before measurements)

Design input: T63 `understand_m43/README.md` and independent R2
`review/understand_m43/README.md`, read 2026-10-17. Their confirmed result is
scoped to M4T3 seed0/noise; it is not evidence that this transport mesh works.
T76 owns a separate H2 boundary-family estimator. We test only mesh placement.

Hypothesis: terminal conditional probabilities have curvature/kinks where the
accepted scalar interval endpoints meet the target, gate, or readout boundary.
A rectangular mesh spreads those features. Alternative A: errors mostly arise
at earlier hidden stages, so changing only terminal-stage placement will not
help. Alternative B: the outer target rule or cancellation dominates, so a
better local interpolation diagnostic need not improve full accuracy.

Freeze this first test: H2T4 exact reference, analytic transitions, h17 and h33,
q8 per sign half. Keep earlier stages and all node counts unchanged. At final
hidden stage build a deterministic monitor using 32 quadratic-spaced bins per
axis, nine quadratic-spaced points on the other axis, and q4 per sign half.
For each nonfinal possible label/need flag, compare the analytic terminal at
bin midpoint with its linear endpoint interpolant. Average absolute residuals
with Beta weights and uniformly over the other coordinate, labels and flags.
Assign half the bin allocation to the original quadratic prior and half to
cube-root residual density; this is a frozen heuristic, not a proven optimal
mesh. A reversed-monitor allocation is the negative control. Rebuild and charge
all monitor intervals/CDFs inside every estimate; no free cross-run reuse.

Report local monitor residuals and full error/CPU for all three meshes, then
repeat q16 for every h33 mesh to distinguish target error/cancellation. No
parameter selection from the exact reference, no new MC, no h129. Rectangular
axes cannot conform to oblique boundary lines, and the monitor's uniform
cross-axis weighting is not the reachable hidden-state law. Preserve failures.

First spatial results (`spatial_cycle.py`, `spatial_results.json`): at h17 the
baseline/curvature/reversed errors are +4.5804e-4/+4.4647e-4/+5.2092e-4;
at h33 they are +1.3648e-4/+7.1463e-6/+1.3274e-4. Charged h33 CPU is
.435/.537/.534 seconds, including about.129 seconds of monitor construction.
But the frozen q16 check (`spatial_cycle.py --refine-q`) changes those h33
errors to +1.9415e-5/-3.2692e-5/-6.0493e-8. Thus the apparent improvement is
not stable; the reversed control nearly matches truth through potential
cancellation. This is a negative finding, not a selected winning mesh.

Next bounded diagnosis, frozen now: measure terminal midpoint interpolation
residuals for each stored mesh using a separate q8 rule and 17 cross-axis
points (monitor used q4 and nine). Weight bin residuals by their mass under
the original quadratic coordinate prior. This is a local diagnostic, not the
reachable-state distribution or a global error certificate. Then q32 and,
if affordable, q64 for all three h33 meshes, preserving every row. This tests
whether spatial or target quadrature can actually be isolated at these costs.
No mesh parameters will be changed based on the result.

Completed first spatial cycle, including negatives:

| h33 target nodes per sign half | baseline error | curvature error | reversed error |
|---|---:|---:|---:|
| 8 | +1.36478e-4 | +7.14635e-6 | +1.32744e-4 |
| 16 | +1.94152e-5 | -3.26924e-5 | -6.04931e-8 |
| 32 | +1.16201e-4 | +2.54092e-5 | +1.06695e-4 |
| 64 | +4.40260e-5 | -5.51089e-6 | +3.73246e-5 |

Commands: `spatial_cycle.py`, `spatial_cycle.py --refine-q`, and
`spatial_cycle.py --q 32/64`. Each process used one thread, the review lock,
20CPU-second cap and 2.5GB virtual-memory limit; each cycle completed below
18CPU seconds. The q64 estimates cost2.38/2.20/2.36CPU. Timing variation and
changed sparse interval counts mean equal node count is not exact equal cost.
The independent `spatial_diagnostic.py` shows h33 mean midpoint residuals in
coordinate0 of .003819/.000756/.003487 and coordinate1 of
.000476/.000417/.000663. These diagnostics use the prior coordinate measure,
not actual reachable-state mass. Local curvature reduction is measured; a
stable global accuracy improvement is not established. Outer target error is
large enough to change the apparent ordering. The q16 reversed near-exact
value is retained as a cancellation warning, not selected as an estimator.

Mechanistic interpretation: Wh[1,0]=-130.4538 while Wi[1]=.436625, so the
second gate's scalar-input root changes by about299 standard deviations per
unit of hidden coordinate0. A typical normal-input transition layer therefore
has width about.00335 in h0. Baseline h33's first positive h0 node is.00974;
the curvature mesh adds nodes.00231,.00462,.00693,.00924. This supports the
narrow-layer explanation for the local interpolation defect, not a complete
account of full-transport error. Earlier stages and target quadrature remain.

`spatial_controls.py` verifies constant-class1/T and positive per-hidden-unit
rescaling with both monitor orientations. The canonical `estimator.py` is
unchanged; experimental mesh_builder is optional and default behavior stays
identical. Monitor and matrix construction are charged inside every estimate.

Next mechanism test, frozen before new runs: eliminate the empirical curvature
monitor and place terminal-stage axis nodes on an exponential map with the
weight-derived gate scale s_d=min_i |Wi_i/Wh_id| over nonzero numerator and
denominator. The gate root x_i(h)=-(Wh_i dot h)/Wi_i shifts by one standard
normal input unit when coordinate d shifts s_d. Set h_d(z)=s_d*expm1(z*
log1p(B_d/s_d)), z uniform[0,1], keeping exact box endpoints and node count.
If no such slope exists retain the original quadratic axis. This heuristic
resolves thin gate layers near zero; it does not locate every oblique readout
boundary or use the reachable distribution. Wh/Wi=0 degeneracies are excluded
from its scale, not asserted harmless.

Alternative A: the observed improvement came from non-gate readout boundaries,
so this gate-only mesh will not reduce local residuals. Alternative B: too many
nodes are moved toward zero, worsening earlier accessible parts of the box;
a local improvement may not improve final accuracy. Compare baseline,
curvature and gate meshes at h17/33, q32 then q64, retaining all results and
charging construction inside each estimate. Reuse the independent local
residual diagnostic; use no exact-reference parameter tuning. Constants and
positive hidden-unit scaling must remain invariant. No new MC or larger grid.

Weight-only gate mesh result (`spatial_cycle.py --gate-cycle --q 32/64`):

| mesh size | q32 gate error | q64 gate error | q64 baseline error |
|---|---:|---:|---:|
| h17 | -2.68271e-6 | -3.53116e-5 | +1.12520e-4 |
| h33 | +2.10492e-5 | -1.08733e-5 | +4.40260e-5 |

Gate scales are .003347 and .61456, computed solely from weights; no monitor
calls. The h17 q32 near-exact result fails q64, so is not a chosen setting.
Independent local residual diagnostic (`spatial_diagnostic.py --source
 gate_q32_results.json --output gate_diagnostic_results.json`) improves h33
coordinate0 from.003819 to.001737 and coordinate1 from.000476 to.000408.
This supports narrow gate layers as one source of interpolation error, while
curvature targeting better captures other terminal features. Earlier-stage
error and outer integration remain unresolved. Gate constant/rescaling tests
(`spatial_controls.py --gate`) agree within8.4e-17.

Next bounded isolation check: all three h33 meshes at q128 and q256 with the
same frozen meshes; no spatial changes. This increases only outer integration
to distinguish apparent spatial gains from the observed q oscillation. Each
process remains<=110CPU/2GB using the primary lock if it can exceed20CPU.
If q128-to256 remains large, stop interpreting global mesh accuracy; report
outer integration as an unresolved bottleneck rather than refine blindly.

Distinct outer-variable discriminator (frozen before implementation/results):
compare equal-node positive Gauss rules in target value u on[-L,0],[0,L]
versus quantile q on[0,.5],[.5,1]. In the value rule q=F(u) and the weight
is T(T-1)F(u)^(T-2)(1-F(u))*phi(u)/(1-2Phi(-L)). This is a change of variables
in the same fixed-target law, not a fitted density. Quantile endpoints can
compress normal-tail and narrow grid-transition features; value coordinates
might resolve them better. Alternative: nonsmooth grid-induced breakpoints
inside the interval dominate, so changing variable will not stabilize it.
Use h33 and the three frozen meshes, 32/64 nodes per half only. Report the raw
constant-mass defect and fail if it exceeds1e-10; no silent renormalization.
Constant-class and gate/reflection toy controls precede interpreting trained
results. No larger hidden mesh, no new MC, and no winner claim if either
coordinate remains unstable. T55 independently reviews the measure formula.

Final q-only confound check completed: at q128 the baseline/curvature/gate
h33 errors are +5.16295e-5/-1.86411e-5/-2.22842e-5 (13.30CPU total); at
q256 they are +5.11377e-5/-3.09970e-6/-8.42545e-6 (23.57CPU total).
The curvature/gate q128-to256 change remains1.55e-5/1.39e-5, chiefly in
label2's contribution. Thus those global mesh errors remain unresolved.
No more q-only refinement is planned for this phase.

The full[-6,6] target-value rule failed its first constant control before any
trained run: sparse conditional row-mass defect7.89096e-10 exceeds the
1e-10 guard (`value_controls_results.json`). Its near-tail u nodes correspond
to1-q around1e-11; reconstructing u and dividing a tail CDF by rounded1-q
amplifies floating error. The guard remains unchanged. This implementation
failure does not reject the change of variable or the transport method.

Bounded followup frozen before measurement: integrate target u only[-4,4]
while all model inputs and hidden boxes remain truncated at[-6,6]. Compute
q_minus=F_6(-4), q_plus=F_6(4) and omitted target mass
betainc(T-1,2,q_minus)+betaincc(T-1,2,q_plus). Do not renormalize weights.
Retained constant-class integral must equal(1-omission)/T; any other bounded
conditional correctness differs from the full-target integral by a value in
[0,omission]. Original-input uncertainty adds Z^T*omission to1-Z^T.
Grid/outer/rounding errors still excluded. T55 independently approved this
measure and tail accounting. This isolates the existing floating-tail issue
without relaxing guards or implementing a new conditional-probability API.

Bounded target-value controls pass (`value_controls.py --target-limit 4`):
T3/4/7 constant errors against retained_mass/T are<=6.2e-16, raw density
mass defects<=1.9e-15, and sign-toy discrepancies fall inside their explicit
omission allowance. T4 omitted target mass is6.0179e-9; its composition with
input truncation is about1.39e-8, excluding all numerical/spatial errors.

Equal-node trained value-rule comparison (`spatial_cycle.py --gate-cycle
--q 32/64 --hidden 33 --rule split_value --target-limit 4`):

| nodes per sign half | baseline error | curvature error | gate error |
|---|---:|---:|---:|
| 32 | +1.92637e-4 | +1.62713e-4 | +1.58678e-4 |
| 64 | -2.87531e-5 | -1.78355e-4 | -1.84753e-4 |

These movements are much larger than the explicit target omission. Changing
the variable alone therefore does not stabilize this implementation. We do
not choose q256 curvature by its fortunate closeness to the reference, nor
reject the general rank-conditioned transport method on this evidence.
The canonical estimator remains unchanged. All experimental rules retain
positive weights, exact scalar tie handling and unchanged probability guards.

Finite alternatives checklist for this bounded phase:

- Rank/normalization defect: independent algebra and constant/sign controls
  support the law, including the target-value change of variables. This is
  not a floating-point proof of the full trained implementation.
- Narrow terminal gate layers: supported locally. The weight-derived scale
  and independent-density diagnostic show that mesh placement reduces their
  interpolation residual, but the diagnostic does not use reachable mass.
- Quantile endpoint steepness as sole global cause: the matched-node target-
  value comparison does not support this explanation; it is insufficient.
- Interior target-value kinks from target insertion/grid crossings and
  conditional interval endpoints: still plausible. Label2 carries most
  observed q oscillation. An explicitly panelled rule or an independent
  fixed-target conditional reference would discriminate this, but was not
  implemented here; blindly increasing q/h is not the next step.
- Earlier-stage spatial error and cross-coordinate weighting: unresolved.
  Terminal-only mesh changes cannot settle them; T77 reachability information
  could support a separate future weighted diagnostic.

Status of this phase: bounded experiments complete, independent T55 review
requested; global accuracy/cost improvement remains inconclusive. This is
neither a method-level rejection nor a challenge solution. Root owns task
registration and any followup prioritization. All commits retain the original
quadrature-cancellation and tail-normalization failures.

Independent T55 review (a8d5898, feynman_fundamentals/spatial_transport_review.md)
accepts scale covariance, target-value/truncation algebra, and the bounded
inconclusive interpretation. Its additional limitation is adopted: midpoint
probes change with each tested mesh, and their weighting is artificial, so
the reported local residual reductions are diagnostic evidence, not a global
or even common-probe dominance claim. A distinct future cheap check would
compare every interpolant on the same frozen off-grid probe set. Interior
target/grid kinks and earlier-stage interpolation remain open alternatives.
This review does not classify rank-conditioned transport as a dead method.

## T67 interior-kink phase (claim pushed1b7bfe1)

Question: do known discrete transitions inside the target-value integral,
rather than broad quadrature resolution, explain the oscillation? Freeze a
complete breakpoint construction for models whose possible predicted labels
exclude the last position (true of the current H2T4 base). Do not silently
apply it to a target-last suffix, whose topology needs additional derivation.

For every hidden stage before the scalar terminal, the existing StagePieces
cache stores all scalar-input values at which ReLU or next-grid coordinates
change pieces. The same list covers inserting the fixed target at that stage.
For each possible nonfinal target, the terminal cache contains accepted
scalar intervals independent of target value; their endpoints are exactly
where lower/upper conditional terminal masses change formula. The union of
these endpoints, input-grid crossings,0 and integration-domain endpoints
therefore contains all target-u formula changes of the supplied finite-grid
interpolant. Between them, conditional Gaussian moments/CDFs and interpolation
are analytic; finite spatial bias is still present. Floating endpoint
construction is not an exact-real geometry proof.

Alternative A: hidden interpolation dominates despite these kinks; panelled
outer rules may stabilize while still missing the exact model accuracy.
Alternative B: missing terminal or target-last transitions invalidate the
panel list; exclude target-last predictions initially and compare omission
controls. Alternative C: finding all transitions is too expensive. Count
panels first on h9/17/33, then integrate only bounded feasible cases.

First experiment: baseline quadratic mesh, existing H2T4 model, target[-4,4]
and inputs[-6,6], with the already-derived target/input omitted-mass bounds.
Compare GL2 andGL4 on every complete breakpoint panel to a uniform-panel
control with exactly the same number of panels/nodes. Stop rather than expand
if there are more than2000 panels at h9. Charge breakpoint/terminal/cache
construction inside each estimate. Stream one target node across all labels
before discarding its conditional matrices; retain full-stage pieces only
within this estimate. This changes memory order, not the transport formula.
Keep no global cross-run model-dependent cache. Test constant-class mass and
same-fixed-node streaming equivalence before interpreting trained results.
Numerically unresolvable intervals may be omitted only with their full
width times a global target-density upper bound recorded separately. Never
renormalize quadrature mass or drop an unexplained probability-guard failure.
