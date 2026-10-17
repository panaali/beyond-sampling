# T67: exact first-input marginal mesh discriminator

## Question
Does matching the stage-1 interpolation mesh to its exact physical source
measure reduce full transport bias at unchanged node count?

## Method
Only stage1 changes. Preserve h9, all later squared meshes, analytic Gaussian
transitions, exact scalar terminal and complete target breakpoints. For T
positions and possible label set S of size K including0, conditional first
input CDF-coordinate density is [(T-K)g(q)+T(K-1)]/[K(T-1)], where
g is Beta(T-1,2). When0 is excluded it is (T-g)/(T-1). This follows from
uniform target-label prior and exchangeability of the other positions;
conditioning on S is only a mesh monitor, not a change to the accuracy law.
T55 independently checked this formula. Use truncated normal [-6,6].

Mixed-sign Wi puts the exact first hidden state on two axes. For each positive
axis, form the exact conditional magnitude CDF given the appropriate input
sign. Invert a frozen equal mixture of that CDF and sqrt(s/6), the baseline
squared-grid quantile law. Keep endpoints0,6 and scale by abs(Wi). The
reversed control replaces physical CDF(s) by1-CDF(6-s), preserving all cost
and node counts. Do not change target conditioning, label weights or grids
from stage2 onward. Reject same-sign/zero Wi outside this bounded prototype.

Use h9 baseline/physical/reversed and complete q-panel GL2/4, cap2000 panels.
Target u remains[-4,4], inputs[-6,6]. Positive Beta polynomial weights retain
raw mass; account collapsed/tiny panels and outside target/input mass. No
renormalization. No random model inputs, new training or canonical edits.

## Frozen alternatives and falsifiers
1. Stage1 mass is poorly resolved: physical mesh should reduce absolute full
   accuracy error more than reversed and baseline, at matched nodes/cost.
2. Earlier measured gains were cancellation: physical local density matching
   may worsen signed/full error. Preserve all variants; no truth-based choice.
3. Outer rule confounds spatial error: GL2/4 change must be below1e-7 and
   below10% of each claimed improvement before interpreting a gain.
4. Marginal rather than q-conditional allocation is insufficient: failure does
   not reject physical/rank-conditioned meshes generally. No further q/h sweep.

Before model integration, verify CDF normalization/monotonicity, sign masses,
independent direct marginal integration, positive scaling, unchanged later
axes and constant/duplicate-readout first-index controls. Then count all
three complete topologies. If any exceeds2000 stop before full integration.
One fixed H2T4 seed0 only; no ensemble unless BOTH comparisons improve error
and error-squared CPU with stable outer rule. Any expansion needs a new
bounded protocol and independent review.

## Status
Protocol frozen before implementation and measurements; claim9804ae0 pushed.

## Results
Pending. Existing reference .30773074302392744 is comparison only.

## Compute accounting
Charge CDF/beta construction, scalar root inversion, grids, all StagePieces,
terminal caches, breakpoint sorting, Gaussian moments, sparse operators and
outer integration. Full cold CPU plus event counts, no portable FLOP claim.
One thread, <=2GB, <=110CPU cycles; locks follow board. No new Monte Carlo.

## Claims
[inferred] Fixed source meshes preserve fixed-interpolant breakpoint proof.
[speculative] Physical first-stage allocation improves final accuracy/cost.

## Alternatives not ruled out
q-dependent meshes, later-stage errors, internal cut-aligned approximation,
finite-volume representations and favorable cancellation remain open.

## Failure modes
Incorrect label mixture/sign normalization, inverse-CDF root precision,
extra target topology, spatial error cancellation, and treating raw positive
mass/GL refinement as a complete numerical error certificate. Floating
geometry/roundoff and spatial error remain uncertified.

## Control checkpoint
Run `python m16/approaches/rank_conditioned_transport/source_mesh_cycle/controls.py`
with one-thread environment and the review lock. Direct conditioning on the
target quantile (separate above/below formulas) reproduces marginal CDFs
within1.11e-16 for four label-set cases, including label0 excluded. Sign
CDF endpoints/monotonicity, unchanged later grids and1e-16 positive scaling
pass. Constant/duplicate readouts differ from retained target mass/4 by at
most2.78e-17. Complete h9 panel counts baseline/physical/reversed are
1190/1191/1221, below the frozen2000 cap. Control CPU .692; full event counts
and exact axis locations are in controls.json. Ready for frozen GL2/4 pilot.

## Frozen pilot: physical allocation loses to baseline
Run `python m16/approaches/rank_conditioned_transport/source_mesh_cycle/pilot.py`
under the primary lock with one thread. All six integrations complete under
the100CPU internal budget; no larger grid, target rule or model sweep follows.

| Stage1 mesh | GL4 estimate | Error vs reference | GL2/4 change | GL4 CPU |
| --- | ---: | ---: | ---: | ---: |
| baseline | .307723248833985 | -.000007494190 | 8.69e-12 | 14.987 |
| physical | .307741572240580 | +.000010829217 | 3.18e-12 | 14.822 |
| reversed | .307539712539522 | -.000191030484 | 1.36e-12 | 15.181 |

Physical allocation improves over reversed but is worse than the original
baseline. Error-squared CPU products are8.42e-10/1.74e-9/5.54e-7 in table
order. Outer refinement cannot explain this material ordering. This rejects
the fixed marginal-placement recipe's frozen expansion gate, not physical
or q-dependent meshes generally. The first-input marginal is(g+8)/9, close
to a uniform CDF-coordinate law; nonlinear interpolation placement and
cancellation can therefore dominate the modest rank-law change.

Cost qualification: each call charges grid/mesh, caches, topology and target
rule anew. The existing polynomial primitive has a process-wide cached
12-point fallback rule, initialized once in the complete pilot; later rows
reuse this small rule. Thus these are not perfectly independent cold starts
as the protocol intended. All its quadrature applications are still counted,
and no efficiency gain is asserted (the accuracy gate already fails).
Full pilot totals and named operation events remain in pilot.json. Floating
geometry, spatial bias and total portable cost are still uncertified.

T55 source/law review2db43ce checks mixture/sign normalization and stage1-only
placement. Independent high-precision elementary-CDF inversion is pending
before the bounded handoff; it will not independently reimplement transport.

## Independent placement check and handoff
T55 independently inverted the elementary T4 marginal polynomial using
60-digit bisection. All36 physical/reversed stage1 coordinates agree within
2.103e-15; the independent check costs .139CPU. This validates placement,
not the shared transport recurrence or full accuracy. Raw review artifact
SHA256 `1643161976030313f1ff83b3640c8548d978c8b1be79b28f48c0e3dc88d6e47e`.
Together with source review2db43ce, constant controls and stable GL2/4,
this supports the bounded conclusion that this fixed marginal-placement
recipe fails the frozen expansion gate. The phase is complete; q-dependent
placement and direct removal of stage1 interpolation remain untested.
