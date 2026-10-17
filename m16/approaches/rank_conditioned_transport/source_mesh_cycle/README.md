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
