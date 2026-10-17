# T49 three-input surviving suffix

## Question
How much unresolved H2T8 accuracy contribution can one additional surviving
input resolve after the certified reset wedge, and at what branch/integration cost?
Claim1dd1493 was pushed before implementation. This continues reset_wedge
handoffc00a89a without duplicating its completed zero/one/two-suffix cases.

## Method
Use the last certified pair reset ending at T-3. Its older T-5 inputs remain
iid unrestricted; the reset pair retains the exact coupled rank law. The
three suffix inputs a,b,c obey a>0, b>-k*a, c>-k*max(b,0). Their possible signs
are +++, ++-, +-+. In each sign orthant use positive L1 simplex coordinates
(a,abs(b),abs(c)) with sum1; scalar positive radius is integrated separately.
Clip this triangle by exact Fraction reset, ReLU and readout planes. Resolve
first-index ties once; retain the fixed final hidden map on each polygon.

Cost-first gate: enumerate only base H2T8, at most256 gate polygons /1024 winner
polygons and30 CPU seconds, then inspect predicted label and selected-input
sign. If old/reset-pair labels or nonpositive selected suffix labels appear,
stop integration and freeze a separate rank-kernel choice rather than silently
applying a positive suffix formula. For positive suffix targets, subdivide by
other-suffix rank comparisons; combine the exact older-binomial/reset-pair law
with positive Gaussian radial/simplex quadrature. Freeze integration separately
after topology/count validation. No new noisy panel or model-truth fitting.

## Alternatives and falsifiers
1. Branch growth erases the reset simplification: exact topology/count gate
precedes all costly mass work; compare incremental resolved event mass.
2. A shorter suffix's positive selected-label property may fail: test labels
and sign on every exact winner polygon before selecting a rank primitive.
3. Shared cone code can conceal an incorrect domain: independently verify
survival inequalities, direct recurrence/winner at rational interior points,
partition area and analytic last-reset event mass.
4. Rank prefixes are coupled through the reset pair: retain its unnormalized
law and use an independent conditional-rank partition control.

## Status
Exact topology complete; positive-suffix-only premise failed. Integration paused at the predeclared discriminator while the old-label rank kernel is specified.

## Results
The base has only13 exact gate cells and17 winner polygons (10/3/4 across
+++, ++-, +-+), far below the frozen caps. All51 rational interior direct
recurrence/winner checks pass and exact polygon areas partition each domain.
Maximum vertex integer size117 bits;276 clips,915 plane evaluations and72
intersections; full setup .057232 CPU seconds,25 MB RSS.

However, predictions include label0 as well as suffix labels5,6,7. Five
positive-area winner polygons predict label0 and have identically zero final
hidden maps: an actual reset can occur without the sufficient uniform-state
pair certificate. This is a successful falsification of the positive-selected
suffix premise, not an implementation failure or a failed reset factorization.
No positive-suffix-only integral is run. The next bounded rescue must use the
existing generic older-target rank law or an independently derived analytic
radial treatment, preserving these five regions and their setup cost.

Reproduce `python topology.py` under the one-thread review lock. `topology.json`
saves exact vertices, maps, histories and source/checkpoint fingerprints.
Prior three contributions total.17257946073009367; unresolved event
mass.14285571617911716. These are qualified numerical partial contributions,
not full model accuracy or certified lower bounds after floating integration.

## Compute accounting
One thread,2GB, primary lock <=110CPU per phase; topology gate30CPU.
Report rational operations/proxy, integer bit lengths, cells, node evaluations,
all source imports/setup, CPU and omitted-event mass. No portable FLOP or
sampling advantage follows from a callback count.

## Claims
[inferred] Last-reset factorization gives a distinct additional stratum.
[speculative] Its rank-conditioned integral is cheap enough to be useful.

## Alternatives not ruled out
Longer surviving suffixes, reachable-state reset enlargements, direct bounds,
coarse support mixtures and conditional-history methods remain open.

## Failure modes
Inherited floating checkpoint conversion, excessive rational bit growth,
positive-volume tied outputs, negative selected targets, thin polygons,
unresolved quadrature/roundoff error and confusion between event coverage and
correctness probability. No method-wide negative follows from this phase.

## Frozen old-target rescue and integration
The five zero-hidden regions predict a particular older input (label0), not
an arbitrary prefix winner. All surviving sign patterns have at least two
positive suffix inputs. With N=T-5=3, largest L and second-largest S>0, its
unnormalized correct rank mass is

`int_S^L phi(t)*Phi(t)^(N-1)*g0(t) dt`
`+ int_L^infinity phi(t)*[(N-1)*sf(t)*Phi(t)^(N-2)*g0(t)+Phi(t)^(N-1)*g1(t)] dt`.

Here g0/g1 are the existing positive-threshold coupled reset-pair masses;
there is no N multiplier for a specified older label. Independent generic
conditional.py checks on fixed suffix triples precede numerical use.

Subdivide every exact winner polygon at all three suffix-equality planes,
so order/rank formulas are smooth inside each cell (cap256 cells). Triangulate
convex polygons from one vertex. In signed L1 coordinates y, Gaussian
Jacobian is `r^2 exp(-r^2/2)/(2pi)^(3/2)/||y||^3` after radius rescaling.
Use positive Duffy triangle Gauss rules angular orders8/16/32, radial32 on
[0,8]. Older-target inner rule32 on[S,L] and[L,8]; check64 against32 on frozen
fixed suffix controls before main integral. Report exact chi3 radial tail and
old-target normal tail allowances separately from uncertified quadrature.
Before correctness integrals, compare angular rule mass against independent
solid-angle triangle atan2 formula and analytic last-reset event mass. Stop if
geometry mass error exceeds1e-7 at order32 or wall/cpu exceeds90seconds; no
unbounded order sweep. Preserve all rule failures. This is one base-model
partial contribution, not an adaptive fitted estimator or noisy evaluation.
