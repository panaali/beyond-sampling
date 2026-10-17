# T67: reachable-state weighted terminal interpolation diagnosis

## Question
Do terminal mesh changes that reduce artificial local residuals still fail
because they allocate resolution away from physically reachable states?

## Method and frozen protocol
H2T4 seed0 only; final target label3 is excluded only after the readout's
componentwise domination certificate passes. For each nonfinal target j,
q=F_6(u) has Beta(3,2) law. The unique above-target position is uniform over
the three non-target slots. Generate the two non-target values before the
last step using independent conditional-CDF quadrature, insert u at j,
and propagate the actual three-step recurrence. These reachable h values
and their weights are identical across all mesh variants. The last scalar
input is integrated analytically by the existing first-index terminal
interval engine. Its need flag is1 exactly when the unique-above slot is
last. Compare this value with bilinear interpolation of terminal-grid
values at the same h, reporting signed and absolute weighted residuals.

Baseline is the existing squared-coordinate grid (mesh_power2), not a
uniform hidden-coordinate grid. Compare baseline, prior terminal-curvature,
and prior gate meshes at h9 and h17. No new mesh design until this frozen
diagnostic finishes. q rules use GL8 in each half [0,.5],[.5,1]; prefix
rules use GL8 then GL16 per conditional input. q refinement to16 perhalf is
one preregistered discriminator only if prefix refinement is stable.
All rules are deterministic; no new Monte Carlo or training. These are
weighted diagnostics, not a certified full accuracy calculation.

The labels contribute1/4, above-position branches1/3. Truncated input law
omits at most1-(1-2Phi(-6))^4 of original correctness. Rule weights retain
their raw mass. No renormalization or clipping of correctness.

## Alternatives and falsifiers
1. Reachability matters: prior mesh ordering changes under physical weights.
   Test all meshes on identical frozen nodes, signed and absolute errors.
2. Error cancellation matters: a mesh can improve absolute residual yet
   worsen signed residual or full finite-grid accuracy. Retain both and do
   not infer a global winner from this local diagnostic.
3. Earlier interpolation dominates: stable terminal residual can disagree
   with full-grid reference error. This diagnostic then localizes only one
   component; no general conclusion about reachable weighting follows.
4. Rule error dominates: GL8/16 prefix or optional q refinement changes
   ordering materially. Mark numerical diagnosis inconclusive; do not launch
   an unbounded degree sweep.
5. Cost dominates: mesh preparation and all exact scalar controls are counted
   separately, not supplied as free advice. No portable FLOP claim.

## Controls before trained diagnostics
Constant/duplicate readouts give first-index correctness1/4 after target
averaging, and terminal interpolation residual0. Wh0 makes terminal values
independent of incoming hidden state, hence residual0. Verify all reachable
states lie in the declared stage box and probability/weight bounds. Mean
and maximum rank-weight mass defects remain explicit.

## Status
Protocol frozen before implementation and trained measurement.

## Results
Pending; preserve every failure and implementation correction.

## Compute accounting
One thread,2GB, shared numerical locks; <=110CPU/cycle, internal deadline100.
Charge conditional inverse-CDF nodes, recurrence matrix work, grid setup,
terminal interval discovery/CDF evaluation and interpolation. Diagnostic
rules and refinement charged separately. No new leaderboard claim.

## Claims
[inferred] The diagnostic isolates terminal-grid interpolation under the
rank-conditioned physical source distribution in exact arithmetic.
[speculative] This identifies a useful mesh-placement mechanism.

## Alternatives not ruled out
Earlier-stage interpolation, target-last topology, non-tensor meshes,
reachable-mass-aware adaptive basis, and full forward transport remain open.

## Failure modes
Unsplit gate/readout kinks in prefix quadrature, q-endpoint amplification,
floating scalar roots, tiny probability intervals, and signed-error
cancellation. Independent review is required before method closure.

## Primitive control result
Constant, duplicate-readout and Wh0 controls pass across all three meshes:
maximum absolute weighted interpolation residual1.74e-17, target-rule mass
correct within1e-13, exact first-index constant accuracy1/4. CPU.272 total.
Controls use q4 perhalf/prefix4 and h5; full arrays/event counts preserved in
controls.json. Trained frozen comparison follows these controls.

## Frozen trained diagnostic result and next discriminator
At q8 perhalf and prefix8/16 the physical absolute terminal residual is:

| h | Mesh | prefix8 | prefix16 |
| --- | --- | ---: | ---: |
|9|baseline squared|.00726456|.00703559|
|9|curvature|.000817789|.000766126|
|9|gate|.000779193|.000702698|
|17|baseline squared|.00232511|.00171563|
|17|curvature|.000275990|.000256593|
|17|gate|.000201066|.000140246|

The absolute improvement survives identical physical probes at both rules.
This contradicts the proposed explanation that artificial source weighting
alone caused prior local gains. However, signed residuals are not converged:
h9 baseline .000269871→.000223785, curvature .000157442→.0000649921,
gate -.0000420292→+.00000586077. Direct source quadrature changes
.307998158→.307582910 (reference .307730743). No q refinement is launched:
the frozen prefix-stability condition fails for signed bias. CPU.628/1.324.
All arrays, shared source cost and per-mesh setup costs are in pilot.json.

T55 independently checked the conditional law and scoped interpretation.
A mesh's smaller absolute terminal error does not imply smaller signed full
transport bias. Earlier interpolation and cancellation remain open.

Supplemental discriminator frozen before measurement (T55 recommendation):
repeat the SAME prefix16/q8 probes once, separating origin, each coordinate
axis and interior contributions. Record mass, signed/absolute residual and
positive-volume terminal readout ties. No new mesh, nodes, q refinement or
accuracy tuning. If non-origin tied affine readouts can occur, report that
limitation rather than silently classify every tie as a zero hidden state.
