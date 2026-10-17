# T66 exact reset-root factorization

## Question
Do exact zero-hidden-state resets explain the residual H2 gate-piece discrepancy between long-double and mixed native circle walkers? TASKS-only claim76556aa was published and rechecked before implementation. Prior owner confirms this variant was untested. Sources remain unchanged in compiled_geometry; this folder owns the followup.

## Method
On a current affine branch h=a+b*t, if every previous hidden coefficient a,b is exactly zero from gate suppression, each next preactivation is Wi_j*(U_step+V_step*t). Cancel the common Wi factor before root division at every such step, using sign(Wi_j*V_step) for activation direction. Wi_j=0 stays off; V_step=0 uses the constant sign. This is the same algebraic cancellation previously used only at step0. No tolerance merging or reference-dependent arithmetic selection is allowed. Keep both long-double and mixed recurrence variants, and count all zero-prefix checks/reset uses.

## Frozen protocol and alternatives
Copy the prior factored/factored_mixed sources into this owned folder with source hashes. Build two separate reset variants; retain the old binaries/code for matched controls. First repeat the same21 Fraction-oracle rational/tie/scale fixtures and cap0/1 failures. Add a constructed late-reset fixture with zero affine history after a nonzero state, and verify first-index/zero-Wi/V=0 cases. Then repeat exactly32 orthonormal planes from RNG660001 on H2T10 and the original H16T10 plus relative-RMS .0025/.02 RNG1 copies. No new planes, seeds or accuracy truths. Compare outputs, piece counts per plane and reset counts for old/new long-double and mixed versions. Require integral agreement<=1e-12 with prior strict variants and oracle controls; report every count discrepancy rather than waiving the criterion.

Alternative1: all remaining H2 discrepancies come from late exact resets; factoring those roots makes paired counts equal. Alternative2: nonzero hidden affine dependencies or recurrence rounding create other coincident roots, so the repair is partial or null. A negative does not establish that double precision is invalid. Alternative3: apparently zero floating coefficients arose through cancellation, not structural gating; instrument active flags if necessary before claiming structural exactness. Cheapest discriminator is the unchanged per-plane count/output comparison and a concrete residual fixture, not a larger benchmark.

## Status
Bounded reset and axis phases complete with independent T55 reviews de5abd2/cfd905e. Bounded <=110CPU cycles, one thread, <=2GB, shared locks; short controls <=20CPU may use review lock.

## Results
Reset and axis exact-factor controls pass on the frozen workload; full results and limitations follow. Independent T55 reviews are recorded below.

## Compute accounting
Charge compilation, copied-source preparation, wrapper loading, plane/model preparation, finite checks and all kernel runs. Added zero-prefix tests cost at most2H coefficient comparisons per recomputed step and are counted explicitly. Record cold/warm CPU separately; existing arithmetic counters are partial, not a complete portable FLOP ledger. Random planes make this a sampling-based baseline primitive, not a deductive challenge solution.

## Claims
[inferred] A structurally zero previous hidden map makes all current nonzero-input-weight gates share the input root exactly. [speculative] This explains the remaining H2 finite topology discrepancies. No speed or accuracy result yet.

## Alternatives not ruled out
Compensated recurrence, general exact dependency detection and interval predicates remain untested. A finite parity result cannot certify arbitrary floating topology. No shared deployment or full challenge claim.

## Failure modes
Floating coefficients can be zero by cancellation, so a structural gate mask is the safer certificate. Tiny nonzero maps, root rounding, overflow/underflow, target ties on degenerate planes and compiler behavior remain qualified. CPU improvement alone is not equal-compute MSE improvement.

## Pre-run structural refinement
Use a cached Boolean all-gates-inactive certificate, initialized true at h0, rather than testing rounded coefficients for zero. Set the next certificate only when every gate is inactive on the represented branch. This excludes cancellation-only zeros; it does not certify the upstream floating gate topology. Charge one certificate lookup per recomputed step and H active-flag conjunctions; the previous proposed2H coefficient checks are not used. Preserve certificate values alongside cached affine coefficients on prefix reuse.

## Initial build defect
The scripted source edit inserted the zero-mask declaration in the mixed source but missed the differently formatted long-double declaration. That compilation failed (`zero` undeclared), then the control script failed loading the absent library. `failure_initial_build.json` preserves the downstream traceback; no numerical control ran. The failed compile/import CPU was not metered and is not silently included in successful costs. Adding the missing declaration is an implementation repair, not a method negative. The mixed build .688125CPU completed and is retained. Subsequent shell runs stop immediately on build errors.

## Bounded reset result and frozen exact-trained control
All28 rational/tie/scale controls and four cap failures pass. H2T10 old long-double/mixed gate-piece totals1240/1310 become1188/1188; every one of32 planes has matching paired counts. Compared with the old long-double integrals, every variant differs by at most8.33e-17. All96 H16 base/noisy planes retain their previous paired counts, with integral differences<=6.995e-15. The whole check costs2.004468514CPU including import/model preparation and controls; controls alone .040947636CPU after imports. Successful compilation .528548/.688125CPU is separate. Original failure and both arithmetic sources remain. This is a structural numerical repair on finite cases, not universal topology or a deductive estimator.

Before handoff, freeze a distinct independent exact-arithmetic trained control on H2T10 planes0 and31 from the same32-plane sequence. Use the existing Fraction exhaustive gate-history oracle (different algorithm from native sweep), 70-digit angular integration, and compare final gate-cell counts and probabilities against BOTH new variants. Run plane0 first; only attempt31 if oracle CPU for0 is under10seconds, keeping total under110CPU and retaining an explicit skipped status if exceeded. No plane is selected by disagreement or accuracy truth. Alternatives: paired floating counts reflect correct structural topology on these instances, or they merely share a rounding error. Exact agreement here remains finite evidence, not certification of all32. Charge Fraction work/time separately; exact rational growth is not free or a scalable solution.

## Exact trained and independent reset controls
Fraction gate-history enumeration on predeclared H2T10 planes0/31 gives38/36 gate cells, exactly matching both reset walkers. Probabilities .20462488942908272/.3443800056929379 agree within5.56e-17. Oracle CPU .04060/.04006 seconds; full script .913996433CPU includes torch/model preparation. Independent T55 de5abd2 uses a separately derived H3T4 late-reset network with closed-form circle probability3/8:24 phase/scale/kernel cases pass exactly, coverage1, positive late-reset counts. It also checks cached zero-flag invalidation. No whole-network floating certificate follows.

A distinct remaining negative is visible: exact Fraction classified intervals128/126, reset long-double168/168, mixed203/238. Gate topology is repaired on these cases, but redundant readout cuts remain sensitive to floating arithmetic. Preserve this failure rather than describing the complete partition as exact. H16 zero-reset count is0; its added mask overhead makes the tested reset kernels about8–14% slower than prior variants. H2 warm long-double .004435 versus old.004376CPU, mixed .003923 versus old.004165; these tiny timings support no portable speed claim.

## Frozen axis-readout followup
With exactly one structurally active final neuron k, h_k>0 on an open gate cell, so the winner is the first-index maximum of Wo[:,k], constant across that cell. With no active neuron the winner is0. Do not multiply every output by the common hidden affine factor or create readout pair roots that lie only at the hidden zero boundary. This is exact support algebra, not epsilon merging. Store final gate-activity flags and implement separate axis-factored variants; retain both reset-only versions.

Alternative1: the residual classified-interval excess consists of axis-readout boundary artifacts, so this factorization matches exact counts. Alternative2: multiple active hidden coordinates or other coincident affine differences produce residual cuts, requiring a different explicitly proved factorization. Repeat the same28 Fraction controls/caps and32 planes on all four frozen models, with exact planes0/31 comparison. Require unchanged integrals<=1e-12 and preserve every count residual; no wider sweep or model-dependent fallback. Count final-support tests, skipped readout comparisons and fixed-winner work; charge compilation and same-run comparisons. <=110CPU, same locks/memory. This is the finite numerical-representation alternative promised by T66, not a new accuracy estimator.

The first axis-variant source-generation attempt stopped before writing files because the long-double declaration used different formatting. The following build argument consequently failed; no compiler or numerical experiment ran. The exact replacement assertion was corrected, and shell fail-fast is now applied before source preparation too.

## Axis result and finite alternatives
All28 Fraction controls and four cap failures pass again. On H2T10,1113/1188 final gate cells are origin/axis-supported. Classified intervals decrease from reset-only long/mixed5411/6749 to4466/4466, with per-plane parity on all32. Gate counts remain1188/1188 and probability discrepancy<=8.33e-17. Exact trained planes0/31 now match BOTH oracle gate cells38/36 and classified intervals128/126; their probabilities match the oracle's final doubles exactly. All H16 gate/interval counts remain unchanged, and no final origin/axis cells are detected.

Matched warm H2 CPU decreases .004866/.004427 to .002895/.002706 (about1.68/1.64x relative to reset-only versions in this same run); the values are millisecond-scale and not a portable speed certificate. H16 adds support-mask overhead without useful axis skipping. Full check2.179100318CPU, exact-trained check separately saved, successful builds .547232/.542281CPU. Partial support/root counters include every kernel call; original benchmark artifacts remain. No claim of equal-compute sampling advantage or improved accuracy is made.

Finite tested alternatives: initial-step-only common factor; structural late-reset factor; exact Fraction gate-history discriminator; final origin/axis readout factor; same model/plane/precision controls; first-index/zero/tiny-gap/scale/cap fixtures. General multiactive affine dependencies, compensated recurrence and outward-rounded geometry remain open. The observed H2 gate and readout discrepancies have explicit algebraic repairs, so this task is a scoped positive, not a family-wide negative. Wider model/plane sweeps are unnecessary for that finite claim and are not automatically authorized.

Reproduce from repo root with one-thread environment and the shared review lock: run `build.py --engine` for factored, factored_mixed, factored_axis and factored_mixed_axis; then `check.py`, `check_exact_trained.py`, `check_axis.py`, `check_exact_axis.py` in this folder. Oracle and baseline source path/hashes are retained; no external internet source was used. Root/DRI decides shared adoption and registry status.

## Independent axis handoff
T55 cfd905e verifies final-support cache refresh, strict first-index Wo-column comparison, and retention of all input-rank cuts. Its separate48-case closed-form control uses a single active hidden coordinate: output3 on the positive ray yields exact3/8 circle accuracy; tied top rows2/3 with first-index output2 yield exact1/2. Both kernels, four frames and three scales pass exactly. No generic floating topology certificate is asserted. This closes the bounded T66 numerical repair; general native extensions remain open for a distinct future claim.
