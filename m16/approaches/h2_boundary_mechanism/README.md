# T76: boundary-family mechanisms for H2T4 and H2T5

## Question
Can the confirmed T63 boundary mechanism transfer to H2T4/T5 with work that scales in hidden width, sequence length and boundary families, rather than the number of full input-space linear regions?

## Method
Root reserved and pushed T76 at05bbd72 before this folder was created. T63 and independent R2 were read read-only. Their positive claim is scoped to M4T3 seed0 and noisy copies; the generic method fails on some other seeds, near-parallel gate/boundary crossings, and wall-clock comparisons. Claude owns T73/T75. This folder will not duplicate either engine or change T71.

For H2, the entire positive-state readout is the upper envelope of T lines Wo[j,0]+r*Wo[j,1], r=h1/h0>=0, with explicit axis and zero-state conventions. This reduces output boundaries to at most T-1 pair-ratio thresholds, but does not imply only two labels are reachable. An absent output label contributes an unavoidable1/T error under continuous iid inputs. At a nonzero output boundary c dot h_T=0, both final neurons are positive and x_last=-(c Wh h_previous)/(c Wi), subject to gate positivity. Thus final decision boundaries are weight-derived linear functions of the two previous state coordinates; small c Wi means a near-parallel sensitivity analogous to R2's lost crossover margin. Dead-state and single-axis readouts are separate boundary families, with first-index ties.

The first bounded diagnostic inventories these weight-derived envelopes, final-input boundary slopes, recurrence gains and unavailable-label ceilings on H2T4/T5 base and existing fixed relative-RMS .0025/.02 noise copies (NumPy seed1, all matrices). It reads existing exact/refined accuracy references only for validation. No random input, fitted constant or region enumeration is permitted in production.

## Frozen hypotheses and falsifiers
1. **Thin boundary displacement transfers directly.** Falsifier: the exact accuracy may be far below the weight-derived reachable-label ceiling, or the envelope may omit target labels. Then an error model close to one minus thin wedges is inappropriate. Record unavoidable representation errors separately, not as small boundary tilts.
2. **Few stable ratio boundaries control the error.** Falsifier: some output-boundary c Wi is near zero compared with the two uncancelled terms, or noise changes the envelope/zero-axis class. Record cancellation ratios and topology changes from the weights; no threshold or case may be selected using accuracy.
3. **Recent-input state dominates.** A possible bounded prototype sets the earlier state to zero and integrates its final two-input decision families with the exact rank weight of the omitted iid inputs. Falsifier: its fixed prediction disagrees with the existing exact references beyond sampling precision at its full charged cost. A weight-based recurrence remainder bound must accompany any use of this approximation; no fitted forgetting factor.

Freeze each prototype's algebra and quadrature orders before executing it. Retain every model and failure. Models and transformations may be used for exact zero/constant-readout and positive-rescaling controls; no new Monte Carlo or model-output fitting. Diagnostics may inspect existing exact-reference artifacts but production imports must not access their geometry or fitted advice.

## Status
Weight inventory and boundary-family derivation in progress. No estimator or transfer claim yet.

## Results
Pending. Existing H2T4 base accuracy is approximately .307730743; H2T5 base .279237371315. Their low accuracies make direct thin-error transfer uncertain.

## Compute accounting
One thread, <=2GB RSS, ulimit -v2500000. Numerical cycles acquire /workspace/work/codex-compute.lock and use<=110CPU seconds; an authorized review lane may use /workspace/work/codex-review.lock for<=20CPU. Count weight/envelope setup, boundary evaluation, all quadrature nodes and exceptional cases. No full-region production enumeration or cached per-model advice. Checkpoint loading and validation costs reported separately.

## Claims
[inferred] H2's readout admits an exact one-dimensional upper-envelope description and explicit final-input boundary families. [speculative] These families plus a weight-derived history approximation can produce a useful accuracy estimate at a cost scaling with H,T and family count.

## Alternatives not ruled out
Distributed long-term memory, broad interior misclassification, unstable ratio boundaries, positive-mass zero-state ties and a cheaper sampling baseline.

## Failure modes
Reachable-envelope labels need not all be dynamically reachable. A final-input margin denominator can vanish. Recurrent gain bounds may be too loose to certify history truncation. Any first-order gate freezing can fail near a parallel boundary, as R2 shows.

## First weight inventory
`inventory.json` records all six frozen base/noisy cases. H2T4 has only nonzero output labels1 and2, with label0 available only at zero state; label3 is impossible, giving at least25% error before any boundary tilt. Its single ratio threshold is .03765 and final-input boundary is x_last=275.83*h_previous,0+.08523*h_previous,1. The recurrent matrix contains inhibition -130.45, so generic small-history assumptions are doubtful. H2T5 has nonzero labels3,2,1,4 in ratio order, plus zero-state label0. Its three threshold families are stable under the fixed noise, and the readout last-input cancellation ratios are all at least .9996. The second input weight is instead exceptionally small (-2.013e-5) and changes sign under both noisy copies, changing reset/axis behavior. Thus R2's near-parallel readout failure is not the dominant initial explanation here; representation and reset/history families matter.

## Frozen two-input prototype and analytic radial integration
Set the state before the final two inputs to zero, without tuning. Resolve only the two first-input charts, two final gates and the at-mostT-1 positive ratio-envelope boundaries. Their angular sectors have exact labels from weights; this is a fixed-depth boundary engine, not full T-step region enumeration.

Let m=T-2, aU=max(cos(theta),sin(theta)), aL=min(...), and R have the Rayleigh density r exp(-r²/2). Write Fn(a)=E[Phi(aR)^n]. Integration by parts gives Fn=2^-n+n*a/sqrt(1+a²)*J(n-1), where J0=1/2, J1=1/4+asin(a/sqrt(1+2a²))/(2pi), J2=1/8+[2asin(a/sqrt(1+2a²))+asin(a²/(1+2a²))]/(4pi). Thus for m=2,3 no radial quadrature is needed. Angular truth weights are m(F(m-1)(aU)-Fm(aU)) for the larger suffix input, Fm(aL) for the smaller suffix input, and1/m-F(m-1)(aU)+Fm(aU)-Fm(aL)/m for each omitted label. They sum to one over all T labels.

Freeze production angular GL16 per exact two-input sector; GL32 is a validation-only refinement and must not replace16 based on results. Include the true suffix-order angles as breakpoints. First check radial formulas independently by positive radial quadrature and the constant-class truth1/T. For history error use the fixed prefix box |Xi|<=4, ebar=4 sum(r=0..m-1) |Wh|^(r+2)|Wi|, and a winner-margin certificate. At each angle the surrogate state is r*h(theta); sufficient agreement requires r>max_k |Woj-Wok| dot ebar / ((Woj-Wok) dot h(theta)). Exclude nontrivial tied/dead supports from this strict certificate. Its angular Rayleigh mass times(2Phi(4)-1)^m bounds agreement from below in exact integration; the numerical integration of this diagnostic has no interval certificate yet. A vacuous certificate is retained, not treated as a guarantee.

For validation, T4's existing exact-reference ensemble uses absolute sigma .05, RNG47001, eight copies plus base; the initial relative-noise inventory has no matching saved references and remains diagnostic-only. Evaluate all nine existing T4 cases, plus all three prior T5 cases, without selecting by result. Reference files are read only by the scoring script. No new exact enumeration or Monte Carlo. Count all boundary/rule/certificate work; any operation-model advantage remains conditional on the existing reference errors and an independent count review.

Before execution, T55 independently checked all radial formulas and proposed a stronger history certificate from the same ebar. If the final surrogate preactivations are r*a(theta), a dead sector certifies the actual zero state when r exceeds max_i ebar_i/(-a_i). An axis sector certifies the same output whenever both final gate signs persist, r>max_i ebar_i/|a_i|. This includes readout ties on an axis without demanding a strict logit margin. Both-active sectors retain the original margin certificate. Include these exact gate-based cases in the frozen prototype; this refinement uses no validation outcomes.

Before trained evaluation, T55 supplied a stronger nonnegative-state history bound: H=4 sum_r Wplus^r |Wi| bounds prefix hidden; d1=max(Wminus H,Wplus H) bounds the first-tail hidden difference; ebar2=|Wh| d1 bounds final preactivation/hidden difference. Use the componentwise minimum of this proved bound and the original absolute-matrix bound, retain both in diagnostics, and charge the extra setup. This changes no prediction or quadrature order and uses no observed accuracy. Radial independent integration controls already agree within1.12e-16; all constant-label T4/T5 controls agree with1/T within1.28e-15.

## Initial trained prototype result
[measured, operation count pending independent review] The frozen two-input estimator passes the reference-based equal-operation comparison on all12 initial cases. H2T4 base estimate .302287020583 versus .307730743024 reference; the eight existing absolute-noise copies have errors -.00339 to -.00732. Their MSE ratios using the complete proposed92,369-operation count are .0964–.4447. H2T5 base/.0025/.02 estimates .273000219534/.317302821740/.317274692211 have errors -.006237/+.006667/-.007786 and ratios .3004/.4678/.6229 at101,018/146,503/146,503 operations. These counts include history-certificate diagnostics and rule setup; sampling uses the favorable affine-only52/65-operation denominator.

The GL16-to32 difference is at most8.02e-14, so the observed bias is not angular quadrature resolution. Full12-case validation plus refinement uses .831 CPU including .788 checkpoint/input preparation; each estimate uses about .0012–.0019 CPU. No new random model-input samples or region enumeration were used. The history certificate is a clear negative: its numerical agreement diagnostic is0 for T4 and only .0095–.0117 for T5, so it does not certify the small observed bias. The near-zero T5 input-weight sign change increases suffix boundary sectors10→15 and changes the predicted accuracy by4.43 percentage points, qualitatively accounting for the observed3.14-point change but overestimating it. Every case and diagnostic is retained in `benchmark.json`.
