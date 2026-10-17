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
