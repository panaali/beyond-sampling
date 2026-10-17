# Independent comparison of H-M1 through H-M8

This review uses the public example checkpoint (SHA256498bd9a8...6218bb)
and Martin Keane's public fork pinned at
[5bec782](https://github.com/mrsirrisrm/alg-zoo/tree/5bec78272af02cf891345bbdaca490fddb6d22fa).
The source was read on2026-10-17 and is post-January-2026 material.
Exact paths, hashes, reading scope and URLs are in `source_log.json` for
integration into `lit/web_log.md`. Numbered documents below refer to that
pinned fork. This is a selective source audit, not verification of every claim
in its roughly49 numbered documents or execution of its downloaded code.

## Eight-claim comparison

| Project claim, current qualification | Classification | Primary-source comparison and limit |
|---|---|---|
| H-M1: phase tracks age since maximum in several common gates; not a universal forced-trajectory law | **Agree qualitatively; numerical periods concern different operators** | Docs16,32,48 discuss temporal/position encoding, multiple modes and nonnormal transients. Doc32's5.4 is the raw Wh complex-pair period; project8.4 is a principal gated matrix. Our independent spectra reproduce5.37064 and8.42149. The source's finite two-impulse countdown is not an independent check of E29's Gaussian phase regressions. |
| H-M2: harmonic series8.4/4.3/2.9/2; weakened to one common gate and some training seeds | **Not covered quantitatively** | Docs15/16 distinguish raw modes, an isolated n10/11/12 subblock and ten-position DFT patterns; doc16 explicitly withdraws a40% frequency-shift story. These neither verify the gated harmonic ratios nor the gate/seed null tests. Our other gated dominant periods7.7301,6.7155,7.5688,8.4037 reinforce the need to specify a gate, but do not repeat the seed/null experiment. |
| H-M3: n3/n13 have fast-mode geometric leverage; causal higher-harmonic role withdrawn | **Not covered as a causal claim** | Doc06 labels n3 integrator and n13 unclear; doc27 calls both bridges. That vocabulary is compatible with distributed coupling but does not test the project's orthogonal leverage calculation or causal specificity. No new intervention was run. |
| H-M4: near-perfect canonical-spacing accuracy does not establish general correctness | **Agree on restricted-input warning; different canonical sets** | Docs27/32/42's90 clean two-impulse pairs are all correct at M1/S.8, independently reproduced here. They are not the project's full-permutation canonical spacing set. Source27 itself reports low-S and third-impulse failures. Our fixed S.2 and.99 controls produce89/90 and77/90, so100% cannot be transferred across magnitudes, let alone to Gaussian inputs. |
| H-M5: error decomposition by chosen rank/blind region, not proof of measured near ties | **Not covered numerically; qualitative agreement on third-rank errors** | Docs01/12 discuss small second/third gaps and errors picking third; doc01 describes inputs in[0,1], unlike the project's Gaussian law. Docs27's restricted impulses contain no negative-input blind region. Their percentages cannot validate Gaussian0.82/2.14/0.52pp categories. |
| H-M6: group ablations support contributions for13/16 tested neurons; not unique causal roles | **Agree qualitatively on distributed contributions; not an independent replication** | Docs06/16/27 report comparator, wave, bridge and latch roles and different zeroing/readout interventions. Doc27 revises early n2/n4 interpretations and shows mismatched-state rescue failure. These are different inputs/interventions from project mean ablations; neither proves all16 assigned causal roles. |
| H-M7: exact fine-region enumeration is expensive at H16T10, with explicit limits on the claim | **Not covered** | Docs29/35 use tropical/piecewise-linear geometry; docs42/48 discuss effective operators and cancellation. No comparable complete cone counts or charged Gaussian estimator benchmark validates the project's cost claim. Doc48's assertion that approximation is impossible from a small cancellation ratio is not a theorem excluding other representations or aggregate-probability methods. |
| H-M8: full rank order plus positive count does not fix gates accurately enough for a0.1pp estimate | **Agree on gate dependence; threshold/cost claim not covered** | `src/canonical_D_after_second_impulse.py` uses fixed M1/S.8. Translation invariance for gap/direction is automatic for a time-homogeneous zero-bias recurrence with leading zeros. Our fixed-ratio alternatives keep order/positive count unchanged yet alter200/330 or150/330 post-second gate masks. This supports the distinction but does not replicate the94.2vs95.3% frozen-gate accuracy experiment. Weighted mixtures remain open. |

## Raw eigenfrequency is not a forced phase law

Let D_t be the diagonal indicator of positive preactivations. Exactly,

    h_t = D_t Wh h_(t-1) + D_t Wi x_t.

The raw operator Wh, a fixed-gate DWh (whose nonzero eigenvalues equal those
of the active principal block), a time-varying product D_tWh...D_1Wh, and a
forced trajectory are different objects. A constant mode angle follows only
for an appropriate unforced fixed operator/eigenmode, not automatically for
the nonlinear recurrence. Source16 already warns about confusing temporal
frequencies with variation of final states across input positions.

Independent float64 decompositions give raw dominant complex eigenvalue
0.43516446+1.02673452i, angle67.03114 degrees, period5.37064. Mask64975 gives
0.69827523+0.64542883i, angle42.74778 degrees, period8.42149. Its second slow
complex pair has period8.37312; saying merely "the slow mode" is ambiguous.
The isolated n10/11/12 block gives12.29373, matching source16's12.3.
Complete pair lists, including all five frozen gate masks, are retained.
None of these spectra alone establishes the dominant direction on data.

## What the offset symmetry does and does not prove

The source constructs

    Delta(m,s)=h(M at m,S at s)−h(S at m,M at s).

Therefore Delta(s,m)=−Delta(m,s) for **any** deterministic network.
The balanced ordered-pair least-squares objective for Delta≈F_m+G_s,
with F0=G0=0, is invariant under (F,G)→(−G,−F). The design has full column
rank: rows(i,0) and(0,i) determine each nonreference coefficient. Uniqueness
thus forces G=−F, regardless of weights, representation or accuracy.
For r_i=sum_j Delta(i,j), an independent closed form is

    F_i=(r_i−r_0)/n,  G_i=−F_i.

Consequently WoF=−WoG for any readout Wo. This directly limits doc33's
interpretation that its fitted antisymmetry is a precise property learned by
this network. It also does **not** guarantee discrimination sign: take n=2,
scalar F=(0,1), G=−F, and readout rows Wo=(0,1). For m0,s1, the margin
(Wo_s−Wo_m)(F_m−F_s)=−1. Antisymmetry holds but favors the wrong row.

Separability remains nontrivial: every triangle-cycle sum must vanish for
exact Delta(m,s)=F_m−F_s. Our independent90-pair implementation obtains
R²=.91011227, not doc33's reported.89; the discrepancy is preserved, not
resolved by tuning. Maximum triangle-cycle residual6.77851 rules out exact
separability for this finite dataset. Actual offset discrimination is positive
for90/90 pairs (minimum7.90764), a separate measured fact. Neither this sign
test nor its fit alone bounds other output competitors or Gaussian accuracy.

The fixed synthetic antisymmetric control has much poorer R²=.25091279 but
the same fitted symmetry to9.44e−16, demonstrating that the symmetry diagnostic
does not distinguish the learned model. Closed-form/least-squares coefficient
agreement is5.78e−15 on the actual model and6.67e−16 on the synthetic control.
Independent reviewer T55 checked the proof, row-sum solution, triangle-cycle
condition and balanced-weight assumptions in commitf813e13. This is algebraic
review, not an independent rerun of this numerical script. Missing/asymmetrically
weighted ordered pairs or asymmetric fitting constraints would change the proof.

## Stable relative gates versus a magnitude-independent mechanism

Leading zeros keep h=0. Shifting two impulses together therefore shifts their
entire relative trajectory exactly, up to the available horizon, at any fixed
amplitude pair. This is architectural time-translation invariance, not a
new learned ability. Positive homogeneity similarly guarantees invariance under
common positive scaling in real arithmetic. The scale3 control has zero gate
or argmax changes and maximum hidden residual1.07e−13 in float64.

For each S/M in{.2,.8,.99}, all240 comparisons between shifted pairs with the
same signed gap agree. Across ratios,200/330 and150/330 post-second masks
change relative to.8. The source canonical-D script freezes.8; it does not
claim or test magnitude-independent D tables. Doc27 already reports failures
at lower S; the.2 reproduction gives its89/90. At.99 all13 errors pick the
maximum; this lies outside doc27's decreasing-S sweep and does not refute its
scoped claim of zero maximum picks over that sweep.

The canonical countdown holds for330/330 post-second states at.8,329/330
at.2,282/330 at.99. Final100% on90 two-impulse pairs is a useful mechanism
control, but doc42's76+14 pair decomposition is not an accuracy estimate over
the Gaussian benchmark and has no demonstrated equal-compute advantage.

## Alternatives and next useful boundary

Raw/gated-operator mismatch is resolved on the same checkpoint. Unrestricted
phase encoding, seed universality, causal neuron roles and Gaussian risk remain
unresolved here. Input-history phase transport is the distinct existingT27
backlog item: it could explain E29's4–8degree departures using exact gated
products plus input forcing. No new Gaussian sample, phase fit, intervention
sweep or broad negative mechanism conclusion is authorized by this review.
