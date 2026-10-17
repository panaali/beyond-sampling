# T80 recovery: verify the negative's scope before extending it

## Question
Which conclusions of the paused E35 negative review are established by its saved evidence, and which require correction or a new bounded rescue? Claim227b201d reserves only this new subfolder. Historical source/results remain unchanged. Claude DRI classifies final claims.

## Method
First inventory all tracked original review/source outputs and missing table entries; pin their hashes and source commit. Do not repeat the prior three-hour experiments. Independently check the claimed covariance-rank theorem, the endpoint “ceiling” implication, and the distinction between empirical cumulants, exact population moments and stochastic endpoint integration. Then choose the most promising genuinely untried, bounded alternative from the evidence rather than declaring a broad family dead.

Freeze analytic controls before evaluation: scalar Gaussian X with h=(X+,(-X)+) has full-rank2 covariance despite one input; compare exact mean/covariance/eigenvalues and support dimension. A second finite rational control uses alternating binomial weights on x_i=i−5/2, i0..5: even-index mass(1,10,5)/16 versus odd-index mass(5,10,1)/16. Independently check identical moments0..4 but unequal positive probabilities; the same independent uniform jitter in[−1/4,1/4] yields continuous laws with those properties. These controls separate valid moment-information limitations from unsupported claims about all trained models or all endpoint encodings. No model inputs, random draws or training for this phase.

Alternative explanations: finite estimator failures can remain valid despite a false mechanism theorem; exact-moment injection diagnoses a fixed endpoint but does not upper-bound all possible propagation/encoding choices; apparent fourth-order divergence can include MC-moment noise or numerical endpoint error. Saved outputs and source resolve what is actually measured before any new experiments.

## Status
Saved-output inventory and frozen analytic controls complete (d71d86b9). Frozen exact-prefix mean discriminator complete; independent source/raw review requested. The separately frozen exact-prefix mean/covariance rescue is complete and improves but does not fix the same closure. Independent T55 source/raw audit2827c813 passes after scope/mean reviewa8f7532d; no family-wide exhaustion claim.

## Results
Preliminary source findings, not a new experiment: prior README claims rank(Cov h_t)<=td, which fails for nonlinear functions of td inputs. Low-dimensional piecewise-linear support is the valid distinct statement. The README also contains unfinished placeholders and claims “Dead (verified)” for untried variants. Saved noisy M16T4 cell-readout K4 reports+0.20pp, whereas its broad best-case prose says>=2.7pp; inventory must retain model-specific distinctions.

See SCOPE_AUDIT.md for the corrected theorem, finite moment-information control, completed subset table and unverified alternatives. All42 unique printed failures remain preserved; no old model calculation repeated.

## Compute accounting
One thread2GB; review processes<=20CPU, primary<=110CPU only if justified later. This phase uses deterministic analytic/rational controls and file analysis, not Gaussian model probes. The analytic control CPU clock returned0.0 at available process-time resolution; this is an unresolved sub-resolution duration, not free computational work. No old3GB queue is resumed. Any later estimator/reference/source-dependent work gets a separately frozen scope and full count.

## Claims
[inferred] Deterministic latent dimension alone does not bound covariance rank after nonlinear maps. [inferred] A fixed approximate endpoint evaluated on estimated true moments is a diagnostic, not a universal accuracy ceiling. Existing empirical failures are preserved pending source/raw audit; no broad success or impossibility claim.

## Alternatives not ruled out
Higher-order/positivity-preserving endpoints, structure-aware mixtures, different conditioning, corrected support propagation and uncertainty from empirical cumulants. Existing tested variants retain their exact scope; no claim that every alternative is promising or cheap.

## Failure modes
Printed/rounded results may lack raw replicates; Monte Carlo cumulants are not exact; endpoint sampling may add uncertainty; noisy-copy definitions and fixed seeds can differ from current protocol. A mathematical counterexample to an explanatory statement does not erase measured estimator errors.

## Exact second-step discriminator
Scope5dc6f275/protocol04778174 fixes one historical H16T4 seed3 checkpoint (SHA7470f434...) and16 controls. With p=ReLU(Wi), n=ReLU(-Wi), a+=Wh p, a-=Wh n and b=Wi, the exact-real second-step mean is [(hypot(a+,b)+a+)+(hypot(a-,b)+a-)]/(2 sqrt(2pi)). Negative-a terms use b²/(hypot(a,b)-a) to avoid cancellation.

All16 controls pass an independent50-digit angular integral. Two naive-formula control evaluations round a positive mean to zero; both are retained in prefix_controls.json. All16 trained means pass an angular reference whose coefficient dot products use exact rational representations of the original binary checkpoint. Every input coefficient is nonzero and every exact mean is positive. Units0,10,13 have means .0001583230354, .008062322542, .005034181112, versus Gaussian-closure means4.047129407,10.296001786,9.986759031. Thus the rare-activation correction does not rescue Gaussian closure: its relative RMS mean error is1.494186402, consistent with the old finite-MC diagnostic1.502.

Reproduce from this folder: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/clean-neurons/.venv/bin/python check_prefix_mean.py controls` and the same command with `trained`, each under codex-review.lock and20CPU/2GB limits. Controls used .085839155 CPU seconds; the trained/reference process .940963829 seconds and200012KiB peak RSS. Named kernel units are1096 arithmetic+64 comparisons+16 negations+32 hypot=1208; hypot internals, loop/index/memory work, constant setup and exact-rational/MP reference work are separate. Kernel CPU itself was below clock resolution, not free. No trained forward passes, random inputs or final-accuracy estimate were used. These are ordinary floating evaluations of an exact-real formula, not outward arithmetic certificates or a competitive full-estimator claim.

## Frozen exact-prefix covariance rescue
The separately published scope0c20b0b5/protocol8b49ff31 tests one historical H16T4 seed3 model. `prefix_covariance.py` splits both first-input sign halves at the exact-real second-step gate angles, then integrates the resulting linear maps' first and second moments in polar coordinates. Its binary64 implementation uses midpoint-rotated angular moments and a small-angle series. `check_prefix_covariance.py` independently evaluates each pair's angular antiderivative at50digits using exact original-binary coefficient dot products. Six frozen zero/diagonal/opposite/duplicate/mixed controls pass; tiny negative synthetic covariance eigenvalues down to−2.95e−17 are retained as rounding, with no PSD repair. The trained model uses34 sectors; maximum mean/covariance discrepancies are8.88e−16/2.84e−14. This is a high-precision comparison, not a floating enclosure proof.

`continue_closure.py` then compares the unchanged inherited Gaussian closure at all four steps with the same closure restarted from those exact-real h2 moments. The frozen reference uses262144 Gaussian inputs in8blocks of32768, seed20261017080. There is no final-accuracy evaluation or new model/variant selection.

| Step | Original relative mean RMS error | Exact-h2 restart | Original full-covariance relative RMS error | Exact-h2 restart |
|---|---:|---:|---:|---:|
|2|1.496842|.002602|9.072071|.004649|
|3|3.890865|1.065742|35.901727|6.878440|
|4|5.663295|2.955124|56.822947|20.974102|

The near-zero step2 errors are finite-reference fluctuations, not error bounds on the analytic primitive. At step4 the paired eight-block mean-error reduction is2.708156 with descriptive SE.003323; covariance-error reduction35.847725 with SE.082626. Thus the reset materially improves this fixed approximation but leaves very large errors. Gaussian closure can fail again immediately after exact two-step moments; this bounded rescue is not a successful full estimator and does not exhaust support-aware mixtures or other moment encodings. All outputs, per-unit values and block moments are saved; no failures were discarded.

Reproduce checks with `check_prefix_covariance.py controls` then `trained` under review.lock/20CPU, and continuation with `continue_closure.py` under compute.lock/110CPU, same interpreter/thread environment as above. Controls used.037132CPU; trained/reference moment validation1.025251CPU; continuation1.066441CPU including imports/model load, with.131912 referenceCPU. Candidate trace timings were.004005/.004043CPU, descriptive only; rule setup occurs at import. Reference charges1048576 Gaussian scalar generations and1124064768 named major forward/moment operations, plus explicit unpriced RNG internals, divisions/aggregation, statistics, dispatch and memory. Exact-prefix source operation structure is recorded per sector; no complete portable cost or hardware bound is claimed. The independent50digit/rational reference work is separate, never free estimator work.

Independent review2827c813 recomputes all256 saved pair references and six controls, checks six representative pairs by separate60-digit direct positive angular quadrature (agreement≤4.43e−48 with the high-precision references), and reconstructs every pooled/eight-block error and paired-SE table. It does not rerun the random-input reference or certify downstream closure.
