# T80 recovery: verify the negative's scope before extending it

## Question
Which conclusions of the paused E35 negative review are established by its saved evidence, and which require correction or a new bounded rescue? Claim227b201d reserves only this new subfolder. Historical source/results remain unchanged. Claude DRI classifies final claims.

## Method
First inventory all tracked original review/source outputs and missing table entries; pin their hashes and source commit. Do not repeat the prior three-hour experiments. Independently check the claimed covariance-rank theorem, the endpoint “ceiling” implication, and the distinction between empirical cumulants, exact population moments and stochastic endpoint integration. Then choose the most promising genuinely untried, bounded alternative from the evidence rather than declaring a broad family dead.

Freeze analytic controls before evaluation: scalar Gaussian X with h=(X+,(-X)+) has full-rank2 covariance despite one input; compare exact mean/covariance/eigenvalues and support dimension. A second finite rational control uses alternating binomial weights on x_i=i−5/2, i0..5: even-index mass(1,10,5)/16 versus odd-index mass(5,10,1)/16. Independently check identical moments0..4 but unequal positive probabilities; the same independent uniform jitter in[−1/4,1/4] yields continuous laws with those properties. These controls separate valid moment-information limitations from unsupported claims about all trained models or all endpoint encodings. No model inputs, random draws or training for this phase.

Alternative explanations: finite estimator failures can remain valid despite a false mechanism theorem; exact-moment injection diagnoses a fixed endpoint but does not upper-bound all possible propagation/encoding choices; apparent fourth-order divergence can include MC-moment noise or numerical endpoint error. Saved outputs and source resolve what is actually measured before any new experiments.

## Status
Saved-output inventory and frozen analytic controls complete (d71d86b9). Frozen exact-prefix mean discriminator complete; independent source/raw review requested. A full exact-prefix mean/covariance rescue followed by the inherited Gaussian closure is the next separately scoped test.

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
