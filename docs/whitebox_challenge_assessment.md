# Could Beyond Sampling help with ARC's White-Box Estimation Challenge?

**Yes: several methods are worth adapting, but none of our existing results demonstrates success on this benchmark.** This is a source-based assessment, completed 21 October 2026 (T472); no challenge models were evaluated. Our numerical results remain in the [results ledger](../INDEX.md), with their existing qualifications.

The [announcement](https://www.alignmentforum.org/posts/Kben8CzS4awCwNw5c/announcing-the-arc-white-box-estimation-challenge) describes a separate challenge from AlgZoo. It explicitly permits sampling and hybrid estimators. That removes the unresolved sampling-versus-deduction classification issue facing some of our AlgZoo candidates. It does not establish their accuracy or efficiency on the new networks.

## What changes

| | Our main AlgZoo problem | White-Box Estimation Challenge |
|---|---|---|
| Model | Trained recurrent network, 432 parameters | Random, untrained, bias-free ReLU feedforward networks |
| Quantity | Probability of correctly locating the second-largest input | Expected post-ReLU neuron activations under Gaussian inputs |
| Current documented size | 16 hidden units, 10 scalar input steps | Phase 2: 1,024-dimensional input, 16 layers of width 1,024 |
| Main test | Error relative to sampling at equal project-accounted compute; mechanistic interpretation matters | Final-layer squared error weighted by metered computation |

Sources: [AIcrowd overview](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026), [problem setup](https://github.com/AIcrowd/whest-starterkit/blob/96e308c518c76a1ae119f1d4c8913bc2e019eee4/docs/concepts/problem-setup.md), [round specifications](https://github.com/AIcrowd/whest-starterkit/blob/96e308c518c76a1ae119f1d4c8913bc2e019eee4/docs/reference/rounds.md). The documented architecture has 16,777,216 weights (calculated as 16 × 1,024²). This is a useful scale increase, but is neither a trained language model nor evidence of language-model safety.

The [scoring specification](https://github.com/AIcrowd/whest-starterkit/blob/96e308c518c76a1ae119f1d4c8913bc2e019eee4/docs/concepts/scoring-model.md) uses, per network,

`final-layer MSE × max(0.1, used FLOPs / allowed FLOPs)`.

Lower is better. Making an estimator cheaper below the 10% floor does not further improve its cost multiplier. The round specification gives Phase 2 a budget of 2^41 FLOPs. The interface returns means for every layer, although the score uses the last layer. Our existing F ratios and operation counts cannot be substituted for this score.

## What transfers, and what does not

| Finding or method | Assessment |
|---|---|
| Exact line and great-circle integration (H-E17, H-A3) | **A genuine mathematical connection.** Bias-free ReLU networks preserve the piecewise-linear and positive-homogeneous structure we exploit. Along a line we can integrate affine pieces; along a circle we can integrate the corresponding trigonometric pieces. But depth and width may make finding all gate changes too expensive. Our trained small-network wins do not establish scalable cost. |
| Structural approximation plus sampled residual | **A useful new experiment, not an established project win.** Approximate the network with something whose expectation is known, then sample its remaining error. This can reduce sampling variance if the approximation is sufficiently correlated and cheap. Our previous linear proxy and near/far control-variate results (T380) warn that plausible approximations can fail. |
| Moment and correlation propagation (E35 and its reviews) | **Directly relevant background.** Our adaptation struggled on scalar-input recurrent networks. That is not a reason to reject it on the wide random feedforward networks for which ARC developed the method. The source controls and unresolved negative-result qualifications still matter. |
| Mechanism-derived second-largest formulas (E36/T63) and gap walks (H-L1) | **No direct transfer.** Their ordering and near-tie assumptions describe the trained task, not arbitrary neuron means in a random network. Their value here is the strategy of exploiting structure, not reusable formulas or transferred speedups. |
| Rare-event estimation, repair, and certificates (H-R1/H-R3/H-C1, T462/T463) | **Mostly indirect.** Average activation is not a rare-failure probability. Altering target weights would change what must be estimated. Reliability checks and complete compute accounting transfer; bounded-indicator confidence formulas do not automatically apply to unbounded activations. Full region enumeration is not a sensible first bet at this size. |

These are applicability judgments, not newly confirmed hypotheses. Exact radius averaging is also available: for Gaussian input `X = R U`, independence and positive homogeneity give `E f(X) = E[R] E[f(U)]`. However, our [existing literature review](../lit/novelty_review.md) already records spherical cubature and radial factorisation by challenge participants. We must not present that observation as new. In 1,024 dimensions radius concentration also limits how much radial averaging alone is likely to gain.

## The baseline we must beat

The pinned problem-setup document reports a covariance-propagation score of about `4.1e-7`, versus roughly `1.2e-6` for ordinary sampling: approximately a threefold advantage. These are **starterkit-reported results**, not our measurements or a claim about the best leaderboard entry. Merely beating ordinary sampling would therefore not demonstrate a competitive contribution.

The [allowed-code contract](https://github.com/AIcrowd/whest-starterkit/blob/96e308c518c76a1ae119f1d4c8913bc2e019eee4/docs/concepts/allowed-code.md) requires numerical work through the metered interface. Our NumPy/SciPy and compiled walkers need substantial porting; their existing costs cannot be carried over. A prototype must account for setup, fitting, proposal construction, and evaluation under the actual contract.

## Recommended bounded pilot

First reproduce the supplied sampling, mean-propagation, and covariance-propagation baselines on a small, frozen development subset. Pin the starterkit and meter versions, and hold separate networks back for evaluation. Do not launch a full benchmark before checking runtime and memory within our local resource limits.

Then test one inexpensive structural surrogate with an analytic Gaussian mean. For a fixed surrogate `g`, use

`estimate = E[g(X)] + average(f(X_i) - g(X_i))`.

For example, a shallow ReLU surrogate has analytically integrable neuron means. Whether it captures enough of a deep random network is the central uncertainty. An approximate covariance-propagation mean is **not** an exact expectation of an arbitrary surrogate: substituting it without a correction introduces bias. If fitting uses samples, separate fitting and residual-evaluation samples and charge fitting costs. Charge every surrogate evaluation too.

Two explanations must be separated: the surrogate may explain little variation, or it may explain variation but cost more than the samples it saves. The cheapest diagnostic is residual variance together with metered total cost; the deciding test is the held-out official score against the strongest reproduced baseline. The covariance baseline's bias may also create an opening, but that must be measured rather than assumed.

Keep exact line integration as a second, gated experiment: measure gate-piece counts and metered cost on shallow controls and a bounded slice of the actual architecture before committing to a port. If that pilot exceeds its budget, record a scoped feasibility result and the untested alternatives, not a universal impossibility claim.

This pilot is recorded as **T473** in the [backlog](../TASKS.md). It has not been run. My recommendation is a small feasibility study, not a diversion of the whole project or a promise of a winning entry.

## Source status and limits

The starterkit was inspected on 21 October 2026; GitHub's API resolved `main` to `96e308c518c76a1ae119f1d4c8913bc2e019eee4`. Its README identifies Phase 2 and flopscope 0.12.3. Pin and recheck both before experiments, since billing and rounds can change. External-source exposure is recorded in [the literature log](../lit/web_log.md).

The official [challenge-rules page](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/challenge_rules) returned a heading but no substantive rules in both browser extraction and downloaded HTML. Submission deadlines, eligibility, and prize conditions were therefore **not verified**. The current documented phase is not proof that submissions remain open. No registration, submission, or organizer contact was made.
