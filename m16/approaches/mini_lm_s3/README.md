# T446: mini-LM S3, rare bad outputs: estimate, locate, repair, rate

Task row: T446 in [TASKS.md](../../../TASKS.md). Spec: [STREAMS.md section S3](../mini_lm/STREAMS.md) (branch
`claude/t433-mini-lm` d45e9d21). Harness (read-only): `../mini_lm/{load,models,langs,metrics}.py`.
Session: Claude child session_01Mp2LCFeNhs7doqRpuyhaHo. Branch `claude/t446-mini-lm-s3`.

## Pre-registration (written 2026-10-19, before any estimator was scored against a truth)

Development so far (disclosed): the estimators below were debugged on `dyck2_rnn_s0` at tau .1 and .2 against its
exact L = 15 truths (about 60 runs). That model stays in the scored set; the debugging informed the CE schedule,
the CE features and the AMS score. Nothing was tuned on any other model.

### Target and truths

- Rare event B_tau = P(max_t p_illegal(x, t) > tau) on dyck2 (harness definition: p_illegal = the model's mass on
  the wrong closer at depth > 0, on both closers at depth 0). Note: p_illegal at position t depends only on the
  prefix y[:t], so the problem is a prefix tree of depth L - 1.
- **L = 15 truths are exact** (`exact15.py`: full enumeration of the 7,365,120 sequences, float64 weights; the
  per-sequence max is saved, so B at any tau is exact). B0.5 reproduces the T433 table (pos_s0 6.21e-6).
- **tau rule (fixed now).** Grid: 400 values of tau, logit-spaced in [1e-3, 0.999]. For each of the 18 dyck2
  baselines, two points: the grid tau whose exact B is closest in log10 to 1e-6, and the one closest to 1e-8,
  each kept only if its B lies in [1e-9, 1e-5] (if both pick the same tau, one point). The truths set the rarity
  level only (as STREAMS.md does); they are not used to select estimator settings.

### (i) Estimate

Cost: 1 FPE = one forward pass over one length-L sequence; a prefix-tree node at position t is charged
(flops(t+1) - flops(t)) / flops(L) (the KV-cache rate), so a full sampled sequence costs 1 FPE. All CE stages,
pilots and lookahead nodes are charged. C_eff = max(C, 1000).

Methods (all in `est.py`, frozen at the commit of this pre-registration):
- `ce`: **cumulative cross-entropy-fitted twisted language law + IS** (sampling). Proposal: a multinomial-logit
  choice among {open same type as the top, open the other type, close} with 25 state features (depth one-hot,
  t / L, top type, same-type stack share, 2nd-entry type, distance to the top's opener, recent closes, same-type
  run length), mixed 5% with the true law (defensive). 10 stages on half the budget, level = 0.9 quantile capped
  at tau, refit on the elites of all stages; 2 refinement stages at tau; final IS on the other half, fresh samples.
- `top`: **top-risk enumeration** (deductive head + sampled tail). CE stages on 30% of the budget fit Q; the
  prefix tree is expanded best-first by log Q(prefix) for 35% of the budget, every child is evaluated and closed
  with its exact language mass if its running max exceeds tau; the last 35% samples fresh paths from Q and counts
  LR x 1[event] only for paths leaving the head through an open frontier node. Unbiased.
- `ams`, `amsl`, `amsce`: **generalised adaptive multilevel splitting** (Brehier et al. 2016; kill all at or below
  the level, so unbiased with ties), 10% killed per level, score = max over positions of a per-position rarity
  transform (calibrated on a charged pilot) of (a) p_illegal (the spec's running max), (b) a close-all lookahead,
  (c) the CE-fitted log Q / P of the prefix. Development on rnn_s0 already showed (a) and (b) going extinct at
  1e5 FPE; they are scored anyway (3 replicates) so the negative is on record.
- Plain MC: F = 1 by definition (not run; it is the reference).

Scoring: ONE stated budget **1e5 FPE** (primary), with 5 replicates (independent seeds) per (model, tau) point for
`ce` and `top`, 3 for the AMS variants; secondary budgets 1e4 and 1e6 (`ce`, `top`, 3 replicates). Per
architecture group (3 models x up to 2 taus), F_rel = mean over runs of e^2 C_eff / (p (1 - p)) and a
bootstrap 95% CI over models (resampling models, then their runs). **Strict: F_rel <= 1 (point); lenient: upper
CI <= 10.** **Within 2x:** the share of runs with estimate / truth in [0.5, 2] over all runs (>= 80% passes). Both
bars and the within-2x share reported for every method, group and budget.

Sealed test (after the freeze above, when the DRI says MLP1 is pushed): the same frozen code and tau rule on the
MLP1 panel (seeds 100-119) at L = 15 with exact truths (fresh models never seen), then L = 32 on >= 10 MLP1 models
with long-MC truths (plain MC at >= 100x the scored budget, only points with >= 100 truth events), pre-registered
separately before those runs.

### (ii) Locate

Smoothed objective B_s = E[sigmoid((logit max_t p_ill - logit tau) / 0.25)]. Estimated gradient wrt all weights:
the CE-IS final samples (1e5 FPE run), mean of LR x grad sigmoid. Exact gradient at L = 15: all sequences with
logit(max p_ill) > logit(tau) - 4 (the rest contribute < e^-16 each, the truncated mass is reported). Models: the
s0 seed of each architecture, at its 1e-6 point. **Pass if** cosine(estimated, exact) >= 0.9 and the top-12 weights by
|g_i| overlap the exact top-12 in >= 8, on >= 4 of 6 models.

### (iii) Repair

On the same 6 models / points. Accuracy surrogate A = mean next-token excess KL on 2e4 language samples (charged).
Weights S = the top 12 by g_i^2 / (F_ii + lambda), F = diagonal empirical Fisher of A, lambda = median F_ii;
direction delta_S = -(H_SS + lambda I)^{-1} g_S with H_SS the exact 12 x 12 Hessian of A (autograd). Step: the
smallest of ||delta|| in {.01, .02, .05, .1, .2, .5, 1, 2} x ||theta_S|| whose CE-IS estimate (1e5 FPE, the
twist refitted for the edited model) gives B ratio <= 0.05 and whose paired 2e5-sample closer-accuracy and Q_ill
estimates are not worse than -.02 pp / +2 s.e. (estimates only; no truth is used to choose). Verification: exact
L = 15 B_tau, closer accuracy and illegal mass of the edited model. **Pass if** exact B ratio <= 0.1, closer
accuracy cost <= .05 pp, Q_ill not worse, and <= 2 of 40 random controls (same 12 weights, Gaussian random
direction, same L2 norm, exact L = 15 scoring) pass the same three tests. Counted per model; the spec's
criterion is met by one passing model; "transfers" is claimed only for >= 4 of 6.

### (iv) Rate

r(k) = sum_t P(dist_t >= k and p_ill_t > tau_r) / sum_t P(dist_t >= k), dist_t = positions back to the matching
opener of the top of stack (depth > 0 only), k in {2, 4, 6, 8, 10, 12}, tau_r = the largest of {.5, .1, .01}
with a nonzero exact r at all 6 k. The denominator is model-free (exact language DP, free); the numerator is
estimated from the weights by `ce` with the event "some position with dist >= k has p_ill > tau_r" and the
per-sequence count as the IS integrand (1e5 FPE per k). Check: the exact L = 15 r(k) (stronger than the
spec's conditional MC). **Pass if** within 2x at >= 4 of 6 k, on >= 4 of the 6 s0 models.

### Addendum A (2026-10-19 23:2x, before any MLP1 model was scored or its truth computed): sealed tests on MLP1

MLP1: claude/t433-mini-lm 5f043812 (merged here), `models_mlp1/`. Per the DRI, seeds 100-109 are S2's; S3 uses
only the held-out **seeds 110-119** (60 models). Code frozen at the commit of this addendum (est.py as of
ebd59776 plus the s3lib loader for MLP1).
- **Sealed L = 15 (exact truths).** All 60 models: `exact15.py`, the same tau rule (`taus.py`), `S3_PLAN=sealed
  score15.py`: `ce` and `top` at 1e5 FPE, 3 replicates each. Same bars (F_rel strict / lenient per architecture
  group, 10 models x up to 2 taus each) and within-2x share (>= 80% passes). This is the primary transfer test of
  the estimators on fresh models.
- **L = 32 (plain-MC truths).** 12 models: seeds 110 and 111 of each architecture. Truth: 4e7 fresh plain-MC
  samples per model (`mc32.py`, seed 1, = 400x the 1e5 budget, 40x the 1e6 budget). tau: the grid tau whose MC
  B is closest in log10 to 5e-6 among those with >= 100 MC events and B <= 1e-5 (one point per model; the truth
  carries about 10% binomial error, reported). `score32.py`: `ce` and `top` at 1e5 FPE (5 replicates; primary),
  `amsce` at 1e5 (3), `ce` and `top` at 1e6 (3; secondary, truth only 40x the budget). Same bars and within-2x.
  Deeper L = 32 tails (B < 1e-6) cannot be checked by MC at this compute; any estimate there is unvalidated.
- Locate / repair / rate stay at L = 15 on the baselines (sections (ii)-(iv)).

## Question

Do the project's rare-event results on M_16,10 (estimating rare failures, locating and repairing them with a small
weight edit, rating safe use) transfer to a tiny next-token language model with discrete inputs?

## Method / Status / Results / Compute accounting / Claims / Alternatives / Failure modes

Filled in after scoring (below the pre-registration, which stays unchanged).
