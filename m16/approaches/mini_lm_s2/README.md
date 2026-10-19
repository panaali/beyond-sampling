# T445 (mini-LM stream S2): discrete-input estimators of the illegal-output rate at L = 32

Harness: [../mini_lm/](../mini_lm/README.md), specs: [../mini_lm/STREAMS.md](../mini_lm/STREAMS.md) (S2).
Session: Claude child session_01Jc3vr4TPRVxxwK9wJAkC1Y. Branch `claude/t433-mini-lm`.

## Pre-registration (written 2026-10-19, before any MLP1 truth or estimator run)

**Panel MLP1.** 20 retrains per dyck2 architecture {rnn, softmax, hard, relu, linear, pos}, seeds 100-119,
`../mini_lm/train.py` unchanged (`../mini_lm/run_mlp1.sh`), checkpoints in `../mini_lm/models_mlp1/`.
Scoring subset: seeds 100-109 (10 per architecture, 60 models); seeds 110-119 stay untouched for S3 / S4 / later
held-out use. Development uses only the 18 dyck2 baselines (seeds 0-2).

**Target.** Q(m) = E_x[Y(x)], Y(x) = (1/32) sum_{t=0}^{31} p_illegal(x, t), x ~ dyck2 law at L = 32 (the
`illegal_mass` metric). Truth: plain MC with 4e6 fresh sequences per model (seed 9,000,000 + model seed), also giving
sigma_m^2 = Var(Y) per model; the truth's own variance sigma_m^2 / 4e6 is subtracted from each squared error.

**Cost.** 1 FPE = `model.flops(32)`. A forward over a prefix of length t is charged `model.flops(t) / model.flops(32)`
FPE. Model-free quantities of the language law (prefix counts, depth-chain tables, stratum probabilities) are free
(precomputed once, listed). Every estimator is charged its pilot runs.

**Estimators** (budget B; all inductive, i.e. sampling-based, except E4):
- **E0 plain MC**: mean of Y over B sequences.
- **E1 exact head + MC tail**: the largest t0 whose head cost count(t0) * flops(t0)/flops(32) <= B/2; the positions
  t < t0 are summed exactly over all count(t0) prefixes with their exact probabilities; the positions t >= t0 are
  estimated from the remaining budget's sequences (1 FPE each). Unbiased.
- **E2 tilted importance sampling**: the proposal is the dyck2 generator with open probability q' (types uniform,
  closers forced); weight = product of per-step likelihood ratios. q' is chosen in {.5, .55, .6, .65, .7} by a pilot
  (2% of B per value, charged, then discarded) minimising the estimated second moment of wY. Unbiased.
- **E3 feature importance sampling over positions**: one item = one position t with its prefix. The exact law of
  (t, d_t, j) (j = position of the matching opener) factorises over the depth Markov chain and is computed by DP
  (free). Pilot (20% of B, charged): items from the law, estimating the second moment of p_illegal per coarse cell
  (d_t = 0 | d_t >= 1 with k = t - j in {1-2, 3-4, 5-8, 9-16, 17-32}). Main: cells drawn proportional to
  P(cell) * sqrt(pilot second moment) (floor: 1% of the cell mass), prefixes drawn from the exact conditional law
  given (t, d, j) via h-transforms of the depth chain; weight = P / proposal. Each item costs flops(t+1)/flops(32).
  Unbiased.
- **E4 (deductive, exploratory)**: rnn only: prefix-tree DP that merges nodes with identical stack content and
  hidden states within eps, swept over eps. No success is predicted; reported as cost-vs-error curve.

**Scoring.** For each estimator and model, R = 5 independent runs at B = 1e4 FPE (primary) and B = 1e3 (secondary).
F_g = mean_m(mean_r e_{m,r}^2 - sigma_m^2/4e6) * C_eff / mean_m(sigma_m^2) per architecture group g (C_eff = max(C,
1e3)), 95% bootstrap CI over models. **Strict: F_g <= 1 (point; significant if upper CI <= 1). Lenient: upper CI
<= 10.** E0's F should be about 1 (control).

**Predictions** ([speculative], scored when done, wrong ones stay): (S2-P1) E1 F in [.5, 1] for every group at 1e4
(the head positions carry little of the variance); (S2-P2) E2 strict at 1e4 for softmax, relu and linear (rare-event
dominated Q) and not for hard; (S2-P3) E3 strict at 1e4 in at least 3 of 6 groups; (S2-P4) E4 does not reach the
strict bar at <= 1e5 FPE.

**Compute plan.** MLP1 training about 4 CPU-h; truths 60 models x 4e6 sequences about 5 CPU-h; estimators
4 x 60 x 5 x 1.1e4 FPE about 1.3e7 FPE, about 4-6 CPU-h. CPU only, 2 heavy processes, `ulimit -v 2500000`.

**Amendment 1 (2026-10-19, before any MLP1 truth or panel run; dev models only seen).** Add **E5 = exact head +
tilted-IS tail**: head as E1 but with head cost <= B/4; tail as E2 (same q' grid and pilot rule, pilot criterion on
the tail positions t >= t0 only). Clarification of E1: the head enumerates the count(t0 - 1) prefixes of t0 - 1
tokens with one forward of input length t0 each (this scores positions 0..t0-1 exactly); the cost rule is
count(t0 - 1) * flops(t0) / flops(32) <= B/2. Dev check on dyck2_hard_s0 and linear_s0 (2e5-sample truths, 4 runs at
1e4): all estimators within noise of the truth; E0 1.48 / 1.15, E1 .90 / 3.46, E2 .75 / .49, E3 3.55 / 3.70
(mse x C / sigma^2; noisy).
