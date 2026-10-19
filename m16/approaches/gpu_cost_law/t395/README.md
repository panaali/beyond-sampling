# t395: 1e10-input census of error-carrying and near-tie gate patterns of M_16,10 (T395, extends T348)

**Question.** How many distinct gate patterns (activation regions) carry the error mass and the near-tie mass of
M_16,10 at prefix lengths k = 7..10, how much of that mass do the observed patterns cover, and how much is unseen
(Good-Turing)? T348 (1e9 inputs) could not resolve error patterns beyond k ~ 7-8 (coverage .96 at k = 8, .86 at k = 10);
every "dead" verdict on exact routes for rare events leans on this growth.

**Status.** PRE-REGISTERED 2026-10-19 (written and pushed before any T395 run). Owner: Claude session_016BGedpds19gq95yjWmDMn5
(T192 child), DRI session_01CD3g9Cvc3qk3pM11eyXXzj. Cap: 10 Colab units (on top of T352). No ARC contact, no Fable.

## Pre-registration (fixed before any result)
- **Model and inputs:** M_16,10 (`m16/arc_eval.py` weights), x ~ N(0, I_10), 1e10 inputs (CUDA Philox, one seed,
  regenerated identically in every pass), float64 forward, gate word w_t = 16 bits of h_t > 0 (as T348).
- **Sets:** error inputs (argmax(Wo h_10) != position of the 2nd-largest x) and near-tie inputs (min(g12, g23) < .1);
  plus all inputs for validation.
- **Patterns:** prefix (w_1..w_k) for k = 7..10 on error and near-tie sets; prefix k = 4..6 on all inputs (validation);
  64-bit order-dependent hash as T348 (exact 64-bit key for k <= 4); collision check with a second independent mixer at
  k = 10 on the error set.
- **Counting:** GPU sort + unique in hash partitions sized to the GPU memory; per (set, k) the full count-of-counts is
  SAVED (not only f1), so coverage-adjusted estimators can be computed afterwards.
- **Reported per (set, k):** n, distinct D, singletons f1, doubletons f2, Good-Turing coverage C = 1 - f1/n (unseen mass
  1 - C), Chao1 richness D + f1^2 / (2 f2), plug-in exp(H) and the Chao-Shen coverage-adjusted entropy exp(H_CS),
  N(delta) = fewest patterns covering 1 - delta of the set's sample mass for delta in {1e-1, ..., 1e-6} where
  delta * n >= 10 (flagged unresolved where 1 - C > delta / 3), and per-step growth factors of D, exp(H), exp(H_CS),
  N(delta).
- **Validation (pass/fail recorded):** k = 4, 5 all-input exp(H) and N(delta) vs the exact enumeration
  (`../results/prefix/k{4,5}_regionmass.npz`): |relative difference| <= 1% where resolved; k = 6 distinct D <= exact
  region count 19,740,596 (T270, a lower bound itself) and N(delta) vs T270's region masses (provisional: T270's masses
  are lower bounds for defective cells).
- **Decision reading (as framed by T346):** error-pattern growth per step (exp(H_CS) and resolved N(delta)) <= ~3x ->
  exact sub-cone enumeration revives for rare events; ~25-44x -> exact routes dead; in between -> report as such.
- **Compute plan:** A100 (float64), partitions P chosen so a pass holds <= ~25 GB of keys; estimate ~1 h of A100
  (~5.3 units) + setup; stop at 10 units and report the k reached.
