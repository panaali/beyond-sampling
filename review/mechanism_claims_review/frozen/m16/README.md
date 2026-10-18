# m16/: the 432-parameter AlgZoo RNN M_16,10

↑ [Project README](../README.md) · [STATUS](../STATUS.md) · [INDEX](../INDEX.md) · [FINDINGS](FINDINGS.md) ·
[Approaches](approaches/README.md)

The narrative of everything learned here is [FINDINGS.md](FINDINGS.md). Claims and their verification status are in
the [INDEX hypothesis ledger](../INDEX.md#4-hypothesis-ledger) (H-M*, H-E*).

## Shared infrastructure (DRI-owned; agents must not edit)

| File | Purpose |
|---|---|
| [common.py](common.py) | Weights of `example_2nd_argmax()`, `run()`, sampling helpers |
| [arc_eval.py](arc_eval.py) | ARC protocol harness: noisy copies, `true_acc`, estimator interface |
| [leaderboard.py](leaderboard.py) | **DRI leaderboard v2**: 18 models, cached truths, 24 reps, bootstrap CIs. Results in [../results/leaderboard_v2.log](../results/leaderboard_v2.log) |
| [truth.txt](truth.txt) | 20M-sample ground truth for the original model (pooled 28M value in [../results/leaderboard_truths.json](../results/leaderboard_truths.json)) |

## Mechanism analysis (FINDINGS sections 1-4, 7)

| File | What it does |
|---|---|
| [state.py](state.py), [features.py](features.py) | Discrete state (p1, p2, case) and order-statistic features |
| [frozen.py](frozen.py) | Gated-linear view; rank-canonical inputs (99.96%) |
| [modes.py](modes.py) | Eigenmodes of the most common gate set (the "clock"; see ledger H-M1/H-M2) |
| [separable.py](separable.py) | Separable position code h ~ m2·U[p2] + (m1-m2)·V[p1] |
| [exact.py](exact.py) | Exact per-input linear logit map (error decomposition) |

## Estimators run by the DRI (FINDINGS section 5)

| File | Estimator | Verdict |
|---|---|---|
| [estimate.py](estimate.py) | Probe estimators v2-v4 (deductive) | Bias +0.74 → +0.25 pp; ~10-50x worse than sampling |
| [lineint.py](lineint.py) | Exact integration along coordinate lines | Unbiased, ~17x worse |
| [qmc_test.py](qmc_test.py) | RQMC, raw and line-integrated | No gain |
| [stratified.py](stratified.py), [stratified_eval.py](stratified_eval.py) | Mechanism-informed stratified sampling | 2.1 / 1.8 / 1.4x on model evals (verified); less with generation cost |

Later attempts by agents: [approaches/README.md](approaches/README.md).
