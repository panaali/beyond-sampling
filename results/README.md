# results/: machine-readable outputs

↑ [Project README](../README.md) · [STATUS](../STATUS.md) · [INDEX](../INDEX.md)

| File | Produced by | Contents |
|---|---|---|
| [summary.json](summary.json) | [../analyze.py](../analyze.py) | Qwen study H1-H3 (E1), see [RESULTS.md](../RESULTS.md) |
| [population.json](population.json) | [../population.py](../population.py) | LLM-free neuron vs random-direction statistics (E2) |
| [algzoo_4_3.json](algzoo_4_3.json), [algzoo_16_10.json](algzoo_16_10.json) | [../algzoo/score.py](../algzoo/score.py) | Held-out legibility scores (E3, E4) |
| [arc_eval_sigma0.0025.json](arc_eval_sigma0.0025.json), [arc_eval_sigma0.005.json](arc_eval_sigma0.005.json) | [../m16/arc_eval.py](../m16/arc_eval.py) | Probe estimator vs sampling under ARC's protocol (E7) |
| [stratified_vs_sampling.json](stratified_vs_sampling.json) | [../m16/stratified_eval.py](../m16/stratified_eval.py) (reproduction) | Stratified sampling comparison (E9; corrected ratios in the INDEX ledger, H-E3) |
| [v4_noisy.npy](v4_noisy.npy) | [../m16/estimate.py](../m16/estimate.py) | Second-order probe estimator on noisy copies |
| [leaderboard_truths.json](leaderboard_truths.json) | [../m16/leaderboard.py](../m16/leaderboard.py) | Ground truths for the 18 leaderboard models (orig pooled from 28M samples) |
| [leaderboard.json](leaderboard.json) | [../m16/leaderboard.py](../m16/leaderboard.py) | Leaderboard rows (pass 1 and v2, keyed `v2\|method\|group\|budget`) |
| [leaderboard_run1.log](leaderboard_run1.log), [leaderboard_v2.log](leaderboard_v2.log) | [../m16/leaderboard.py](../m16/leaderboard.py) | Human-readable leaderboard logs (LB2 summary in [../STATUS.md](../STATUS.md)) |
| [leaderboard_truths.log](leaderboard_truths.log), [leaderboard_truths_v2.log](leaderboard_truths_v2.log) | [../m16/leaderboard.py](../m16/leaderboard.py) | Truth computation logs |
