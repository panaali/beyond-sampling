# T24 independent rescore

I wrote this check from scratch. I did not read `algzoo/score.py`, `analyze.py`, `sensitivity.py` or `score_runs.py`. I opened `results/summary.json` only after `rescore.json` had been written.

## Code (this folder)
- `recon_check.py`: loads each `runs/<code>/labels.py` with importlib and builds `zoo_2nd_argmax(H,10)`. It draws 200,000 N(0,1) sequences with **seed 77123** (numpy `default_rng`, chunks of 20k, float32 inputs, the same inputs for every run). It computes model accuracy, replacement accuracy (argmax of `W_out @ f(x)`, where `W_out = model.linear.weight` and there is no bias), gap recovered = (repl-0.1)/(model-0.1), and per-neuron R^2 = 1-SSE/SST against `h_10`. `h_10` comes from `model.rnn(x.unsqueeze(-1))`. Output: `recon.json`.
- `judge_check.py`: parses `judging/key.json` and `judging/raw/*.txt` and computes the rest: pair scores S_matched, S_control and D per size; Cohen's kappa between the two judges on primary ROLE; legibility (judge CAT in {A,B} and my R^2 >= 0.8); G; and the frozen verdict rules. Output: `rescore.json`. It also checks for missing, duplicate, extra and invalid judge lines and for neuron coverage. None were found: there are 168 role items, covering every neuron once, and 168 matched plus 168 control pair items.
- `compare.py`: compares `rescore.json` with `results/summary.json`. Output: `compare_out.txt`.

Run with: `source .venv/bin/activate && ulimit -v 2500000 && OMP_NUM_THREADS=1 python recon_check.py && python judge_check.py && python compare.py`

## My numbers
| run | H | model acc | repl acc | gap | median R^2 | #R^2>=0.8 | legibility (j1, j2) |
|---|---|---|---|---|---|---|---|
| agsc | 8 | 0.4264 | 0.2919 | 0.588 | 0.767 | 2 | 0.25, 0.25 |
| gcap | 8 | 0.4264 | 0.2840 | 0.564 | 0.772 | 3 | 0.375, 0.375 |
| wfpn | 8 | 0.4264 | 0.2970 | 0.604 | 0.758 | 2 | 0.25, 0.25 |
| hnrt | 16 | 0.9490 | 0.3597 | 0.306 | 0.950 | 15 | 0.8125, 0.8125 |
| wduk | 16 | 0.9490 | 0.3853 | 0.336 | 0.937 | 15 | 0.875, 0.9375 |
| yyuc | 16 | 0.9490 | 0.4072 | 0.362 | 0.934 | 14 | 0.8125, 0.8125 |
| dncx | 32 | 0.9943 | 0.8358 | 0.823 | 0.900 | 26 | 0.78125, 0.78125 |
| krvc | 32 | 0.9943 | 0.7948 | 0.777 | 0.904 | 24 | 0.6875, 0.6875 |
| sfbp | 32 | 0.9943 | 0.9287 | 0.927 | 0.918 | 25 | 0.6875, 0.71875 |

| size | G | S_matched | S_control | D | legibility |
|---|---|---|---|---|---|
| 8 | 0.5879 | 0.9167 | 0.2604 | 0.6563 | 0.2917 |
| 16 | 0.3360 | 0.9531 | 0.2396 | 0.7135 | 0.8438 |
| 32 | 0.8227 | 0.9792 | 0.0938 | 0.8854 | 0.7240 |

Inter-judge Cohen's kappa on primary ROLE: **0.9063** (n=168).

Verdicts:
- Reconstruction is **mixed**. G_32 = 0.823 is below 0.90 but above G_8 - 0.05 = 0.538.
- Labels **hold**. D_32 = 0.885 is at least D_8 - 0.15 = 0.506.
- Overall: **mixed/inconclusive**.

## Comparison with results/summary.json
The full table is in `compare_out.txt`. **No number differs by more than 0.01.**
- All judge-derived quantities match exactly: S_matched, S_control, D, the role kappa and the legibility fractions per run and per size.
- The integer counts n(R^2>=0.8) match exactly.
- Seed-dependent numbers differ by small amounts: accuracies by up to 0.0013, gaps by up to 0.0043 (gcap), median R^2 by up to 0.002, and G by up to 0.003.
- **All three verdicts are the same** (mixed / holds / mixed/inconclusive).

No discrepancies were found. I skipped the D CI check, as instructed. It cannot change the labels verdict here, because the point estimate D_32 = 0.885 is far above the 0.506 threshold.
