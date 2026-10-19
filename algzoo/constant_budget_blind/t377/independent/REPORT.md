# T377 independent rescore

Checker code was written from scratch and does not import any project scoring code. `score.py`, `analyze*.py`, `pipeline.py`, `score_seed.py`, `fragility.py`, `fragility.json` and `results/` were opened only after `rescore.json` was saved (step 5).

## Files
- `rescore.py`: accuracies, gaps and fragility F. Uses 200,000 N(0,1) sequences, input seed 4242, noise seed 777, float64 model, chunks of 20k. Writes `stage1.json`.
- `stage2.py`: leave-one-model-out (LOO) regressions, within-size Spearman and judge D. Writes `rescore.json`.
- `compare.py` (output in `compare.out`): compares this check with `results/summary377.json` and `fragility.json`.

Reproduce with: `source /home/user/clean-neurons/.venv/bin/activate && ulimit -v 2500000 && OMP_NUM_THREADS=1 python rescore.py && python stage2.py && python compare.py`

## Method notes
- h_10 = `model.rnn(x.unsqueeze(-1))[0][:, -1]`. Replacement logits = `W @ [f_0..f_{H-1}](x)`, with no bias. Target = argsort position of the 2nd-largest input.
- F: every neuron gets N(0, 0.1 var(h_i)) noise at once, mean over 3 draws. This matches `F90` in the published files. The fragility.py docstring confirms the definition, which I read only at step 5.
- D = mean item score (mean of the 2 judges, SAME/PARTIAL/DIFFERENT = 1/0.5/0) over matched items, minus the same over control items, per `model` field. All 224 pair items had exactly 2 votes, and no IDs were unknown.

## Results (mine)
| model | H | acc | F | g_bar | gaps |
|---|---|---|---|---|---|
| M8 | 8 | 0.4282 | 0.5623 | 0.5808 | agsc .5824, gcap .5605, wfpn .5994 |
| M16 | 16 | 0.9485 | 0.1735 | 0.3353 | hnrt .3045, wduk .3390, yyuc .3623 |
| M16_s0 | 16 | 0.9099 | 0.2469 | 0.2901 | bnic .3917, rtiv .1885 |
| M16_s1 | 16 | 0.8797 | 0.2602 | 0.2750 | briy .2882, kzuy .2619 |
| M16_example | 16 | 0.9533 | 0.2485 | 0.4219 | pqid .5234, phwb .3203 |
| M32 | 32 | 0.9946 | 0.6467 | 0.8431 | dncx .8244, krvc .7772, sfbp .9277 |
| M32_s1 | 32 | 0.9935 | 0.6126 | 0.7826 | htmm .8315, rdaq .7337 |
| M32_s2 | 32 | 0.9914 | 0.4525 | 0.6144 | vbya .7373, incg .4914 |
| 160-call arm (M32) | 32 | | | | dqfw .8279, vnna .8203 |

LOO RMSE:
- 7 models (H in {16, 32}): size 0.1111, F 0.0752. The verdict is **fragility dominates**, with ratio 0.677 against the 0.7 threshold.
- All 8 models: size 0.2918, F 0.0935. The verdict is **fragility dominates**.

Spearman(F, g_bar): -0.40 within the 4 H=16 models, and +1.00 within the 3 H=32 models.

Judge D:
| model | D |
|---|---|
| M16_example | 0.7813 |
| M16_s0 | 0.6875 |
| M16_s1 | 0.7656 |
| M32_s1 | 0.8672 |
| M32_s2 | 0.8984 |

## Comparison with the published results
- **Gaps, g_bar, F, model accuracy:** no difference exceeds 0.02. The largest |diff| is 0.0044 (wduk gap). The largest F difference is 0.0016. These differences fit fresh-sample and noise-draw variation.
- **D:** all 5 values match exactly, as do S_matched and S_control.
- **LOO:** the RMSEs agree to within 0.001, and both verdicts agree ("fragility dominates" in both regressions).
- **Spearman:** the signs agree, and the values are identical (-0.4 for H=16, +1.0 for H=32).

## Caveats (my reading)
- In the 7-model regression, "fragility dominates" passes narrowly: RMSE_F / RMSE_size = 0.677 here and 0.670 published, against a 0.7 threshold. A small change in a model or an account could flip it to "not separable". The 8-model result is robust (ratio 0.32), but it is driven by M8, whose small size goes with high g_bar.
- The within-size Spearman rank correlations use n=4 and n=3, so they carry almost no statistical weight. Within H=16 the sign is negative, which does not support fragility as the explanation inside one size.

## Summary
The independent rescore reproduces T377: every gap, g_bar and F is within 0.005, every D is identical, both LOO verdicts agree ("fragility dominates"), and the within-size Spearman signs agree (-0.4 for H=16, +1.0 for H=32). The only qualification is that the 7-model verdict sits close to its threshold.
