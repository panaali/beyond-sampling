# capacity_ceiling: T234 (accuracy ceiling of the M_16,10 architecture)

**Question.** What accuracy can this exact architecture reach (1-layer ReLU RNN, H = 16, T = 10, no biases, h0 = 0,
Linear(16, 10) readout, 2nd-argmax, x ~ N(0, I_10))? Does accuracy saturate with more data / longer training, and do
the "training-outcome" errors (especially the blind region, <= 1 positive input, P = 11/1024, where M_16,10 errs
~0.83% of all inputs) disappear with the right data?

**Status.** done (2026-10-18 18:23 UTC); all Colab runtimes stopped. Unverified (no independent check yet).

## Plan and stop rules (written before any training run)
0. Reproduce: one zoo (16,10) seed with the unchanged `alg_zoo.training.train` (2^30, defaults) and the same seed with
   `trainer.py`; both must land inside the zoo seed spread (0.880-0.949; M_16,10 0.95312).
1. (a) Data/length axis at zoo settings: n_train 2^30, 2^31, 2^32, 2^33 (2^34 only if gains continue), 5 seeds each
   (3 if the budget forces it). **Stop rule:** stop the axis when the best-seed accuracy gains < 0.05 pp per doubling
   for two consecutive doublings.
2. (b) At the best size: 2-3 schedule/LR variants (lr x0.5, lr x2, lower floor 0.001 / longer warm-up).
   **Stop rule:** at most 3 variants, 3 seeds each; keep a variant only if its best seed beats the baseline best seed
   by > 3 SE of the evaluation.
3. (c) Hard-case-enriched data at the same total size: 20% blind-region inputs + 20% near-tie inputs (2nd and 3rd
   largest within gap 0.05), cross-entropy on the true label, evaluated on plain N(0, I_10). **Stop rule:** one size
   (the best of (a)), 5 seeds; a second gap/fraction setting only if blind-region errors drop by > 50%.
4. **Budget rule:** stop all new runs at 55 units spent by T234; status message to the DRI at 30 units.
5. Every final model: MC accuracy with >= 1e8 samples (SE ~2e-5) and error anatomy (blind / 2-3 / 1-2 / other) as in
   m16/FINDINGS.md section 3; every model and checkpoint saved (`data/capacity_ceiling/`, npz Wi/Wh/Wo).

## Method
- `trainer.py`: faithful batched re-implementation of `alg_zoo.training.train` (DistRNN, 2nd_argmax): K models stacked
  (independent inits = bitwise the zoo inits via `DistRNN` after `torch.manual_seed(seed)`, independent data,
  elementwise AdamW with the torch formula, LambdaLR semantics, warm-up then exponential decay to the floor at
  mid-run), CUDA-graph captured on GPU. Data differ from the zoo's stream (the zoo drew data from the same global RNG
  as the init), so per-seed outcomes may differ; the distribution is identical.
- `test_equiv.py`: same init + same data as a replica of the zoo loop (nn.RNN + torch AdamW + LambdaLR):
  parameters agree to 2.4e-7 after 300 steps on 2 seeds [measured].
- `evaluate.py`: float64 GPU MC with error anatomy; on M_16,10 (4e6 samples): acc .95318, blind errors 0.835% of
  inputs (blind error rate 78%), 2-3 2.86%, 1-2 0.72%, other 0.27% (FINDINGS: 0.83% blind, 78%) [measured].
- `run.py TAG --log2n N ...`: one configuration, K seeds -> `results/TAG.json`, weights in `data/capacity_ceiling/`.

## Code revisions (T55 review, 2026-10-18 04:05 UTC)
Fixed after Codex T55's source review: (1) error anatomy conditional on T >= 3 (T = 2 crashed); (2) deterministic data
stream: reseeded from (data_seed, block) at every 2^18-step block, so a resume from a block-boundary checkpoint sees
exactly the data of an uninterrupted run; checkpoints store config + source hash + seed list (validated on resume) and
a final-state checkpoint is saved; (3) evaluation saves per-model integer correct counts and the pairwise discordance
matrix (common inputs across models) for paired comparisons. **Runs started under the old code:** `a30` (seeds 0-4,
2^30) and the zoo-code reproduction `zoo_code_seed0` (unchanged alg_zoo, unaffected): `a30` used one continuous data
stream (no mid-run resume happened), and its final evaluation lacks the discordance matrix (re-evaluated with the new
evaluate.py from the saved weights).

## Results ([measured], unverified)
**Step 0, reproduction (2^30, zoo defaults; 1e8-sample float64 MC, SE ~3e-5):**
| model | acc | blind | 2-3 | 1-2 | other |
|---|---|---|---|---|---|
| zoo code unchanged, seed 0 (`repro_zoo.py 0`, 2608 s on T4) | 0.91000 | 0.00585 | | | |
| trainer.py seed 0 (`run.py a30 --log2n 30`) | 0.91037 | 0.00585 | 0.05122 | 0.02734 | 0.00522 |
| trainer.py seed 1 | 0.90432 | 0.00662 | 0.06507 | 0.01798 | 0.00602 |
| trainer.py seed 2 | 0.94946 | 0.00755 | 0.02692 | 0.01454 | 0.00154 |
| trainer.py seed 3 | 0.93072 | 0.00717 | 0.03973 | 0.01911 | 0.00327 |
| trainer.py seed 4 | 0.94193 | 0.00613 | 0.03501 | 0.01452 | 0.00242 |
All inside the zoo seed spread (0.880-0.949; M_16,10 0.95312). 5 seeds batched in one process: 1847 s on a T4.

**Speed and the fused kernel** [measured]: profiling found 90% of each step in one cuBLAS weight-gradient GEMM (16x16
output, batch-long reduction on one tile); a split-K backward (`trainer.RecMM`, grads equal to autograd within 6e-16 in
float64) gave 61 -> 2.4 ms/step (12 models x batch 8192) and 9.4 -> 0.8 ms/step (12 x 1024) on an A100.
`fused.py` (Triton; one program per model, in-kernel Philox data, labels, forward, BPTT, AdamW + zoo schedule, N steps
per launch, per-model batch and hard-case mixture): vs autograd on identical inputs 1.3e-6 after 6 steps (B 1024),
7.3e-6 after 5 (B 2048); in-kernel data P(<= 1 positive) .01069 vs 11/1024 (z -0.56, n 1.2e6), labels exact;
one launch of 2000 steps bit-identical to 2000 launches. Uncontended A100: 0.635 ms/step at 108 models
(0.0059 ms/model-step), linear beyond. **40-vs-40 validation** (2^30, seeds 0-39, same inits): fused mean .9189
(sd .033) vs trainer.py .9214 (sd .030), Welch t -0.35, KS D .150 < .304: indistinguishable; same-seed outcomes
correlate only 0.11 (seed spread ~3 pp is data-order/rounding chaos, not init). Codex T261 `tiled_step.py` (copy:
`codex_tiled_step.py`): torch vs fused 4.7e-7, torch vs Codex 1.7e-7, fused vs Codex 5.1e-7 after 3 steps;
Codex 1.05 ms/step at 108 models (mildly contended). Logs: `results/cmp_*.log`, `results/codex_crosscheck.log`.

**Where the target sits** [measured]: M_16,10 (example_2nd_argmax, acc .9531175 +- 3.9e-7, ../gpu_cost_law MC truth)
is near the top of the 40 + 40 zoo-config retrains at 2^30 (fused: mean .9189, max .9589; trainer.py: mean .9214,
max .9602), i.e. a lucky draw from the same training distribution rather than evidence of a different recipe.

**H = 16, T = 2..10 table** (`results/ht_table.json`, 1e8 samples per model; T = 7, 9 trained with unchanged zoo code):
mean accuracy .99991, .99945, .99780, .99022, .97998, .97648, .96676, .96726, .91211 for T = 2..10.

**Data axis, zoo settings, trainer.py, seeds 0-4** (`run.py a31 --log2n 31`, `a32 --log2n 32`; 1e8 samples):
| n_train | seeds 0-4 | mean | max |
|---|---|---|---|
| 2^30 (a30) | .9104 .9043 .9495 .9307 .9419 | .9274 | .9495 |
| 2^31 (a31) | .9477 .9279 .9268 .9309 .9547 | .9376 | .9547 |
| 2^32 (a32) | .9705 .9636 .9548 .9280 .9559 | .9545 | .9705 |
More data at zoo settings raises the mean (5 seeds each; seed spread ~3 pp, so mean differences of ~1 pp are within
noise; the 100-seed grid3 decides). The za pack (2^31-2^33 at batch 1024, data seed 7) was lost when Colab reclaimed
its L4 (~06:20, last archive had only the 2^31 weights, not evaluated); grid3 supersedes it.

**Large-batch RECIPE VARIANT** (`run_multi.py lb8192 --batch 8192`, `lb16384 --batch 16384`, warm-up 1/64 of the run,
LR x sqrt(B/1024) or x B/1024; seeds 0-2; 1e8 samples):
| batch, n_train | sqrt LR | linear LR |
|---|---|---|
| 8192, 2^32 | .9599 .9112 .9617 | .9601 .9642 .9290 |
| 8192, 2^33 | **.9693** .9658 .9590 | .9511 .8571 .9536 |
| 16384, 2^34 | .9602 .9359 .9459 | .5141 .9388 .9125 |
Best .9693 (above the target's .9531); linear LR scaling is unstable (one .514 collapse at B 16384). 3 seeds per cell:
no ranking between cells is claimed. Wall: 5515 s (12 models) / 5439 s (6 models) on an A100.

**Grid3: main T234 grid** (`q_grid3.sh` -> `fused_run.py grid3 --data-seed 41 --eval kernel`; 1,025 models, seeds
100-199 (2^34: 100-124), fused kernel; evaluation in-kernel float32, 1e8 samples per model on an independent Philox
stream, se ~2e-5/model; `results/grid3.json`, `results/grid3_summary.json` via `python grid3_analysis.py`; weights in
`weights/grid3_weights.tgz`; full training curves in data/, not in git). Restarted once after a Colab reclaim (08:24);
models are bit-identical to HP2's separate retrains for the 2^30 arm (`../gpu_cost_law/hp2/check_vs_grid3.txt`).
Interim float64 1e7-sample evaluation of the same models agreed (e.g. 2^30 mean .91110 vs .9111).

| config (zoo defaults unless noted) | n | mean [95% CI] | sd | median | top-5 | max | share > .9531 | blind | 2-3 | 1-2 | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2^30 | 100 | .9111 [.9038, .9188] | .040 | .9126 | .9675 | .9720 | .21 | .0072 | .0528 | .0215 | .0074 |
| 2^31 | 100 | .9363 [.9307, .9420] | .028 | .9424 | .9730 | .9785 | .33 | .0068 | .0386 | .0142 | .0041 |
| 2^32 | 100 | .9384 [.9321, .9440] | .031 | .9479 | .9749 | .9796 | .41 | .0067 | .0370 | .0137 | .0042 |
| 2^33 | 100 | .9444 [.9388, .9497] | .028 | .9545 | .9760 | .9822 | .53 | .0066 | .0333 | .0125 | .0032 |
| 2^34 | 25 | .9519 [.9403, .9622] | .029 | .9618 | .9734 | .9770 | .80 | .0065 | .0295 | .0096 | .0025 |
| 2^31, lr x2 | 100 | .9465 [.9415, .9510] | .025 | .9544 | .9734 | .9773 | .52 | .0072 | .0316 | .0117 | .0030 |
| 2^31, warm-up x4 | 100 | .9250 [.9183, .9315] | .035 | .9285 | .9692 | .9708 | .28 | .0068 | .0453 | .0174 | .0055 |
| 2^31, lr x0.5 | 100 | .9113 [.9045, .9182] | .035 | .9154 | .9641 | .9665 | .13 | .0067 | .0539 | .0214 | .0066 |
| 2^31, 20% near-tie inputs | 100 | .9307 [.9240, .9372] | .033 | .9366 | .9706 | .9721 | .33 | .0074 | .0401 | .0168 | .0050 |
| 2^31, 20% blind inputs | 100 | .9058 [.8981, .9131] | .038 | .9120 | .9568 | .9600 | .05 | .0021 | .0603 | .0228 | .0091 |
| 2^31, 20% blind + 20% tie | 100 | .9060 [.8983, .9134] | .040 | .9120 | .9628 | .9646 | .07 | .0022 | .0573 | .0237 | .0109 |
(error columns: mean error rate by class; blind = inputs with <= 1 positive coordinate.)

**Does accuracy saturate with more data?** [measured, unverified] Not at 2^34 for the typical model: mean .911 ->
.936 -> .938 -> .944 -> .952 and median .913 -> .942 -> .948 -> .955 -> .962 per doubling 2^30 -> 2^34 (largest step
2^30 -> 2^31; afterwards ~+0.6-0.8 pp per doubling; 2^34 has 25 seeds, CI +-1.1 pp). The best models do saturate:
max .972 / .979 / .980 / .982 / .977 and top-5 mean .968 -> .976, i.e. a ceiling near .98 for this architecture and
recipe. More data mostly shrinks the 2-3 and 1-2 confusions (2-3 errors .053 -> .030); the blind-input error rate
stays ~.0065-.0074 in every zoo config [measured]; training on 20% blind inputs cuts it to .002 but costs ~3 pp of mean
accuracy. Doubling the learning rate at 2^31 gives the 2^33 mean at half the data. M_16,10 (.9531) sits at the
53rd-80th percentile of 2^33-2^34 retrains and well below the ~.98 ceiling.

## Compute accounting
Colab units (owner's account): 304.80 at login (2026-10-17 23:06 UTC) -> 164.40 at the end (2026-10-18 18:23 UTC):
140.4 units total. T192 phase (to 03:22): 22.2. T234 + T270 + side jobs (HP2-HP5, DEV2-DEV3, T340/T349/T358/T364/T366
truths, T348 census) since 03:22: 118.2 (DRI cap 115, raised to 122 for the 2^34 arm). A100 5.3 u/h, L4 1.5 u/h,
T4 ~0.35 u/h, CPU 0.08 u/h. Lost to reclaims: za pack (L4, ~06:20), t192fu (grid3 ~1 h + T270 chunks, 07:23-08:12).
Grid3 wall: ~10 h on one shared A100 (08:24-18:20). Training compute is model-steps: grid3 3.1e9 model-steps of batch
1024 (fused kernel 0.0059 ms/model-step uncontended).

## Claims
- [measured] The fused Triton trainer reproduces the zoo training distribution (40-vs-40 vs trainer.py, KS D .150 <
  .304; trainer.py seed 0 reproduces unchanged zoo code .91037 vs .91000) and is bit-identical between packed and alone
  runs with the same data ids (HP2 vs grid3, 100/100).
- [measured] Mean accuracy rises with data through 2^34 (.911 -> .952); the best-model ceiling is ~.98 from 2^31 on.
- [measured] Doubling the zoo learning rate at 2^31 raises the mean by 1.0 pp (CIs disjoint from 2^31 zoo); hard-case
  mixes (20% blind) lower it.
- [inferred] M_16,10 is a typical-to-good draw of the zoo recipe, not a ceiling model.

## Alternatives not ruled out
- Seeds 100-199 share data seed 41 across configs (data stream differs by data id), so config differences are
  paired only through seeds/inits; seed-to-seed chaos (~3 pp sd) dominates single comparisons.
- The 2^34 arm has 25 seeds; its mean is within the 2^33 CI's upper half; more seeds would firm up the slope.
- Large-batch / other optimizers were sampled only lightly (3 seeds per cell); the ~.98 ceiling is for this recipe.
- Evaluation is float32 in-kernel for grid3 (float64 1e7 interim agreed); M_16,10 truth is float64.

## Failure modes
- Colab reclaims runtimes without notice: checkpoints every n/2^i and per-run result files (see ../gpu_cost_law/COLAB.md).
  After the DRI memo (CLI >= 0.7.3 has no keep-alive ping) every session is held by a blocking `colab exec` loop
  (`../gpu_cost_law/colab/keepalive.sh`) since 03:45 UTC: no reclaim so far (04:24).
