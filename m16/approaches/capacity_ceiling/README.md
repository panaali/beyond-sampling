# capacity_ceiling: T234 (accuracy ceiling of the M_16,10 architecture)

**Question.** What accuracy can this exact architecture reach (1-layer ReLU RNN, H = 16, T = 10, no biases, h0 = 0,
Linear(16, 10) readout, 2nd-argmax, x ~ N(0, I_10))? Does accuracy saturate with more data / longer training, and do
the "training-outcome" errors (especially the blind region, <= 1 positive input, P = 11/1024, where M_16,10 errs
~0.83% of all inputs) disappear with the right data?

**Status.** running (owner-requested via DRI; budget <= 60 Colab units; status to DRI at 30 units).

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

## Results
(pending)

## Compute accounting
(pending) Colab units per stage; wall time per run.

## Claims
(pending)

## Alternatives not ruled out
(pending)

## Failure modes
- Colab reclaims runtimes without notice: checkpoints every n/2^i and per-run result files (see ../gpu_cost_law/COLAB.md).
