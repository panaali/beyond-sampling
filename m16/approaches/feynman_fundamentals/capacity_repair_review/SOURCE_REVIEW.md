# T249: independent repaired T234 source review

Author `d2fdf6bd`, prior source `3e5e09a1`, prior independent review `cb7dd705`. Eleven text blobs are pinned; exactly README.md, trainer.py, run.py and evaluate.py change. No author imports, checkpoint/result/array decoding, model, training, evaluation, RNG, GPU or Colab calls. This reviews the correction's control flow, not an executed study.

## Corrections supported by source

**T2 anatomy:** the only third-rank access is now guarded by T>=3; otherwise third is the sentinel -1. Predictions lie in 0..T-1, so the third-rank error count is zero for T2. Outside the blind event, the sole wrong label for T2 is the largest, so the existing other-category predicate also yields zero. This removes the previously identified index failure for the intended T>=2 domain. No general input-validation claim follows.

**Resume metadata:** checkpoint metadata is exactly compared with current config, seed list and SHA256 of the concatenated trainer.py/run.py bytes before loading state. Parameters, both AdamW moments and the completed-step tensor are restored. Old checkpoints without metadata and changed arguments/source are refused; this also deliberately refuses a changed horizon or evaluation-only argument. It does not silently migrate historical state.

**Block streams:** BLOCK=2^18. Fresh execution reseeds block0 before its first step. After completing block b, the uninterrupted path saves state then reseeds for block b. The resumed path restores completed step b*BLOCK and reseeds with that same block number before the next step. Subject to identical generator/backend behavior and identical training operations, both paths therefore begin the same next block with the same parameters/moments/step and seed. The old saved-step-dependent stream restart is removed. Logging occurs every2^14, which divides BLOCK, so accumulator resets align with these periodic checkpoints.

This is a control-flow argument for the intended block schedule, not a tested CUDA bitwise-continuation certificate. Custom graph generator registration/default-generator fallback, graph replay offsets, libraries, device and precision must still be fixed and tested. No CUDA capture-step-offset defect is asserted: the earlier reviewer inference was withdrawn.

**Final state:** the save predicate is now block boundary OR final step, with temporary-file replacement. For an unchanged completed run, start=final_step+1 makes the training loop empty. A final non-block checkpoint does not need to reconstruct its mid-block RNG position because the identical config permits no remaining training. Evaluation and final-output writes still repeat; immutable completed-attempt behavior is not introduced.

**Evaluation counts:** per-model correct is returned as an integer, and row k of discordant_correct_vs contains D[k,j]=number(k correct,j wrong) on common inputs in this invocation. The double matrix product sums 0/1 products; at the fixed default chunk2^22 all exact integer sums are representable in binary64. For a consistent saved table, diagonal entries are zero and D[i,j]-D[j,i]=correct[i]-correct[j]. The difference-of-proportions is (D[i,j]-D[j,i])/n and its empirical second moment uses (D[i,j]+D[j,i])/n. Thus the retained table supports paired comparisons within that evaluation call. No actual counts or matrix were decoded here.

## Remaining limits and concrete missing evidence

1. Paired rows cover the models passed together, ordinarily the seeds of one configuration. Separately evaluated training configurations are not jointly evaluated merely because seed777 is reused; no cross-invocation discordance is retained. Cross-arm paired inference needs a joint comparison or appropriate saved outcomes. Best-seed/multiple-arm selection still needs its declared inferential treatment.
2. src_hash binds trainer.py and run.py only. It does not bind evaluate.py, alg_zoo initialization, library/runtime versions, checkpoints' original generation or final weight bytes. config includes requested device and evaluation n but not every backend setting. This is an improvement, not complete provenance.
3. Integer correct/discordance are retained; the other anatomy counts remain proportions. The anatomy still classifies wrong output rank, not numerical rank-gap bins. MC uncertainty remains distinct from training-seed uncertainty and architecture capacity.
4. ht_table.py and q_ht.sh are unchanged. Broad loader exceptions/partial seed coverage and unconditional HTDONE remain. A table completion marker is not an exit-status/coverage audit.
5. The README says old a30 was uninterrupted and its weights were re-evaluated with the new evaluator. Those are owner reports, not independently verified historical facts in this review. New source does not retroactively alter an old run. No finite T2 or uninterrupted/resumed control record is retained in the reviewed source delta.
6. Timer still excludes Trainer initialization/warm-up/capture and final evaluation, and prior resumed invocations. Total resource and stopping decisions remain external owner accounting. No budget-excess claim follows.

## Disposition

The specific T2 index access, absent resume metadata, missing final state and absent within-call paired counts are repaired in source. Exact runtime resume, actual new/old evaluations and statistical ceiling claims remain unverified. T249's source phase is complete; its finite executed-control prerequisite remains open for the owner. No source adoption, training stop, new evaluation or rerun is authorized by this review.
