# T228 fixed saved-only result

One completed attempt under the published source gatebe07096b and execution
gateff4af35a/shared776ebd3c. `results.json`, `progress.jsonl` and `run.log`
preserve the original run. Independent saved-result review3db2d849/shared3677b343 passed all records with
integer-scaled aggregation, insertion-rank marginals and conditional-log entropy
(max alternate entropy difference1.066e-14). No author numerical calls.

All nine NPZ/JSON pairs pass:118,586 pieces, all finite and strictly positive;
280 displayed arithmetic/event-marginal comparisons, maximum absolute discrepancy
1.6086687537608668e-11 against the frozen5e-10 tolerance; all discrete coverage,
per-event99% counts, gate-id and schema assertions pass. Exactly22 progress
records. Author/checkpoint/model/geometry/mass/RNG calls:0. CPU2.189840052seconds,
peakRSS61,024KiB, external20CPU/2GiB review lock.

| Model | k | Modal full-history share | exp H(G\|E) | Event-weighted99% history cover |
|---|---:|---:|---:|---:|
| target | 2 | .339242430835 | 6.22611647090 | 11.5 |
| target | 3 | .236985071309 | 13.9029975654 | 44.8333333333 |
| target | 4 | .205730926318 | 26.0315748990 | 138.625 |
| zoo16_4 seed0 | 2 | .368525577147 | 5.39489920101 | 12 |
| zoo16_4 seed0 | 3 | .266912048479 | 13.0214804924 | 46.3333333333 |
| zoo16_4 seed0 | 4 | .134795501237 | 45.0658906745 | 241.291666667 |
| zoo16_5 seed0 | 2 | .376500614235 | 5.51478839312 | 10 |
| zoo16_5 seed0 | 3 | .283320372609 | 10.8472420303 | 35.8333333333 |
| zoo16_5 seed0 | 4 | .174692244504 | 29.7665284187 | 144.791666667 |

Six prefix summaries and all three symmetry summary files are preserved. The
retained cosine tables contain only17 distinct pair records, versus120 scanned
in the source. Their duplicates/order/maxima agree internally, but the absent
103 records and original weights are not reconstructed. Four paired-MC summary
mean/second-moment constraints are consistent; three variant sample counts are
only prose assumptions. This is not raw-outcome or CI validation.

Reproduction command (original filenames refuse overwrite; do not rerun the
completed attempt):

    flock /workspace/work/codex-review.lock bash -c 'ulimit -t 20; ulimit -v 2097152; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; exec /workspace/clean-neurons/.venv/bin/python review/width16_structure_review/audit.py' > review/width16_structure_review/run.log 2>&1

The unsigned saved arithmetic supports the tested modal-rule no-go, conditional
on source geometry/masses/labels. No sign0 reconstruction, exact Gaussian mass,
historical runtime/model binding, permutation-family exclusion, T222 coverage
claim, final-model accuracy or full-cost inference follows. Source findings and
alternatives are in SOURCE_NOTES.md/PROOF_LIMITS.md; T229 registers the distinct
exact positive-monomial weight-automorphism discriminator without running it.
