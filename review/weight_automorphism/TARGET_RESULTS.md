# Fixed target exact-readout result — saved review pending

Target source/provenance gate19153f92/shared9191a56a, independently completed
controls4e5eb0b9, immutable executiongate4f1356b0/sharede546379c. Raw result
e88938a0/sharedfb34b3ca preserves payload bytes,160 exact dyadics,120 pair
certificates, six progress records and original launcher log. Independent
saved target review is explicitly assigned0d166612.

Every readout column is nonzero, and every pair has an exact nonzero2x2 minor.
All120 witnesses already use readout rows0 and1. Largest retained determinant
numerator/denominator bit lengths are53/52. Thus no two columns are proportional,
in particular none positively proportional. The proven readout condition
forces identity for any global positive-monomial transformation that leaves
the actual target weights/logits unchanged. This is an exact real-arithmetic
statement about the pinned stored weights, pending independent saved audit.

CPU0.006609040999999996seconds, peakRSS25,740KiB. Partial source counts160
normalization divisions and120 minor evaluations; conversions, comparisons,
rational internals and native work are not a full cost census. Zero checkpoint
reloads, model/forward/geometry/mass/RNG calls. The archive was previously bound
to checkpointSHA498bd9a8 by independent T219/T201 reviews; this phase checks the
archive SHAfea77e7c and raw Wo bytes SHA098c8b07, not a new historical execution.

Original command (completed outputs refuse overwrite):

    flock /workspace/work/codex-review.lock bash -c 'ulimit -t 20; ulimit -S -t 16; ulimit -v 2097152; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; exec /workspace/clean-neurons/.venv/bin/python review/weight_automorphism/target.py' > review/weight_automorphism/target_run.log 2>&1

No conclusion about output-label-only/common-logit-shift invariance, reachable
subsets, noninvertible quotients, approximate symmetries, arbitrary changes of
parameterization or ARC complexity. GLOBAL_RELU_COROLLARY.md only characterizes
the same class under an explicit universal preactivation commutation assumption;
its proof is separately submitted for review. No new numerical extension.
