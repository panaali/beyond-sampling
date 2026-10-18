# T231 independent saved-mass review

Question: do the twelve archived prefix-mass arrays reproduce the owner's exact saved-data coverage counts, normalizations and finite growth summaries?

Status: independent saved audit PASS. Source reviewer f66d39f0, published23f76c5d; immutable input/gate d10a4de8/30d93dec published3c3deacb before the sole run. Existing delegation7316c33f/cc00dba5; parent T231 is owned by h4. No author module, network or mass calculation was executed by this reviewer.

Method: check.py uses a standard-library NPY reader, direct IEEE754-bit integer units2^-1074 and a maximal-omitted-mass ascending sweep. This differs from the owner's rational conversion and descending coverage scan. PROTOCOL.md freezes all12 datasets, the two threshold conventions from the fixed source protocol, and resource/durability boundaries. Protocol/source hashes and eventual raw-input hashes are kept separate.

Claims: all12 archives/772,804 stored piece or region masses,126 exact threshold checks and eight within-source growth records pass the independent saved-data audit. Exact contractions concern the normalized stored discrete measure, not true Gaussian masses. Prefix-cylinder containment gives monotonicity of ideal coverage counts only under true consistent partitions; it does not force observed multiplicative growth, complete accuracy-estimation cost, or an all-method impossibility.

Compute accounting: sole review2.331250254 in-main CPU seconds; whole process2.23user+.22system/2.47wall,243,292KiB peakRSS. Caps19/20CPU2GiB with17CPU internal guard. No author/model/geometry/mass/RNG calls; no failures or retries. Owner separately reported2.412955564CPU. The command ran check.py under the review flock and25-second post-lock wall timeout; run1.log and output/ preserve the attempt.

Alternatives and failure modes: integration errors may redistribute probability without changing total; approximate histories may omit/merge real regions; historical model/binary provenance is not recovered by matching counts. Accurate aggregation can avoid enumerating enough histories for full probability coverage. NumPy source-report arithmetic remains source-qualified; exact saved integer arithmetic is independently checked. Any discrepancy or resource stop is retained, not repaired automatically.

## Results and limits

All92 input pins,51 snapshot payloads,28 required owner evidence files and26 owner progress events bind. Every archive is finite/nonnegative; width IDs are dense, GPU packed histories are unique and its scheduled-completion flags are true. Width archives have no complete field. These are saved-schema facts, not exact partition proofs.

At delta=1e-5, the width target sequence is N=34/634/10,233 at k2/3/4. The distinct GPU records give634/10,233/152,956 at k3/4/5. At GPU k5, N is2,548/14,658/42,499/89,216/152,956/226,080 for delta1e-1..1e-6. Both exact-decimal and rounded-one-minus-binary-delta conventions match the stored count reports; predecessor and selected-margin fractions are independently retained for every threshold. Matching counts across constructions do not establish identical masses or runtime/model identity.

Exact binary-bit grouping, totals, extrema, inverse Simpson ratios and ascending omitted-mass cuts reproduce the saved rational results. The separately reported NumPy accumulation kernels were NOT rerun by this reviewer; their report bindings, subtraction identities and discrepancy flags were checked, retaining source-qualified status. No true-mass accuracy, Gaussian coverage certificate, complete numerical cost or future multiplicative growth is established. The prefix-cylinder containment argument supports only nondecreasing ideal coverage counts under valid partitions; accurate aggregate methods need not enumerate enough histories to cover nearly all probability.
