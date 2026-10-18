# T231 independent saved-mass review

Question: do the twelve archived prefix-mass arrays reproduce the owner's exact saved-data coverage counts, normalizations and finite growth summaries?

Status: checker prepared; source review and immutable completed-input pins required before the sole run. Existing delegation7316c33f/cc00dba5; parent T231 is owned by h4. Source preparation only: no archive decoded, author imported, network or mass calculation performed by this reviewer.

Method: check.py uses a standard-library NPY reader, direct IEEE754-bit integer units2^-1074 and a maximal-omitted-mass ascending sweep. This differs from the owner's rational conversion and descending coverage scan. PROTOCOL.md freezes all12 datasets, the two threshold conventions from the fixed source protocol, and resource/durability boundaries. Protocol/source hashes and eventual raw-input hashes are kept separate.

Claims: source preparation is not a passed saved audit. Exact contractions concern the normalized stored discrete measure, not true Gaussian masses. Prefix-cylinder containment gives monotonicity of ideal coverage counts only under true consistent partitions; it does not force observed multiplicative growth, complete accuracy-estimation cost, or an all-method impossibility.

Compute accounting: no numeric review execution yet; hashing/source inspection only. Future single process19/20CPU2GiB with17CPU internal guard. No author/model/geometry/mass/RNG calls.

Alternatives and failure modes: integration errors may redistribute probability without changing total; approximate histories may omit/merge real regions; historical model/binary provenance is not recovered by matching counts. Accurate aggregation can avoid enumerating enough histories for full probability coverage. NumPy source-report arithmetic remains source-qualified; exact saved integer arithmetic is independently checked. Any discrepancy or resource stop is retained, not repaired automatically.
