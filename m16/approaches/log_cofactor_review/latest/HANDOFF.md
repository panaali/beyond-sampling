# T179: latest source passes the full finite corpus check

The pinned source is `139a7b9e`, under scopes `54573740` and `7603c96f`. The controls and 256-cone panel were independently reviewed in `292b54cc`. All 35,098 original T131 error cones have now been evaluated; the 266 previously evaluated actual cones are reused once. The full independent raw audit passed in 4cc3fe39: all indices, reference links, reuse, Decimal totals and 288 raw sentinel/reused records were checked. This checks the existing seed-0 corpus, not every E46 model or noisy copy.

The accuracy is **0.9761608075404442**, exactly reproducing the latest DRI result. The saved positive reference is 0.9761608075466027, a difference of −6.16e−12. The largest individual mass error is 1.668e−14 and the sum of absolute errors is 1.018e−11. Every case passes the frozen diagnostic threshold of 1e−10 plus its reference bound, and every observer result matches the unobserved source bitwise. There are no negative masses, negative or zero interpolated determinants, denominator floors, or nonfinite angle arguments. The maximum unit-diagonal drift is 4.44e−16.

These are finite comparisons, not floating-point certificates. There are 9,321 discrepancies larger than the reference's own truncation bound; those bounds omit general floating-point errors. Their sum is 1.042e−10. The maximum relative discrepancy is 9.946e−7. The minimum observed thin-rule quadratic value, 0.3643, is a quadrature-node minimum, not a global radial lower bound.

## Cost and provenance

There are 17,955 Duffy cases and 17,143 log/atan2 cases. Observed native calls comprise 13,440,112 scalar 3×3 determinants, 2,229,402 scalar 4×4 determinants, and 35,098 4×4 inverses. The initial inverse executes before both branches despite the source's “no inverse” commentary. The inherited mass counter, 1,064,714,168, is incomplete; it is not a full-cost result.

Candidate calls take 9.4484 CPU seconds and observer replays take 10.1928 CPU seconds, counting the reused 266 cases once. The eight new shard processes take 29.4599 CPU seconds, including setup, imports, hashing and storage. Earlier control and panel processes take 0.3553 and 0.2460 CPU seconds. Zero per-call timings reflect clock resolution, not free computation. No new geometry, network inputs, or positive-reference integration was run; new candidate mass quadrature calls were run.

## Preserved failures and remaining alternatives

The general thin-cone stress test still fails four cases. For a unit-ray nearly antipodal block cone, true mass approaches 1/8 while the determinant threshold selects a fixed Duffy rule whose estimate tends to zero. At epsilon 2^-24, the source gives 7.668e−8 versus the analytic 0.1249999976. This does not assert that such cones occur in the current solver's initial charts. It disproves a guarantee based on determinant alone. Radial bounds can supply additional real-arithmetic truncation evidence; recover_34 is coordinating that separate check. Recover_78 is taking the native cost audit.

The older `ac321d97` phase remains immutable in `../FULL_HANDOFF.md`: one frozen-tolerance failure, one tiny negative mass, and nine clipped cones. Results do not transfer between versions. The earlier cutoff determinant-count omission is archived and corrected without rerunning masses. Neither the old numerical failure nor the generic stress failure rejects positive integration as a family.

Reproduction commands from this directory are `python controls.py`, `python panel.py`, `python full.py --batch N` for N=0,…,7, and `python summarize_full.py`. Root/Claude DRI registry review is requested; the author does not confirm its own claims.
