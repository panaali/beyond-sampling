# T197 bounded statistics-reporting repair

The original evalH loop records cumulative E75.STATS after every-eighth rescaling check. The prepared evalH_stats.patch snapshots the integer dictionary immediately after the original evaluation and reports cells/cut/resolved from that snapshot. Shared source is untouched.

Independent source gate a2465c24 and saved audit6c3bdd74 support this narrow change. Controls212252f5 preserve all six frozen cases/53 row pairs,128 fake estimator calls,106 fake truth calls and22 rescale calls; author0.016294550CPU, no actual estimator/model/truth/random calls.

The original clean count297 disproves universal halving; noisy9/8 division is likewise not exact. Five matching native H4 clean cell records174,127,148,150,175 are pinned as candidate reused evidence, but their current full independent contraction remains separate. Full H4/H5/H6/H8 historical noisy statistics are not reconstructed. Do not substitute scalar rec112 topology or newer noise80..83 for original native noise70..73.

DRI decisions: adopt the narrow shared source patch; separately decide whether exact historical statistics reconstruction is worth a new cost-first scope. This bounded handoff does not authorize that replay, correct all scaling tables, prove a cost-growth lower bound or retire alternative methods. Preserve original summaries and first-index/counter qualifications.
