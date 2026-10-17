# split_attribution — T195: which recipe108 splits drive cell growth along T
**Question.** recipe108 (frozen per-cell recipe, eps .1, `estT_fix`) is accurate at (8,4)/(8,5) but its cells grow
x13-250 per extra step. T187: an input's influence on the final margin follows its rank, not its age. **H:** the growth
comes mainly from gate splits triggered by low-rank inputs (rank >= 4 in the sorted input list), which a rank-conditioned
bound could prune.

**Alternatives written down before measuring (2026-10-17).**
- (a) splits (and class-changing mass) are spread evenly over trigger ranks (no rank structure to exploit);
- (b) growth is driven by the top-3 near-tie geometry itself (thin tie bands, H-E12): triggers are mostly ranks 1-3 and
  class-changing mass sits in cells near x_(1)=x_(2) / x_(2)=x_(3);
- (c) growth is driven by a few "clock" neurons (time-since-max modes) regardless of trigger rank.
Separating tests: (a) vs H: share of splits / class-changing mass by trigger rank vs the uniform share 1/T per rank;
(b): mass share with trigger rank <= 3 and its growth along T; (c): concentration of splits by neuron and whether the
top neurons' rank distribution matches the pooled one.

**Status:** running.
