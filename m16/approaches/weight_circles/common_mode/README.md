# Common-mode correctness discriminator

## Question
Does a common shift x+c*ones preserve correctness on the existing two T187
trained models? Such invariance is necessary to drop the common input offset
and keep only the natural zero-sum projection. It is not necessary for every
other low-dimensional projection or conditional representation.

## Method
Scope9df81752 precedes implementation; `PROTOCOL.md` freezes all768 saved
witnesses and four analytic controls. `core.py` reconstructs exact symbolic
preactivation/logit rows, intersects the open gate-valid common-shift interval,
and partitions every pairwise logit crossing. First-index ties are preserved.
Only full-vector identical winning rows permit a robust permanent tie.
Exact positive cube radii certify actual stable correctness at each witness.
Two robust opposite probes have the same projected coordinates and certify a
common open neighborhood of projections, avoiding a null-line-only inference.

## Status
Source/theory independently cleared3ee0667b. Four controls pass4b3886c1 in
0.004423146CPU, independently saved-reviewed443d4875. Trained driver505220d3
was separately source-gated443d4875 and bound by dd9260be. All768 saved-map
contractions completed743fd576; independent full saved audit c84d99e8 passes. No dense
network forward or reference draws; symbolic map and affine probe work is charged.

## Results
Analytic H2T3 crossing case has gate interval(-1,infinity), true label1 and
margin1-c; correct status changes atc=1 while earlier label0 owns that tie.
The invariant control and allzero permanent tie show no change. Distinct
full rows tied only on a common line correctly receive no robust certificate.
The original circle-law phase is separate, in the parent folder.

All768 original matrices/denominators/gate histories reproduce exactly. Robust
opposite-correctness pairs occur in100/384 H4T4 and2/384 H8T4 fixed witnesses.
All observed open-interval correctness changes satisfy the robust criterion.
These are not Gaussian frequencies or coverage estimates. Remaining rows do not
prove global invariance, and no alternative projection is excluded.

First H4 witness index5 starts at(-1,2,-3,4), target1: common shifts approximately
-1.60696065 and-0.59273206 give predictions3 and1, with exact certified radii
approximately.135397 and.308880. First H8 witness index64 starts at(-1,-4,-2,-3),
target2: shifts approximately-4.15117637 and-1.88350960 predict0 and2, with radii
approximately.0191465 and.0438958. Rounded displays are explanatory only; every
coefficient, endpoint, shift and radius is exact in `trained_rows.jsonl.gz`.
No probability-mass or practical approximation-error lower bound was evaluated.

## Compute accounting
Four exact synthetic controls plus768 symbolic saved-witness verifications
and1782 affine probe classifications, with per-expression integer/rational
arithmetic counts and maximum integer bit sizes. Gcd, comparisons, hash/index,
control, allocation, bit complexity and full runtime are not priced by those
counts. Named source counts are not a complete estimator cost. The fixed saved
contraction used2.617548972CPU and33584KiB peak RSS in one review-lock process,
20CPU external/16CPU graceful cap,2GB and one thread. H4/H8 expression counts:
mul270887/611070; add366849/725310; sub35513/30834;
rational_div36122/37776; abs105708/98612. Maximum integer bits397/443.
No cap/failure or automatic continuation occurred. All row outcomes are saved.

Commands (one-shot, refuses overwrite):
```
ulimit -v 2500000
ulimit -t 20
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock timeout 25 /workspace/clean-neurons/.venv/bin/python m16/approaches/weight_circles/common_mode/controls.py
```
No automatic continuation or expanded sweep on failure.

## Claims
[inferred, independently source-reviewed] Positive cubes with opposite
correctness at common-shift partners reject this natural projection beyond
a single ambient-null line. [measured, controls independently reviewed443d4875; trained saved audit c84d99e8 passes]
Four controls pass and both trained models have exact robust common-shift
counterexamples. No probability, practical error floor or mass/cost saving
is inferred from this finite, deliberately nonrandom witness set.

## Alternatives not ruled out
Another projection; conditional decomposition; approximate invariance on most
probability mass; or a useful weight-derived orientation without exact reduction.
A finite absence of witnesses proves none of these. Nonzero margin slopes
alone may be irrelevant because correctness can stay fixed.

## Failure modes
Treating a gate endpoint as interior; missing rival-rival crossings; unstable
ties mistaken for permanent ties; output-label variation mistaken for correctness
variation; ignoring input ranks; finite probes mistaken for a probability law;
or extending a local failure to every possible circle representation.
