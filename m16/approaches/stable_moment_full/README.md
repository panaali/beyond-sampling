# Full frozen stable-moment mass phase (T109)

## Question
Does the independently reviewed stable-moment pilot retain accuracy and save named arithmetic over the existing35,098-cone H8T4seed0 corpus?

## Method
Claim96fc5f40 pushed before implementation. Freeze stable_moment_h8 ef2309cf/29654d70 selector, existing target1e-7/35098, orders2..32, analytic-inner kernel and exact acute/KKT fallback. Reuse64 pilot rows once; evaluate remaining35,034. Existing reference and baseline numerical values are validation only. Every non-moment output must match saved baseline bitwise. Shards256; retain every exception, failed guard, unresolved result and stop-gate checkpoint. Baseline/reference source hashes are pinned.

## Status
Full35,098-cone phase complete after independent source and corrected resume gates; saved-data audit requested.

## Results
All35,098 return:14,768 center intervals,12,596 moment intervals,7,602 center quadratures,132 exact rescues.22,502 non-moment values/bounds match prior baseline bitwise;64 pilot rows reused once.20,329 moment attempts, no invalid guards or unresolved cones. Named121,379,672 versus142,873,286 (15.0438% reduction21,493,614);3,458.31/cone still fails300 goal. Accuracy.9761608085722152 versus reference.9761608075466027, delta+1.02561248e-9; conditional bound1.20221573e-8, L1 error3.82018923e-9, max2.74433506e-12. No outward floating certificate.

## Compute accounting
Same per-cone scalar ledgers, all failed selectors, canonical cold rules and staged repeated cold costs. Exact Fraction/MP operation inventories/bit widths and CPU separate. Pilot exact35 validation remains separate from estimator cost but included in staged observedCPU. Actual full phase validation/read/serialization CPU retained; not a runtime benchmark. Numerical library internals, Legendre, integer/control/copy/memory/runtime and floating certification remain open. One thread2GB, review lock<=20CPU with18CPU soft cap. Unique session files preserve retry overhead, no discarded completed shards.

## Claims
[inferred, independently reviewed] Real-arithmetic moment interval and equivalent positive-square variance identity. [measured] Pilot behavior only; full output pending. No FP or fullcost/noisy-rung claim.

## Alternatives not ruled out
Pilot shortcuts may not cover costly cones, so full savings may be smaller. Cheapest discriminator is fixed all-cone cost accounting. Positive variance can still underestimate true variance through Gram/mean rounding; preserved exact pilot checks are finite evidence, not outward proof. An outward arithmetic variant remains unimplemented; no broad mass-method negative.

## Failure modes
Nonpositive/reversed floating intervals retain the original fallback. Reference/bound or non-moment parity failures stop. Finite cap unresolved is explicit; do not raise cap or relax target. No source-based mass fallback.

Reproduction if needed: one-thread2GB review-lock Python run.py FIRST_SHARD STOP_SHARD, then summarize.py under the same limit. Existing completed shards are reused; do not rerun completed experiments.

Pre-execution independent review found an inherited resume-harness defect: prefix validation alone could skip a persisted explicit failing row and later mark its shard complete. Before any trained execution, the runner now stops on preloaded failure/comparison/bound/parity failures, including already-complete shards; aggregation independently rejects them. Legitimate finite unresolved remains visible. Original pre-fix source is preserved in3dcc7b6c; no numerical kernel change.

Full cost distribution explains the smaller-than-pilot saving:7,602 remaining center quadratures contribute109,924,037 known row units (90.56% of the total), exact rescues4,720,455, moment shortcuts3,866,972 and center intervals2,850,224. Cold canonical17,978 plus6 universal units complete the total. No full-corpus CPU comparison is inferred from different session layouts: staged observed10.2053CPU includes imports/validation/serialization, plus aggregation0.6921CPU; actual repeated rule setup55,720 is retained. Exact certificate and MP selection0.44824/0.12120CPU; max rational bits804/788 remain separate from namedfloat ledger.

Independent pilot source/exact/raw gates precede full run; written T181 review hash forthcoming. Full source gate passes1c0aedd2. New summary by-method arithmetic is descriptive and requires saved-data review. Full baseline reference/geometry/target/kernel unchanged. Full phase avoids order2 because the moment shortcut accepts those cases, and charges the eight actually used cold orders. No kernel parameter tuning or reference-derived allocation.

Independent T181 review292b54cc verifies all35 exact variance references,64 stored pilot outputs/counts and the corrected full-runner resume gate. Full-corpus numerical review remains separate.
