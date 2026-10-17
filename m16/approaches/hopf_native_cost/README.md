# T183: native Hopf numerical accounting

## Question
What numerical work does the frozen Hopf estimator actually execute under the
same source-expression units as its sampling comparator?

## Method
Claim bf7491f5 precedes implementation. Pin T178's original Hopf37e5cb source,
analytic controls, sampler, and reviewed T146/T180 accounting helpers.
Inventory before execution. Count scalar multiplication, addition, comparison,
elementary function and modulo separately; Boolean array operators retain separate tags; scalar integer expressions
have separate categories (integer-array comparisons remain included). Numerical sorting and
search are opaque initially, not free. Memory/index/dispatch/control are
explicit exclusions. Preserve every original output and legacy counter.

Two source arms: unchanged original and exactly the independently reviewed
repeat-to-tile root-mask substitution, with no other numerical change.
No shared source adoption. Freeze root controls from T178, complete analytic
H2T4 circle witness, zero/constant/sign models at fixed unit circle geometry,
and nonzero deterministic coefficients. Independent adapter gate precedes any
saved-model trace. No new MC/noise/model/reference draw or panel tuning.

## Status
Bounded analytic, h8 and h64 phases complete. Independent saved audits:
1d008fde (sort/search), 6da114dc (network/closure), 155b3b18 (h8),
0cfdfa39 (h64). Same-checkpoint reference contraction a283a329 passed independent exact-rational
arithmetic review (3ed60f55). No full panel, full runtime or challenge claim.

## Results
Both original and tile-only source arms preserve the original saved clean
H4T4 seed0 values and counters. No source correction is adopted globally.

| Grid | Estimate | Legacy counter | Explicit expressions | Conditional source allowance total |
|---|---:|---:|---:|---:|
| nu8/nd16 | .4810975064800962 | 1,032,520 | 2,324,112 | 3,960,665 |
| nu64/nd128 | .47710001754048736 | 65,934,700 | 148,561,137 | 252,465,096 |

The existing same-checkpoint positive-cone reference is .4770131259941194
with an integration-only bound 3.157995918627031e-8 (6347 cones; independent
review 449a01e). Floating geometry/kernel error is excluded from that bound.
The h64 squared error divided by equal-source-budget sampling variance is
.0395893 for the unchanged dense comparator, or .0474580 for the separately
derived zero-state-optimized comparator. Conditional integration-only upper
ratios are .0396181/.0474925. These are single-model comparisons under the
qualified cost/reference assumptions, not a noisy-rung result.

The h8 allowance-budget ratios are 1.37224/1.64500; because the candidate
allowance is an upper envelope, these do not prove an actual-cost failure.
The old saved h128 reference differs from the positive reference by
+.00015865237928625 and is not treated as exact.

## Compute accounting
One thread,2GB; review processes <=20CPU, primary <=110CPU. Elementary function
units do not recursively count libm instructions. Dense einsum dots count k
multiplications and k-1 additions even with zeros. NumPy sorting/search keep their opaque dimensions/call census alongside
separately qualified source allowances; installed-binary equivalence remains open.
All actually executed circle/base setup is initially included once per call;
legacy n_ops bookkeeping is diagnostic and left uninstrumented.

Sampler192 (H4T4 dense) or160 (zero-state optimized) counts one prediction and
correctness indicator given free true label. Actual sample aggregation adds
m-1 reduction additions and one batch accumulation; the final division and
loop setup are separate. Neither192 nor160 alone is a complete mc_acc call
count. RNG/true-label computation and memory are separately excluded or traced,
never silently mixed into the estimator convention.

## Claims
[inferred] Dense/root site discrepancies may increase numerator substantially;
a complete per-site ledger is the cheapest discriminator.
[speculative] A larger sampler denominator or reusable circle geometry may
partly offset numerator corrections; test the same declared implementations,
not a transferred ratio. No full-cost or rung conclusion is asserted.

## Alternatives not ruled out
The legacy counter can overcharge some sites; preserve it alongside actual
ledger. Sorting may be inexpensive relative to recurrence, or dominate at many
pieces; retain per-call dimensions before selecting an envelope. The local
mask repair may change trained partitions; do not combine arms or transfer
old accuracy evidence. Common-unit cost alone cannot certify quadrature bias.

## Failure modes
Changed tie/duplicate ordering, omitted broadcast factors, treating numerical
sort as memory, counting n_ops as estimator work, double-counting nested native
kernels, confusing indicator and batch-average sampler cost, source changes or
unsupported calls, or extrapolating one circle/model to all saved cases.

### Initial numerical controls
Four original coefficient-root controls each count130 explicit units, matching
16m+2 at m=8 (including Boolean arrays). The original analytic circle remains
.125, the tile-only arm.051208191174783355, each unchanged by observation.
The fixed2x3 generic Hopf grid remains.27181996624992727 with legacy15396;
visible arithmetic34246 plus42 opaque sorting/search calls. No H4 ratio follows.
The unchanged5-input H4 sampler returns.2; raw983 =966 prediction, hit
aggregation and final-division units +17 explicit loop/index units. True-label
argsort remains separately opaque/free-label convention, and no random draws
occurred. Native counts do not certify quadrature bias.

### Sorting/search closure controls
After independent source gates,1850 fixed stable-sort cases and40 right-search
shapes pass. The source-shadow keeps the complete numeric timsort body; actual
NumPy permutations remain authoritative. Parent Python CPU.0365s excludes the
compiler child; the full capped command took.487wall seconds, so no exact
combined CPU claim is made. All108 finite four-label network controls pass
in.004CPU, and10 old circle/grid controls preserve every raw site and output
under the separate closure observer (.173CPU). `controls_summary.json` keeps
unpriced/raw and conditional source allowances distinct. None establishes
compiler/binary equivalence or a complete runtime budget.

The original and tile-only seed0 H4T4 h8 probe is frozen by admin fec66d8b.
Existing checkpoint83ddef... matches canonical public object MD5 metadata;
converted float64 model086adb... is saved, no new weight download. Exact
source/controls/checkpoint hashes and original ev_44 h8 value/counter are
required before execution. The h64 continuation was separately frozen by c57b8cb6; no noisy-panel claim is licensed.

### Commands and preservation
All numerical commands use the venv Python, OMP/OPENBLAS/MKL threads1,
`ulimit -v 2500000`, and a process CPU limit. Scripts refuse to overwrite
saved artifacts; reproduce in an isolated owned checkout with prior outputs
archived separately. Original commands relative to repo root:

```
.venv/bin/python m16/approaches/hopf_native_cost/inventory.py
.venv/bin/python m16/approaches/hopf_native_cost/controls.py
.venv/bin/python m16/approaches/hopf_native_cost/build_sort_shadow.py
.venv/bin/python m16/approaches/hopf_native_cost/sort_controls.py
.venv/bin/python m16/approaches/hopf_native_cost/argsort4_controls.py
.venv/bin/python m16/approaches/hopf_native_cost/closed_controls.py
```

The research worktree uses the shared interpreter at
`/workspace/clean-neurons/.venv/bin/python`. Source controls use the serialized
codex-review lock, <=20CPU. build_sort_shadow generates source only;
sort_controls invokes g++17/O2 to make the ignored source-shadow library.
No external library binary is committed. Network source metadata is pinned;
initial renamed-file404s and wrong-interpreter metadata remain preserved.

### Saved seed0 H4T4 h8 discriminator
Both fixed arms completed df62497a. Each exactly preserves original saved
value.4810975064800962 and legacy1,032,520. Explicit arithmetic including
one constant setup is2,324,112. Separate source allowances are136,633 stable
sort comparisons,42,960 right-search units and1,456,960 conservative four-key
sort units, giving3,960,665 conditional source units. Explicit/legacy2.2509
and conditional/legacy3.8359 apply only to this nu8 grid/model. Each numerical
trace took.228/.232CPU with198MB peakRSS; imports/identity loading precede the
trace timer. The tile-only arm changes no output/count here, which does not
undo the independently demonstrated coefficient-zero defect.

No new model/noise/input stream was generated. No whole-panel ratio follows. Saved model ledger audit passed155b3b18;
the separately frozen direct h64 trace passed0cfdfa39.
The largest conservative allowance is the SIMD label sorter; any replacement
or model-free setup exemption is a separate candidate/convention, never silently
substituted into this original-source result.

### Direct h64 and reference reproduction

```
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/model_probe.py --arm original
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/model_probe.py --arm tile
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/model_h64.py --arm original
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/model_h64.py --arm tile
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/archive_h64.py
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/reference_compare.py
```

Each primary process uses the shared compute lock and <=110CPU/2GB/one thread.
The h64 trace CPU totals are13.8247/14.1811, peakRSS215MB, excluding startup
and identity loading. Every one of57,344 opaque calls has a closure record.
Lossless model_h64_{original,tile}.json.gz archives contain the full ledgers;
model_h64_summary.json records raw/compressed hashes. Read with gzip without
rerunning. Original model_probe/model_h64 outputs refuse overwrite.

The comparison includes actual hit aggregation and explicit loop/index terms:
193N+8ceil(N/1,000,000)+2 for the dense sampler and the separately labeled
161N+7ceil(N/1,000,000)+2 zero-state comparator. At h64 these admit
1,308,109/1,568,106 draws. RNG and true labels are free under the stated
comparison; prediction and aggregation are charged. Claims remain conditional
on source allowances and reference geometry, with no panel multiplier transfer.

[measured] The original and repaired mask arms agree on this one checkpoint at
both grids; the exact analytic mask-defect witness remains valid.
[inferred] The h64 conditional comparison is promising for a direct broader
saved-panel cost audit; it does not establish that audit's result.

### Existing panel continuation
TASKS304a1ddb reserves direct original-source h64 coverage for the existing
25 H4T4 rows. Canonical checkpoint metadata and exact original noise code are
pinned in f8a58823; reconstructed identities e158adb1 preserve seed0 clean hash.
Pilot seed0 copy1 (8d6119ec) matches saved value .35535204853183044 and legacy
64,861,332 exactly. Explicit146,777,020 and conditional247,259,835 are measured
directly,13.385CPU, not transferred from seed0 clean. Saved audit pending.
The remaining23 entrypoint ecd62bc1 makes only index/pin substitutions in the
frozen pilot source. Its execution waits independent pilot and source gates.
See PANEL_SCOPE.md for exclusions and failure-preserving metadata correction.

```
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/panel_models.py
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/panel_probe.py --seed 0 --copy 1
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/panel_continue.py --seed S --copy C
```

### Complete direct H4T4 cost coverage
All25 original saved h64 rows complete throughf36cfc2d; summarydb770a8e checks
25 unique model identities, exact saved outputs/counters, all source pins,
every explicit site and all1,433,600 opaque/closure records. Explicit total
3,524,673,740; conditional source total5,975,034,725. Per-copy conditional
budgets range219,914,303–257,767,782. This directly measured variation replaces
any proposal to transfer a single seed's multiplier. Full native/binary/runtime
qualifications and the generic root-mask defect remain.

The25 numerical trace CPU sum is348.245485924, including prior clean/pilot
records once; newly executed remaining23 use321.035824367. Each trace includes
its uninstrumented baseline parity execution, not just an estimator call.
Imports/provenance loading and compression are outside those timers. Saved
contraction adds3.288778888CPU. Maximum peakRSS215,464KiB. No new MC truth or
input stream. Metadata-header setup failure remains documented; no failed
estimator run was replaced or dropped.

```
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/panel_summary.py
/workspace/clean-neurons/.venv/bin/python m16/approaches/hopf_native_cost/panel_compare.py
```

The comparison is frozen by TASKS3963b86c and source241d3fd4, independently
source-gated. It waits complete saved-cost audit, uses original75-family
simultaneous CP truth boxes, both dense/optimized costs and both uniform/matched
integer draw allocations. No noise subtraction or h128-as-truth is used.

### Reviewed panel comparison
Full cost audit e61dffe3 and all60 exact-rational budget/interval comparisons
792106be passed independently. All raw point ratios are below1, maximum
.5034968603. Optimized matched pooled point ratios by seed0..4 are
.2351224/.0453680/.1266595/.1335504/.0578261, but simultaneous truth-box
upper ratios are3.13647/2.19832/2.58346/2.82010/2.25329, with all lower0.
These intervals do not establish a panel win or failure. Reference precision,
incomplete native/runtime accounting and the generic mask defect remain
separate qualifications. No h128 truth or noise-subtracted-only success claim.

The next useful scientific discriminator is fixed higher-precision truth for
this already frozen panel. Owner34 confirms no matching reviewed positive
references beyond clean seed0; new reference work requires a separate claim,
fixed sample size/seeds and cost-first pilot, never accuracy-adaptive stopping.
