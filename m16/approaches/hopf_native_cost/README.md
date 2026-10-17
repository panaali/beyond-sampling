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
Initial source gate passed; all15 frozen analytic controls preserve outputs and
legacy counters (5f24544a),0.248CPU. Sorting/search source controls are prepared
and await independent gate; no trained model run yet.
T180 bounded handoff is complete and independently reviewed ba401fec.

## Results
No numerical estimator experiment yet. T178 proves the coefficient-zero mask
bug on an analytic circle, not on every saved trained row. Its source inventory
identifies root, dense recurrence, gate and sorting categories missing from the
old formulas; no total multiplier is assumed.

## Compute accounting
One thread,2GB; review processes <=20CPU, primary <=110CPU. Elementary function
units do not recursively count libm instructions. Dense einsum dots count k
multiplications and k-1 additions even with zeros. NumPy sorting/search remain
opaque pending source-supported closure, with dimensions/call census retained.
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
required before execution. No h64 or noisy-panel claim is licensed.

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

No new model/noise/input stream was generated. No h64 cost or whole-panel
ratio follows. Final saved model ledger audit is pending; next useful phase is
one direct existing h64 seed0 trace, after a new board freeze and review gate.
The largest conservative allowance is the SIMD label sorter; any replacement
or model-free setup exemption is a separate candidate/convention, never silently
substituted into this original-source result.
