# T172 native RECIPE112 arithmetic audit

## Question
Does the actual pinned NumPy RECIPE112 have a defensible arithmetic cost consistent with its comparison to sampling, after the repaired scalar wrapper and finite-panel parity audit?

## Method
Claim940aaac0 was pushed before this folder was created. T113 and T115 remain separate: no training-axis or5,120-case panel rerun. Static source is est75.py SHA426020c67a52678e5cccb20c552837fe2d233c3fe5e13557916034a3c46877d3, frozen K=(99,99), margin_all,eps.1,cliptol1e-12. `inventory.py/json` records all reachable-function operators/calls and prior T146/T153/T128 tracer hashes. Read-only prior tracer reuse must be checked against this source's actual operations rather than transferring cost multipliers.

Freeze a source-level explicit numeric convention: each scalar addition, subtraction, multiplication, division, abs, negation, comparison and elementary-function call costs1; dense dot/matrix multiply costs n*k multiplications plus n*(k-1) additions. NumPy norm costs n squares+(n-1) adds+sqrt. Cross product3D costs6 multiplications+3 subtractions. NumPy sum/mean/min/max charge reduction operations per actual output size, plus mean division. Boolean reductions are separately identified, and actual source comparison expressions are retained even when involving zeros. Model-free constant geometry is a declared reusable dependency, not free model-dependent advice. Source initialization and per-model operations are separated. Integer/index/control, Python dispatch, allocation/copy/memory and elementary-function internals are tracked or named separately; this is not hardware instruction accounting. Matrix numeric prices do not silently equate arithmetic to walltime.

Instrument an isolated AST copy, preserving evaluation order, short circuit, ndarray receivers, augmented assignment semantics, source values, legacy counters and STATS. Exclude only historical `ops.add(...)` arguments and STATS bookkeeping from estimator work. Record unsupported calls rather than silently ignoring them. No estimator optimization, pruning or tie correction. Comparing this native trace to scalar counts does not establish a universal relation.

Frozen analytic controls: exact3D basis dot/cross/norm; dense2x3 times3x2; scalar/array numeric comparisons including zero equality; computed ndarray max/sum/mean; generator min/max and short circuit; augmented assignment identity; then four H1T3 constant-first-label models from T113 (zero, zero-readout, identical-readout, dominant-first readout), plus H2T3 rational mixed-gate model. Require bitwise baseline/traced estimates, identical legacy counters and STATS; retain original tie failures. Independent T55 source/control gate before only the original H4seed1 noise89copy2 and H8seed2 noise89copy3 native traces. No trained corpus expansion unless a subsequent frozen task scope justifies it.

Two alternatives: (1) native dense/repeated arithmetic materially exceeds the scalar count despite output parity; direct source trace distinguishes. (2) accounting wrapper itself changes branches or costs; analytic operator controls and full output/counter/STATS parity distinguish. (3) a false cost increase comes from counting bookkeeping or nonexecuted branches; site ledger and independent source audit distinguish.

## Status
Static inventory,12 analytic operator checks and five synthetic model parity controls complete; independent source/control gate ea2609ef passed before both frozen trained traces. Both traces completed with bitwise source parity; independent two-case raw cost review a558c8a9 passes. Full1,280-case H8 panel now complete; final independent aggregate review a51e2287 passes.

## Results
The actual `_resolve_margin2.margin` evaluates `P @ A3f.T` twice and uses dense matrices for both; scalar rec112 instead computes each point separately and skips inactive terms. Source `_est75` also constructs full HxH products after masks. These source differences motivate direct accounting and prevent cost transfer. The repeated expression is preserved. Original known positive-volume tie defects are not repaired in this task.

Reproduce inventory: `/workspace/clean-neurons/.venv/bin/python review/recipe112_native_cost/inventory.py`.

## Compute accounting
Inventory parses source only, zero model evaluations. All future numerical checks one thread,2GB, review lock<=20CPU or primary<=55CPU; source/hash loading and setup tracked separately. Prior research dependencies remain explicit.

## Claims
[inferred] Direct unchanged-source accounting can close gaps invisible to the scalar Num counter. [measured] Static source contains dense/repeated operations not identically represented by the sparse scalar implementation. No cost upper bound, sampling win or ARC solution claimed.

## Alternatives not ruled out
A different efficient implementation, shared setup, compensated arithmetic, source simplification and a proof covering uninstrumented operations. Native performance may differ from arithmetic units. The recipe family remains open irrespective of this finite audit.

## Failure modes
AST evaluation-order changes; omitted NumPy methods or reductions; integer-versus-numeric category confusion; double-counting returned arrays; generator exhaustion; scalar/array coercion; source setup conflated with per-model work; floating branch changes; treating known degenerate tie defects as new trained failures.

## Control results and preserved harness failures

`controls.py` checks twelve primitive cases and five synthetic estimators. Every model preserves bitwise estimate, historical counter, STATS and all12 grouped family error values. Known zero/identical-readout accuracy defects remain2/3 against analytic1/3. Named native numeric totals for zero/zero-readout/identical/dominant-first/mixed are4526/8100/8100/11573/22944; legacy totals2736/4536/4536/6416/10771. These finite controls validate tracing, not trained cost or generic source completeness. Updated successful controls record.287578893CPU excluding imports; original successful control.332445782CPU is preserved in commitcbf6a73f. A proactive source audit additionally identified implicit nonzero comparisons in `np.any` of floating arrays: all n truth conversions are now conservatively charged, even if a library could short-circuit; a new exact control covers this. Prior harness failures are saved separately: an augmented-assignment fixture omitted its local initialization; the evaluator passed `info` positionally into K; JSON attempted tuple dictionary keys after all parity assertions passed. Repairs affect fixture plumbing/serialization only, and all controls were rerun.

## Frozen trained trace results

After independent source/control approval ea2609ef, `trained.py 4` and `trained.py 8` reconstruct only the two original noise89 controls. Native source value, historical counter, STATS and all12 grouped family contributions match the uninstrumented source exactly. H4seed1/copy2 named native arithmetic165415, repaired scalar137268, legacy70471; H8seed2/copy3 native458917, scalar326667, legacy160564. Native/scalar ratios1.20505 and1.40485 establish a finite implementation difference; they are not transferred to any other model. The corresponding separately recorded integer/control expression counts are5756 and11593, with1456 and3150 memory/copy call sites executed. Omitted loop/index/dispatch/memory work remains explicit; these integer totals are not complete control costs. No LAPACK kernel is used in the per-model recipe, but source sum/dense matrix/reduction/elementary operations are named conventions rather than compiled instruction ceilings. Model-free HALVES construction is inherited and unpriced here, separate from per-model work. Source module startup creates one identity array; no priced numeric startup expression.

Successful trained processes record1.185948163 and1.235728071CPU including imports. `summarize.py` independently groups the saved site ledger by function and preserves all categories; no model reevaluation. Higher H costs shift toward dense map/margin construction, while clipping and spherical mass remain explicit. This closes the two-case source-accounting diagnostic only. A model-dependent panel still requires its own native trace or a proved envelope; full scalar-panel results are not silently upgraded.

## Frozen H8 direct-cost continuation

TASKS7db9ae6f was pushed after the two-case independent raw audit passed. This phase prices exactly the existing1,280 H8 core cases (five zoo seeds, originalnoise80..83,64 absolute-sigma.05 copies), without transferring two-case multipliers. First `panel.py 0 80` is a cost/parity gate; independent raw review9b4c6a94 passed before the other19 shards. Maximum55CPU per64-case primary-lock shard, stop/save on any error or50CPU soft cap. Each model verifies its T113 byte hash and preserves baseline/traced hex, grouped errors, legacy counter/STATS, saved estimate and cell count; full site ledgers are stored in compressed JSON-lines beside compact totals. No new noisy stream, truth integration, recipe settings or model-input sampling. H4/5/6 native panels are outside this scope.

The explicit numeric-only sampling comparator is4H²+12H−1=351 atH8: first/recurrent arithmetic4H²+3H, ReLU3H, readout6H−3 and output argmax2. Model-free input ranking and final integer correctness are excluded in both estimates; input generation, loop/control/memory and elementary-function internals remain named exclusions. This is a source-expression comparison under a declared convention, not a compiled/hardware ceiling or full ARC rung pass.

First64 native pilot (H8seed0/noise80) completed23.965232631CPU,64/64 model hashes, bitwise traced/baseline outputs, historical counters and saved cell counts pass. Total named native numeric30,628,550, with each full source ledger in `panel/H8_z0_n80.jsonl.gz`. T55 independently checked every compressed site ledger, hash, baseline/traced field and prior cell/counter identity; numeric total30,628,550 and integer777,410 reproduce. Review took.024CPU without traces. The gate passed before the remaining19 shards launched; written independent review9b4c6a94. The bounded cost gate is feasible under55CPU; no accuracy/cost conclusion is drawn from the single pilot group.

Independent two-case review a558c8a9 verifies model/source hashes, every saved ledger total, and baseline/traced value, legacy counter, STATS and all12 grouped family keys. The reviewer initially expected24 keys (the number of initial halves), preserved that assertion failure, then verified the actual12 grouped keys. The first64 pilot review subsequently passed as recorded below; remaining shards require a final independent aggregate audit.


## Complete H8 panel

All 20 frozen shards completed, covering exactly 1,280 original H8 models. Every native estimate equals its saved original estimate bit for bit; source tracing also preserves every legacy counter, cell count and grouped family result. No cap, unsupported call or numerical failure occurred. Each model's own integer budget is used, with sampler351 under the explicit numeric-only convention.

| Zoo seed | Models | MSE / sampling variance | Conditional bootstrap95 interval | Mean named units |
|---|---:|---:|---:|---:|
| 0 | 256 | 0.000317303 | [0.000217971, 0.000431947] | 476852 |
| 1 | 256 | 0.000584660 | [0.000327240, 0.000882001] | 448043 |
| 2 | 256 | 0.020706812 | [0.006975190, 0.043529044] | 436508 |
| 3 | 256 | 0.024834689 | [0.004719236, 0.055135376] | 440013 |
| 4 | 256 | 0.191699510 | [0.005287945, 0.534805244] | 461549 |

These five intervals are conditional on saved truths and source-count convention, not simultaneous guarantees or population-wide proof. The finite H8 core advantage survives direct named-source pricing; the first two controls' cost ratios were never transferred. Full H4/5/6 native core pricing, the separate borderline relative-noise group, library/hardware costs and setup qualifications are unchanged. In particular, the T113 borderline failure is not silently reclassified by this core-only result. No full ARC solution, full ladder acceptance or generic simultaneous-freezing theorem follows.

Recorded replay CPU465.857324962 seconds across bounded shards; final aggregation.072352408CPU. Research timing includes imports, tracing and archival overhead and is not the estimator's production runtime. Reproduce all summaries with `panel_summary.py`; archived full site ledgers permit review without repeating any model calls. Final independent review a51e2287 verifies all1,280 full site ledgers and20 source/prior/archive manifests, exact native output parity, numeric579,318,688 and integer14,682,950 totals,65-digit budget ratios and conditional bootstrap endpoints. No model traces were rerun. DRI registration is requested.
