# T25: explicit structural description-length audit

## Question
How much additional representational information accompanies the fitted scalars in the original M16/M32 investigator formulas and the fixed scalar-budget families? This audits a declared code, not minimal Kolmogorov complexity or interpretability.

## Method and frozen protocol
TASKS-only claim94f2ce6 pushed before implementation. Read-only original `algzoo/labels.py`, `labels_32_10.py`, constant_budget/sweep.py and the T23 worker's current run.py/results; pin hashes before analysis. Exclude prose LABELS, comments, docstrings, model loading, fitting and scoring. Preserve all executable predictor helpers, dispatch, positions, subset literals and chosen operators. For universal dictionaries preserve the complete checked-in feature function, including its reporting-name construction, and a fixed inference wrapper ReLU(features(x)[:K] @ beta). This is a faithful executable upper bound and intentionally not an optimizer for shortest code.

Freeze one binary prefix grammar conditioned on Python's recorded AST schema and NumPy semantics: 3-bit value tags (None, bool, signed integer, UTF8 string, binary64 float, list, AST node, tuple); booleans add1bit; nonnegative lengths use Elias-gamma(length+1); signed integers use zigzag then gamma(value+1); strings add8bits per UTF8 byte; floats add64bits preserving the actual stored coefficient; each AST node adds an8-bit code from sorted builtin AST class names, then recursively encodes its fixed schema fields. Omit line/column metadata. Normalize unary-minus floating literals into a signed float literal. Preserve identifiers (no undocumented alpha-renaming). Decode and compare the normalized AST exactly. The decoder/codebook/environment are shared conditioning, not claimed free universal complexity.

Split total into fitted-float payload (64 bits each) and other code/structure. Original M16 fitted constants are floating literals in _n0.._n15 bodies; M32 are P's coefficient rows. Structural fixed float literals remain charged in other code. For dictionary models charge the feature/wrapper program once, plus exact beta binary64 payload, dimensions H,K and global family selection using gamma integers. A global family selector is not a per-neuron selected feature; if one were selected per neuron, it would incur separate codes. Repeated family declarations may be amortized across a shared collection only when that collection/library is explicitly specified. Report both standalone cost and conditional payload/header cost; never compare one side standalone against another conditional.

Separately inventory selected position-subset literals and explicit position selectors; a nine-position subset has a9-bit mask under a different fixed primitive, but this diagnostic is not added again to the AST total. Operator/functional choices are already charged by the syntax code. Report possible sharing and code dependence rather than treating either encoding as canonical truth.

Controls: binary codec roundtrip on signed integers, floats including signed zero, strings, lists/tuples and representative ASTs; mutation controls changing a subset or operator must change the encoded program. Decode source-derived programs and compare formula predictions on fixed deterministic arrays (no source-network calls, no fitting); all coefficient dimensions/counts must match saved artifacts. T23 training-only QR is excluded from prediction complexity because coefficients are returned to the original basis; record its computation separately as unknown here.

Two alternative explanations: scalar-count comparisons can hide searched structures (explicit AST and subset audit distinguishes); code lengths can mainly reflect source style/library decisions (report conditional costs and name/reporting overhead, no minimality ranking). Small raw code length is not evidence of simple human meaning. No new held-out scoring, quantization experiment or post-hoc best-family selection.

## Status
Protocol frozen; audit pending.

## Results
Pending.

## Compute accounting
Read-only parsing/encoding and bounded synthetic predictor controls only. Charge all work; no original model evaluations or training. One thread/2GB, at most20CPU under review lock.

## Claims
[inferred] Fitted scalar counts omit model-dependent discrete choices when those choices are selected. [inferred] A declared decodable program provides an upper bound relative to its decoder, not a lower bound on inherent complexity.

## Alternatives not ruled out
Shorter symbolic grammars, useful semantic macros, alternative bases, reusable subsets, coefficient quantization with validated error, fresh blinded investigator replication, other seeds and training. This audit cannot settle H-A1 from code length alone.

## Failure modes
Source-style dependence, incorrectly identifying fitted versus fixed literals, missing runtime-dependent helper definitions, double charging shared dictionaries, conflating search compute with deployed description length, or comparing unequal precision/library conventions. Preserve all parser failures and qualification limits.

The initial completed audit is preserved as `results_unframed.json` (.230034CPU). A source self-review identified missing common standalone framing: individual values were prefix coded, but concatenating a library and argument payload does not itself give a uniform prefix-free format across original and dictionary predictors. Correct complete predictor format is the single tuple `(AST, argument_payload)`; original argument payload is None. This adds9bits to originals and6bits to universal standalone totals. Shared-library and conditional payload costs remain separately reported, with framing explicit. No formula, fitted scalar count, precision or subset changes. The first launch also had a pre-execution syntax typo, preserved in failures.json.

## Completed bounded audit
The final framed audit passes15 basic codec roundtrips, three mutation/length controls, exact normalized-AST roundtrips, all40 universal predictor configurations and35,280 synthetic scalar output comparisons. Original formulas are compared against their original unmodified executable source, including normalization and decoder reconstruction. No source RNN, fitting, accuracy scoring or held-out data were used.

| Original formulas | Fitted scalars | Binary64 fitted payload | Other syntax/framing bits | Complete code bits |
|---|---:|---:|---:|---:|
| M16 |78|4,992|31,190|36,182|
| M32 |142|9,088|55,854|64,942|

The source contains6 position-subset literal occurrences (4 distinct subsets) in M16 and41 (37 distinct) in M32. Nine-bit-mask descriptions would require54/369 bits by occurrence, or36/333 bits for distinct subset definitions **plus references**. These are diagnostic alternative encodings, not extra costs added to the AST total and not total discrete search complexity. Explicit position selectors and structural operators are also charged in the complete AST; the supplemental syntax inventory is not an independent count of fitted parameters.

| Dictionary / neurons | K | Fitted payload bits | Conditional header/list overhead | Complete standalone bits |
|---|---:|---:|---:|---:|
| T7 /16|5|5,120|350|13,929|
| T7 /16|20|20,480|1,258|30,197|
| T7 /32|5|10,240|602|19,301|
| T7 /32|20|40,960|2,260|51,679|
| T23 /16|5|5,120|350|20,722|
| T23 /16|20|20,480|1,258|36,990|
| T23 /32|5|10,240|602|26,094|
| T23 /32|20|40,960|2,260|58,472|

T7's complete fixed feature/wrapper library costs8,453 bits in this code; T23's three-family library costs15,246 bits. Each standalone total adds that library once and6 framing bits. The table uses T23 Fourier's global selector; the other families' integer selector codes differ by the declared gamma encoding, not by a per-neuron family choice. All40 rows are saved, not just these examples. Across an explicitly shared predictor collection, charge one library and each separate coefficient/header payload, rather than repeatedly charging the library or silently giving only one side free structure.

Under this particular source-preserving code, T7's20-coefficient descriptions can be shorter than the original <=5-coefficient formulas. This is a counterexample to identifying coefficient count with total description length, **not** a robust shortest-description or interpretability ranking: source names, reporting code, reusable helper choice, binary64 precision and library conditioning all affect these totals. A shorter grammar can change their order. Original scalar-budget results remain exactly as reported; no accuracy/complexity frontier was refitted.

The frozen Fourier and centered polynomial families at full20 have the same function span, but their basis-evaluation algorithms differ. T23's reversible training-only QR rescue introduces no new prediction coefficients after conversion back to the original basis. Training search cost, failed optimizers, human formula search and choosing a best family after seeing results are different quantities and are not measured here. This audit charges the specified deployed representations only.

The recorded runs cost .230033854CPU(initial unframed), .303249362CPU(framing repair), and .288631386CPU(final explicit-position/original-source controls), total .821914602CPU, plus an early syntax-error launch whose shell duration is preserved. No original network calls or fits. Input and code hashes are saved in results.json/manifest.json. Reproduce with:

```sh
ulimit -v 2500000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python algzoo/description_length_audit/audit.py
```

Independent T55 source/theory review supports the grammar, framing,78/142 fitted counts and interpretation limits; separate length recomputation pending. Status: bounded audit complete, DRI confirmation pending.
