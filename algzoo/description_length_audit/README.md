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
