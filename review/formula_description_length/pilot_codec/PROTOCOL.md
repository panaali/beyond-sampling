# Frozen fixed-pilot code experiment

Claim af2abda5 renews T260 on the completed T258 pilot, independently audited by
03ac9a3a. This phase has one numerical run, eight fixed artifacts in order
(all12, no_hinge8, network, HA1) × (hidden, labels). The latter two are source-style
reference upper codes, without transferred pilot performance scores. No choice
is made from the resulting lengths. Existing validation selections are immutable.

The unchanged T25 codec uses the recorded Python AST class schema and binary64
float tags (67 bits each including tag). AST module and payload form one tuple,
including its six-bit framing. Program lengths, payload lengths and framed totals
are independently counted by a recursive grammar census and checked against
actual bitstrings. Packed files have leading zero padding with the original bit
length saved. JSON payloads preserve binary64 values including signed zero; they
are auxiliary inspectable files, not the claimed code format. Decode must consume
all bits and match every AST field/container type/float bit exactly. No precision
search, code dictionary search, or model-dependent compression is attempted.

All23 retained input archives are loaded once: weights, fixed selection and21 fit
models. Loading only inference arrays leaves training counts, Gram/cross and fit
history out of the inference payload, while their archived provenance remains
bound. All source archive hashes and selected array hashes must match the reviewed
pilot. Final table rows retain all16 columns for each selected family, preserving
the original BLAS output shape; unused-column coefficient overhead is separately
reported. Fallback, ordered cell IDs, output routing, selected families and union
of comparator-bank dependencies are charged. No evaluation-dependent simplification.

Full feature function AST, including unused family branches, is charged once per
independently deployed table code. Family selection/fitting functions are omitted.
Banks use family5 with no own bank dependence; final8 needs step9, final9 steps7–9,
final10/11 steps1–9. The wrapper preserves retained-cell and pooled routing, ReLU
and float32 storage before original binary64 readout and first-index argmax.
Conditional hidden code omits readout; standalone label code includes it. This is
valid-finite-input inference equivalence, not equivalence of failure side effects
from evaluating all unselected families. There is no general floating parity test
in this phase. Actual predictors are never called or rescored.

Before artifacts: 15 codec roundtrips,3 mutation controls,8 dependency unions,
4 invalid-family guards,1 exact synthetic two-column table-routing control,1
trailing-bit guard:32 saved controls. Codec calls are22 encode attempts/returns,
16 decode attempts/15 returns (one expected trailing-bit rejection). The single
synthetic table call contains no fitted model. Eight target encode/decode calls
follow; limits are80 million bits per artifact and240 million total,128MiB per
auxiliary JSON. No input-specific raising of these caps or retries.

One process, primary lock,95 internal CPU seconds,105 soft/110 hard CPU seconds,
2GiB address space,120 seconds wall after lock, single BLAS thread. Internal timer
excludes top-level standard-library imports and output-directory creation; Bash
whole-process timing and resource limit cover those. Every target started row,
attempt count, returned length/binary and roundtrip result is saved before the
corresponding check. All failures/partial files persist. No overwrite; a failed
run needs separate scope/review. Twenty progress events on success. Frozen sources,
prerequisites and completed source review must be hashed into the immutable gate
and published successfully before launch (a separate checked publication call).

Independent56 saved review follows source/resource review by55. Desired evidence:
actual declared upper code lengths and exact roundtrips for fixed retained fits;
shared-dependency/full-column overhead and existing agreement shown separately.
Excluded: minimal complexity, interpretability, runtime speed, unseen agreement,
full historical C2 reproduction, population rate-distortion, setup/training costs
as inference bits, and any new model/fit/RNG/ARC evaluation. Prior regression
publication-order failure remains recorded; this launch cannot repair it.

Alternative explanation1: large tables dominate despite simple feature names;
framed dependency-closed lengths test this. Alternative2: apparent advantage is
conditioning on readout or omitting fallbacks/banks; paired hidden/label totals and
explicit table inventory test this. They do not establish an optimal code.
