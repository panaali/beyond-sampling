# T260 independent saved audit — complete

The fixed eight-frame saved audit passes in `output_v2/results.json` after
source-gated reviewer recovery. All32 controls,23 inference archives/68 decoded
arrays, packed grammar/payload/AST bindings, dependency closures, full-column
counts,20 producer events and20 reviewer events reconcile. No author codec,
representation, predictor, model, fit, scoring or RNG calls were made.

| Artifact | Hidden conditional bits | Standalone-label bits |
|---|---:|---:|
| all12 | 28,408,630 | 28,419,477 |
| no_hinge8 | 433,558 | 444,405 |
| original network | 24,520 | 35,367 |
| H-A1 | 39,480 | 50,327 |

Eight frames total57,855,764 bits and7,231,972 packed bytes including leading
padding. The all12 closure includes bank steps1–9 and retains298,208 unused final
coefficient columns' scalar entries under the full-output-shape convention.
The selected no_hinge8 representation needs no banks and has zero unused-column
float count. This is the fixed representation that was actually encoded; it is
not a claim that full-column storage is minimal or semantically indispensable.

The independent byte reader checked every tag, gamma/zigzag field, AST field and
binary64 scalar, including signed zero, against separately constructed payloads
and stdlib NPY records. It verified full frame consumption and subtree category
lengths. Source-only AST normalization was reconstructed without executing the
wrapper. Float values remain exactly binary64; there was no quantization,
coefficient pruning, predictor replay or new score selection.

## Provenance and failure conservation

- Producer source44c7450e, rawd30fe7ab/sharedcb169ca1; T55source2b314033 and immutable
  gatebc6e9aa1/sharedf01a4b30 preceded the producer's sole run.
- Independent checker a72b80ee/shared5f103dc9; source gate5020286e/shared51895f92;
  immutable211132d1/shared6b2f6ee4 preceded attempt1.
- Attempt1 reviewer-only progress-parameter collision retained5f596edb, including
 32 passed controls/68 array bindings,3 events,zero decoded artifact frames and
 1.0331012650000002CPU. Producer evidence was unaffected.
- Recovery scopeac325e25, source45b5079e/shared339cddc1, source review2a81c8b8/shared
 6b618e68, immutable01795a5a/sharedfaa10bc6 all preceded the sole recovery launch.
 Reader/formulas/fixtures were unchanged; only event-name/fresh-path/CPU-accounting
 repairs were made. The repeated reviewer work is charged, not discarded.

Recovery CPU3.088197805s, peak200,816KiB; total recorded reviewer CPU4.121299070s.
Both launches used105soft/110hardCPU and2GiB,95internal/120wall-after-primary-lock.
Producer CPU6.191943126s and277,172KiB remain separate. External run1/run2 logs and
all original sources/outputs/gates remain immutable. These timings describe
encoding/audit overhead, not predictor evaluation cost. The successful recovery
made no further changes or attempts.

## Scope and remaining questions

This is a declared-code retained-artifact upper length and exact roundtrip/
dependency consistency result. It does not establish minimal description length,
interpretability, predictor numerical equivalence, historical C2 artifact recovery,
fresh agreement, a population rate curve or a full inference-cost comparison.
Pilot metrics are inherited descriptive evidence; no new accuracy computation was
performed. The cost of learning/constructing the retained pilot remains inherited.
Alternative precision, dead-column removal with independently checked execution
semantics, different feature families and fresh evaluation remain untested here.

Original README/PROTOCOL files remain frozen as prerequisite evidence. This new
handoff supersedes their preparation status without mutating their pinned bytes.
Owner h4 may attach this independent result and close the bounded T260 phase;
DRI scientific registration/confirmation remains separate.
