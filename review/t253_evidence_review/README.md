# T253 independent retained-evidence review

## Question
Does the new inventory distinguish directly retained T241 evidence, possible
recovery from aggregates, and evidence that remains unavailable in its scope?

## Method
Delegation `f751a1da`, frozen producer `dd4de999`, owner revision `a24a544c`.
Read the ten retained text/JSON payloads, inventory implementation and findings;
independently verify Git identities, all JSON key sets and structural lengths.
The two NPZ files are checked only as Git tree entries. No binary payload is
read, no producer script is invoked, and no score or model is recomputed.
T251 source/algebra review `d15ef043` is an inherited prerequisite, not a new
claim of model correctness. No external source is introduced.

## Status
Independent written source/schema/provenance review passes in this bounded
scope. `source_review.json` records exact pins and execution=false. This closes
the evidence-inventory component, not owner corrections or missing-data recovery.

## Results
The frozen subtree has 12 entries: 10 copied text/JSON files and two unopened
NPZ identities. The five source/README texts remain identical to T251, so its
energy-label correction has not appeared at this reached revision. That does
not assert absence of unpublished work.

The retained single-site schema has three arms, six metrics and 16-by-10
matrices; groups have 22 names and six arms. Modal JSON has five pattern
records with scalar summaries, active labels and mode descriptors. There are
no explicit case/pattern integer denominators, paired transitions, row labels,
projector matrices, cross terms or conditioning diagnostics in these JSON
schemas. Logs/report texts contain summary presentation, not these missing
raw records or a complete historical environment manifest. The numerical
values, score-to-log parity and source-to-output execution are not validated.

The findings correctly leave two possibilities separate:

- Exact marginal counts might be recoverable from sufficiently precise saved
  means, shares and known denominators under a validated rounding convention.
  This review performs no such recovery; absence of an explicit integer field
  is not proof of nonrecoverability. Binary contents remain unknown here.
- Even exact marginal accuracies and agreement generally do not fix paired
  correctness transitions. For a written 100-row example, list counts in the
  order (both correct, gain, loss, both wrong agreeing, both wrong differing).
  Both (50,0,10,20,20) and (40,10,20,30,0) give original accuracy 60/100,
  modified accuracy 50/100 and agreement 70/100. Yet correctness-change
  squared sums are 10 and 30. Thus marginal equality does not identify the
  paired change variance. This is algebra, not a test on the study's scores.

The conditional/donor design qualifications from T251 survive additional
decimal precision. A Git weight-blob identity is not original-checkpoint
equivalence; reading the source writer is not an archive schema check.
The named energy diagnostic and absence of projector/cross-term records do
not support removed-SSE reconstruction from these fields alone. They also
do not establish that any saved ablation score is false or every mode is
numerically unstable. Phase B/C/D results remain unvalidated in this tree.

Metadata/schema reproduction:
`python review/t253_evidence_review/check_metadata.py`.
The helper only hashes text, parses JSON structure and checks Git tree metadata;
it performs no score arithmetic or model/author calls.

## Compute accounting
Text/Git/hash/schema work only; no binary reads/decodes, NumPy imports,
eigenanalysis, fitting, RNG, score contraction or model calls. The producer's
reported metadata CPU excludes child Git time; no complete-cost claim follows.

## Claims
[measured source fact] The inspected schemas and reached bytes match this
bounded inventory. [inferred] Marginal recoverability and paired uncertainty
are distinct. Neither is a historical execution or empirical validity proof.

## Alternatives not ruled out
External or unopened archives may retain useful records. A separately scoped
integer compatibility check, provenance audit or owner correction may close
some gaps without replay. No automatic numerical extension is authorized.

## Failure modes
Changed source/output revisions, unsupported rounding assumptions, silent
binary-content inference or treating aggregate compatibility as raw paired
evidence require renewed review. Owner corrections remain owner-controlled.
