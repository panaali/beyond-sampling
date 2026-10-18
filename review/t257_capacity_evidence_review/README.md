# T257 independent capacity-evidence review

## Question
Do the retained records support the source/schema and provenance qualifications
in the new T234 evidence inventory?

## Method
Delegation `8e739420`; producer `d9208ca4`; owner
`2ea6766d8fa0d30cc98348de247b9bec331545d1`. Read the new SOURCE_REVIEW, source
writers, README and structural JSON description. Independently bind all 15
text/JSON payloads, the complete 21-file Git tree and six binary metadata
entries. Check JSON keys/types/list lengths without score arithmetic. No
producer script, author module, training, model or binary payload is invoked.

## Status
Independent written/source/schema review passes in this scope. The durable
`source_review.json` has execution=false. No score, model identity, runtime,
training trajectory or architecture-capacity conclusion is validated.

## Results
The scoped Git difference from repaired source `d2fdf6bd` changes README and
adds result/weight records, not executable source. This supports reuse of
the prior implementation review's source boundary; it does not transfer a
runtime or CUDA resume-parity conclusion.

The retained a30 record has configuration, sample/step counts, a training-time
field, 64 five-model curve entries and five evaluations. Its keys omit
src_hash; each evaluation omits correct and discordant_correct_vs. The single
zoo-code record has seed, training wall time and the same older evaluation
schema. Current run.py's output constructor includes src_hash and current
evaluate.py's output constructor includes both integer fields. Thus the
retained files are not the complete schema written by those current success
paths. A claimed later evaluation may exist elsewhere; the missing fields
do not prove it never happened. No field value or displayed accuracy is
tested here beyond structural declared types/list lengths.

Six NPZ names/blob IDs/sizes are retained, but there is no evaluated-weight
payload-hash binding in the inspected JSON. Their contents, shapes, numerical
types and correspondence to the evaluated model remain unopened and unknown
in this review. A repository blob and matching seed filename do not establish
checkpoint-to-evaluation identity.

The source reproducibility wrapper delegates optimizer/schedule defaults to
the installed alg_zoo package; this snapshot does not identify that historical
package implementation. Its timer brackets training only. The batched driver's
timer begins after trainer construction and ends before evaluation; it also
includes checkpoint/report work in that interval. Neither is a full end-to-end
cost or Colab-unit ledger. The producer correctly avoids treating them as such.

Falling inside a coarse historical seed spread is an outcome sanity check,
not equal data streams, optimizer trajectories, a training-outcome distribution
test, or resume parity. Matching evaluation seeds alone cannot reconstruct
missing cross-model discordance records. Seed variability, conditional MC
uncertainty, selected-best outcomes and multiplicity remain separate. Rounded
range endpoints do not define an exact numerical acceptance test without an
explicit rounding convention; no such test is supplied by this review.

The anatomy writer separates a global-second-input sign stratum, then
partitions remaining errors by predicted input rank. This describes a counting
rule; it does not establish that the network is blind to rank in that stratum.
The records retain proportions rather than the integer joint anatomy table.
Possible recovery from precise marginal fractions is not ruled out, but no
recovery or anatomy-conservation calculation is done here.

If later independently validated, an achieved finite model supplies evidence
for a feasible performance level, not an upper bound on the architecture's
best attainable accuracy. The inventory correctly preserves old-run and
external-artifact possibilities, source/runtime gaps and separate future
arithmetic/CUDA controls. No reported score is invalidated.

Metadata/schema reproduction:
`python review/t257_capacity_evidence_review/check_metadata.py`.

## Compute accounting
Text/hash/Git/schema reads only. Zero author, model, training, RNG, binary
payload or score-contraction calls. Child Git time is not a full scientific
cost measure and no such cost is claimed.

## Claims
[measured source fact] Text/tree and schema correspondence at the frozen
revision passes. [inferred] Retained output fields do not establish the
stronger runtime, paired uncertainty or architecture-ceiling claims.

## Alternatives not ruled out
Missing metadata may survive in other Colab or checkpoint artifacts; old
execution may be correct. Separately scoped provenance and integer-compatibility
checks may recover evidence without training replay.

## Failure modes
Inferring binary contents from source, binding old outputs to current writers,
assuming seed equality establishes paired outcomes, or promoting finite best
scores to a capacity ceiling would exceed this review. No new execution is
authorized and the owner's numerical budget remains unchanged.
