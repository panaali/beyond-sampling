# Fixed missing-k30 continuation

Board extension b71c9744 freezes the original4,096 noisy copies only:
training seeds1..4, noise seeds20..23,256 copies each, unchanged RECIPE85B.
All source/checkpoint/noise/model/truth pins are inherited from provenance.json.
No new clean/seed0 model, recipe, training level or reference computation.

The first64 seed1/noise20 pilot is reused once; its copy0 is historical
T176 control1. Historical T176 control4 enters once at seed4/noise23/copy255.
The remaining63 fixed64-copy shards use their own unchanged baseline and
native-source trace, exact original values/legacy/cells and grouped-error/STATS
parity. No cost multiplier or favorable-row omission. Preserve incomplete
results and stop the relevant shard if a parity/resource condition fails.

Each process one thread,2GB,90CPU graceful stop and105/110CPU OS limits,
primary lock released between shards. Pilot10.57CPU gives rough676CPU total;
this is research tracing effort, not production estimator cost. Pilot review
and continuation source gate are mandatory before execution. The gate artifact
pins all pilot/source/provenance records and the independent reviewer result.

Saved-only comparison reports four1024-copy training-seed groups and16 noise
blocks, not just a favorable pooled average. Existing exact stored-float
Fraction contractions, integer own-cost sampling budgets, continuous/mean
budget conventions and1000 bootstrap RNG0 are retained. Exact individual
ratios/losses remain visible. Numeric111 and historical72 conventions remain
separate. Bootstrap is conditional on saved numerical truths and not a
simultaneous population or geometry/reference certificate.

Numerical source operations exclude integer/control/memory/library internals
and reusable model-free setup; separate observed categories remain. No complete
native/hardware cost, rung or M16T10 claim. Independent full saved-ledger review
must precede final interpretation. No estimator work in the reviewer.
