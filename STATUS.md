# Project status (as of 2026-10-19 06:45 UTC)

↑ [Project README](README.md) · [TASKS](TASKS.md) · [INDEX](INDEX.md) · [Owner Q&A](OWNER_QA.md) · [Status history](docs/archive/STATUS_history.md)

One page, plain words. Every number comes from the [INDEX](INDEX.md) row named after it; INDEX is the only place
where numbers and claim status are kept. "(unverified)" means no second agent has checked it. Older dated updates and
the long technical summary are in [docs/archive/STATUS_history.md](docs/archive/STATUS_history.md).

**The essay question (Goodfire, "Neural Networks Are Already Clean"): answered on our test.** Blinded explainers rate
about 65% of Qwen3-0.6B neurons as having "one main job" against about 21% of properly mixed random directions,
above the 30-point bar set in advance (H-Q5, confirmed, scoped; T6, T383). Part of the gap may be activation texture.
Readable neurons are still not enough to calculate what a network does (T24, T377; [Owner Q&A 4.2](OWNER_QA.md#q4-2)).

**How M_16,10 works: mostly understood, not precisely enough to calculate with.** A running-max tracker, a record
alarm, comparators and rotating "clocks" (H-M1..M8, [m16/FINDINGS.md](m16/FINDINGS.md)); the per-neuron account is
core-confirmed (T241), but no estimator has yet been improved by it (T338, T341).

**ARC's challenge (estimate M_16,10's accuracy from its weights, at least as precisely as sampling at equal compute):
open.**
- Best sampling-free calculation with declared approximations: T317f, F 2.63 on the sealed panel HP4: lenient bar
  (F <= 10) passed, strict bar (F <= 1) failed, retrained networks only, breaks under weight noise (H-L1, T337). The
  strict bar is dead (verified) for that whole approach, scoped to it (H-S1; T356, T357).
- Fixed-line walk quadratures sit at or just below the sampling line at 10 steps on 100 M_16,10 retrains (T10 recipe
  via T380, T389, T391), none confidently below it, so no sealed HP5 run yet; E33 tie-slicing loses at executed cost
  (T397). Whether ARC counts fixed-line averaging is unknown ([Owner Q&A 4.7, 4.8](OWNER_QA.md#q4-7)).
  At 16 neurons and 5 steps a fixed-directions walk beats sampling (H-E16, confirmed, scoped; T379).
- Exact region counting is ruled out for the target (H-E13; T270); exact computation survives as a component.

**Rare failures (the owner's priority; ARC's long-term goal).** For far-from-tie rare mistakes on ordinary networks
the estimator lands within about 2x of the truth, deep into the tail (H-R1, confirmed, scoped). Its failure mode, three-way
near ties, is partly fixed: the graded rule recovers 16/16 earlier misses on noised copies (T388; blind check T403) and
rates the target M_16,10 for gaps up to 1.5, but is far too low from gap 2 on (T407; closure fix T370).
**Bridge-style uses:** safe-use ratings work on two retrains and partly on the target (T401, T407); the weights driving
the rarest error are located (H-R2, confirmed), and a 12-weight edit makes that error ~17x rarer for -.011 pp accuracy,
while random edits of the same size fail 40/40 (H-R3, confirmed, scoped; T402, T410); it partly turns the rare mistake
into a common near-tie one, so a repair of all far-gap errors is running (T411). A sampling-free rare-event gradient is
still open (T386, T405 negative).
**Guaranteed upper bound** (T398, T400, T412): every pruning certificate re-verifies exactly, and the rigorous mass bound
is now fast (~100-500x, T412; audit T417) and tight in 10-D. But stepping through the network in time cannot prune the
240 'late' input orderings (T398), so a bound over all inputs needs a different split; an input-space test is running
(T414). A sampling-class bound says P <= 1.7e-9 at confidence 1 - 1e-6 for the gap >= 2.5 event (1 hit in 1e10
draws; 1e11 running).

**Running now** (owners in [TASKS.md](TASKS.md#in-progress)): closure fix (T370), blind tie check (T403), certified
bound (T398 per-triple, T400/T416 masses, T414 input split, T415 fast geometry, T417 audit, 1e11-draw sampling bound),
all-far-gap repair (T411), noisy-copy follow-ups (T404), chunked-input RNN (T392), ladder (T86/T111, T46). A sealed HP5
run happens only for a frozen candidate whose DEV3 confidence bound is below 1; none yet.

**Standing rules** that shape the work (no ARC contact, no Fable, CPU-first, tutorial paused) are in
[AGENTS.md](AGENTS.md#standing-rules).
