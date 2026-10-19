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
the accurate estimator lands within about 2x of the truth, deep into the tail (H-R1, confirmed, scoped; T360, T368).
It is not yet cheaper than sampling at equal compute, and three-way near-tie networks remain the failure mode; the
closure fix is running (T370).

**Running now** (see [TASKS.md](TASKS.md#in-progress) for owners): rare-event closure fix (T370), quantised
networks (T352), all-at-once network and its check (T345, T394), chunked-input RNN (T392), compiled walk (T393),
de-aliased rare-event gradient (T386), verification-style bounds for a certified rare-event bound (T398, after T390's
STOP), GPU error-pattern census (T395), the ladder (T86/T111, T46). A sealed HP5 run happens only for a frozen
candidate whose DEV3 confidence bound is below 1.

**Standing rules** that shape the work (no ARC contact, no Fable, CPU-first, tutorial paused) are in
[AGENTS.md](AGENTS.md#standing-rules).
