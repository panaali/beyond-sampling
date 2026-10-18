# Question
Does the reviewed NumPy recurrence match the original library RNN on one fixed checkpoint prefix?

# Method
The frozen pilot compares the same stored binary weights converted to float64, zero initial hidden state and unchanged linear readout on exactly 1,024 inputs. The library hidden state comes from a hook on its single forward. All inputs, candidate/library hidden states, logits, labels, gaps, weights and CPU RNG states are retained. Source/resource review f568e61c and immutable gate e177a123 were published before execution (shared2cb86c93).

# Status
Pilot completed once and independently saved-audited in75da4bad, now published. No full fit or holdout run.

# Results
All 1,024 predicted labels match. Maximum absolute hidden difference 4.973799150320701e-14; maximum logit difference 2.9665159217984183e-13. Both are below the predeclared reference-scaled tolerance. CPU RNG was unchanged through meta construction, assignment and forward. No input ties, resampling or surrogate fits. Results are finite wiring checks, not outward arithmetic certificates or performance estimates.

# Compute accounting
Exactly 1,024 distinct input vectors (10,240 input normal values), one candidate forward of 1,024 sequences and one library forward of the same 1,024. Script CPU 0.849919784 seconds; Bash timing and import/loading overhead are retained in ../pilot_run1.log. No surrogate fits. One thread, primary lock, 110 CPU-second external cap, 90-second soft guard, declared 2 GiB RSS check and 2,500,000 KiB virtual-memory limit.

# Claims
[measured] Frozen finite implementation parity and independent saved audit pass. [inferred] This does not validate the historical 98%/90–91% reconstruction rates or pass an ARC rung.

# Alternatives not ruled out
Rare floating label differences, real-data fit deficiencies, omitted-information reconstruction error and independent holdout variability remain open. Future training requires a new scope, must reuse this exact prefix once and resume the saved PCG64 state without a new prefix forward.

# Failure modes
Meta/assignment failure, input ties, label mismatch and resource exhaustion would have stopped the run and retained partial results. None occurred. The reference is a separate library implementation of the same recurrence, not an exact-real oracle.

Command from parent directory: `flock /workspace/work/codex-compute.lock bash -c 'ulimit -t 110; time -p /workspace/clean-neurons/.venv/bin/python pilot.py' > pilot_run1.log 2>&1`, with outer virtual cap and OPENBLAS/OMP/MKL threads1. Do not rerun.

Independent review checked all eight archives, all 1,024 state/true/predicted labels and gaps, weight-conversion hashes, full RNG equality, counters and progress. Reviewer CPU was 0.019670135 seconds excluding imports; no checkpoint load, forward or random draw. Terminal RNG advancement and checkpoint deserialization were not independently replayed and remain source-gated premises.
