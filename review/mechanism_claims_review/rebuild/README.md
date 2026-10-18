# Question
Do the fixed reconstruction helpers implement the declared state, design, fitting and scoring conventions?

# Method
`core.py` and `controls.py` cover216 legal states, known homogeneous lookup coefficients, the18-feature case2 map, domain/rank guards, score/tie rules, an independently labeled cross-tab and six exact dyadic recurrence inputs. Published independent source gate e0281c91 preceded controls_gate a9e5a3d5.

# Status
All seven synthetic groups completed once and independently audited in a944f1aa, published186907c3. No checkpoint expansion yet.

# Results
`controls_results.json` and nine progress records preserve all outputs. Known coefficients/predictions satisfy the fixed1e-10 control tolerance. Exact synthetic recurrence outputs agree. No trained/historical performance claim.

# Compute accounting
219 lstsq calls (including two intentional rank-deficient guards), one synthetic forward batch of six sequences, zero actual checkpoint calls and zero random inputs. Script CPU0.219838866s; whole-process Bash timing is in run2.log. One thread,20CPU/2GB review lock. Initial run1.log records exit127 before Python because /usr/bin/time was absent; independently cleared launcher-only recovery used Bash time, no completed scientific work was repeated.

# Claims
[measured] Frozen synthetic controls pass, with independent source and saved verification. Saved coefficient/prediction arrays were not serialized: their residual maxima are source-gated author summaries, not independently refitted residuals. [inferred] This does not establish actual-model reconstruction performance or a challenge rung.

# Alternatives not ruled out
Real-data rank deficiencies, omitted-information prediction errors and fit variability remain open. A separately scoped checkpoint pilot is required before the proposed held-out rebuild.

# Failure modes
Finite floating tolerances are not outward certificates. Input ties stop without resampling. Zero-variance R2 is None, negative R2 remains allowed, and target accuracy is distinct from network agreement.

Command: `flock /workspace/work/codex-review.lock bash -c 'ulimit -t 20; time -p /workspace/clean-neurons/.venv/bin/python controls.py' > run2.log 2>&1`, outer2500000KB memory limit and OPENBLAS/OMP/MKL threads1. Do not rerun completed controls.

The independent saved audit reconstructed the 864 legal-key observations, 216 states, score table and six exact recurrence oracles without author calls. It checked all 219 recorded solves and singular-value invariants; reviewer parent CPU was 0.005030145 seconds, with Git-child work separate.
