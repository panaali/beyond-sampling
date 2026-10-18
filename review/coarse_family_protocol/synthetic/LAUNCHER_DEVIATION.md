# Preserved resource-launcher deviation — sole completed attempt

The immutable gate743bdf16 was publishedb17233ba before this sole run. The shell
was invoked as follows (stdout/stderr of Python alone went to run1.log):

```sh
set -o noclobber
ulimit -Hv 2097152
ulimit -Sv 2097152
ulimit -Ht 20
ulimit -St 20
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 flock /workspace/work/codex-review.lock /workspace/clean-neurons/.venv/bin/python review/coarse_family_protocol/synthetic/controls.py > review/coarse_family_protocol/synthetic/run1.log 2>&1
```

The tool returned these shell diagnostics before Python ran:

```
/bin/bash: line 2: ulimit: virtual memory: cannot modify limit: Invalid argument
/bin/bash: line 4: ulimit: cpu time: cannot modify limit: Invalid argument
```

The hard limits were requested before lowering existing soft limits, so those
commands failed. The shell did not have errexit; it continued. The following
soft-limit commands succeeded, and the Python harness verified positive soft
RLIMIT_AS<=2GiB and RLIMIT_CPU<=20 before creating outputs. The external HARD
limit requirement was therefore not established; do not claim full launcher
conformance. Existing hard limits were not recorded. The internal17CPU limit
was checked before each case. The numerical source/fixtures/gates were unchanged.

The sole run completed all8 cases/10 guards/39events in.098147731 mainCPU and
55396KiB peakRSS. No resource excess is observed. All outputs are retained,
including empty Python run1.log; this separate note preserves the shell errors
which were outside that redirection. The program exited0, not proof that every
preceding shell command succeeded. No successful control will be repeated.

Future launches must set both soft and hard limits with the default `ulimit`
form or lower soft before hard, fail immediately on any setup error, and record
both actual limit pairs before scientific calls. This is a launcher correction,
not a numerical algorithm change or reason to discard these saved records.
Independent review must keep this qualification separate from numerical parity.
