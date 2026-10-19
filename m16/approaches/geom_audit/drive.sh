#!/bin/bash
# T420 phase B driver: one process per cell (a GMP out-of-memory abort kills only that cell's process).
# usage: drive.sh I0 I1   -> results_b_I0.jsonl via audit.py; on a crash retries without the reference, then
# records {"j": j, "crash": ...} in crashes_b.jsonl
. /home/user/clean-neurons/.venv/bin/activate
cd "$(dirname "$0")"
ulimit -v 2500000
for ((j=$1; j<$2; j++)); do
  grep -q "\"j\": $j[,}]" results_b_$1.jsonl 2>/dev/null && continue
  python -W ignore - "$j" "$1" <<'PY' && continue
import json, sys, resource, audit
j, i0 = int(sys.argv[1]), sys.argv[2]
r = audit.run_cell(json.load(open("cells.json"))["cells"][j], True); r["j"] = j
r["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
open("results_b_%s.jsonl" % i0, "a").write(json.dumps(r) + "\n"); print(j, r.get("ref_mismatch"), (r.get("indep") or {}).get("ok"), flush=True)
PY
  echo "cell $j crashed with reference; retry noref"
  python -W ignore - "$j" "$1" <<'PY' && continue
import json, sys, resource, audit
j, i0 = int(sys.argv[1]), sys.argv[2]
r = audit.run_cell(json.load(open("cells.json"))["cells"][j], False); r["j"] = j; r["ref"] = "crashed (GMP/memory) under 2.5 GB"
r["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
open("results_b_%s.jsonl" % i0, "a").write(json.dumps(r) + "\n"); print(j, "noref", (r.get("indep") or {}).get("ok"), flush=True)
PY
  echo "{\"j\": $j, \"crash\": \"both runs crashed under 2.5 GB\"}" >> crashes_b.jsonl
done
