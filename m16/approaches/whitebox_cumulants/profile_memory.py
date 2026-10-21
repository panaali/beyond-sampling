"""Diagnostic line-level peak-RSS attribution; never a scoring run."""
import json
from pathlib import Path
import resource
import sys

import run

source = Path(json.loads((run.HERE / "sources.json").read_text())["v25"]["local_path"])
records = []
peak, previous = 0, None


def trace(frame, event, arg):
    global peak, previous
    if frame.f_code.co_filename != str(source):
        return None
    if event == "line":
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if sys.platform == "darwin" else 1024
        if rss > peak + 1_000_000:
            records.append({"after_line": previous, "next_line": frame.f_lineno,
                            "peak_rss_bytes": rss, "layer": frame.f_locals.get("li")})
            peak = rss
        previous = frame.f_lineno
    return trace


class Args:
    method = "v25"
    width = 512
    depth = 16
    seed = 2026
    output = run.HERE / "results" / "memory_diagnostic_512.json"


sys.settrace(trace)
try:
    run.worker(Args())
finally:
    sys.settrace(None)
    path = run.HERE / "results" / "memory_lines_512.json"
    path.write_text(json.dumps(records, indent=2)+"\n")
    print(records[-12:])
