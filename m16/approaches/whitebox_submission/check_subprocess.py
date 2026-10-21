"""Bounded preparation harness; never shipped with the estimator."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import psutil

HERE = Path(__file__).resolve().parent
env = {k: v for k, v in os.environ.items() if not re.match(r'^(V\d+_|X_)', k)}
env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
command = [str(Path(sys.executable).with_name('whest')), 'run', '--estimator',
           str(HERE/'package/estimator.py'), '--runner', 'subprocess',
           '--n-mlps', '1', '--n-samples', '128', '--seed', '479',
           '--max-threads', '1', '--wall-time-limit', '120',
           '--residual-wall-time-limit', '0.4', '--format', 'json', '--detail', 'full']
variant = sys.argv[1] if len(sys.argv) > 1 else 'v25'
if variant != 'v25':
    command[command.index('--estimator')+1] = str(HERE/('package_'+variant)/'estimator.py')
prefix = variant+'_'
start = time.monotonic()
peak = 0
reason = None
with (HERE/('results/'+prefix+'subprocess.json')).open('w') as out, (HERE/('results/'+prefix+'subprocess.stderr')).open('w') as err:
    child = subprocess.Popen(command, env=env, stdout=out, stderr=err)
    root = psutil.Process(child.pid)
    owned = {}
    while child.poll() is None:
        try:
            processes = [root] + root.children(recursive=True)
            for process in processes:
                owned[process.pid] = process
                try:
                    rss = process.memory_info().rss
                    peak = max(peak, rss)
                    if rss > 8_000_000_000:
                        reason = 'per-process RSS cap'
                except psutil.NoSuchProcess:
                    pass
            if time.monotonic()-start > 240:
                reason = '240-second wall cap'
            if reason:
                for process in reversed(list(owned.values())):
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
        except psutil.NoSuchProcess:
            pass
        time.sleep(.02)
    code = child.wait()
report = dict(command=command, exit_code=code, stop_reason=reason,
              max_process_rss_bytes=peak, wall_seconds=time.monotonic()-start)
(HERE/('results/'+prefix+'supervisor.json')).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
sys.exit(code or bool(reason))
