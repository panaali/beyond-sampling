"""Run the unchanged shared scorer, preserving its 200k held-out protocol."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

if __name__ == '__main__':
    timings = {}
    for name, label, extra, hidden in [
        ('frozen_16_10', REPO/'algzoo/labels.py', [], 16),
        ('frozen_32_10', ROOT/'e5_labels.py', ['--zoo', '32,10'], 32),
    ]:
        output = ROOT/f'{name}.json'
        sys.argv = [str(REPO/'algzoo/score.py'), str(label), str(output), *extra]
        c0, w0 = time.process_time(), time.perf_counter()
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            runpy.run_path(str(REPO/'algzoo/score.py'), run_name='__main__')
        timings[name] = {'process_cpu_seconds': time.process_time()-c0,
                         'wall_seconds': time.perf_counter()-w0,
                         'hidden_forward_sequences': 200000,
                         'full_forward_sequences': 200000,
                         'single_neuron_replacements_scored': hidden*200000}
        print(name, output.read_text(), flush=True)
    (ROOT/'frozen_compute.json').write_text(json.dumps(timings, indent=2)+'\n')
