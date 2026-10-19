"""Exact small-horizon reference for the soundness check (T447): every L = 15 (dyck2) / n <= 10 (modsum) sequence's
max_t p_illegal by the harness's own metrics.per_sequence (float32 model), keeping the sequences above 0.4 (all that
can matter for tau >= 0.5) and the exact B_tau. Usage: python ref.py dyck2_pos_s0 [...]  -> data/ref_<name>.npz"""
import os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mini_lm"))
from load import load
from metrics import per_sequence, TAUS

torch.set_num_threads(2)
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
for name in sys.argv[1:]:
    t0 = time.time()
    model, lang = load(name)
    ys, ms, ws, B = [], [], [], {t: 0.0 for t in TAUS}
    for y, w in lang.enumerate_chunks(chunk=100000):
        mx = per_sequence(model, lang, y)["max_ill"]
        for t in TAUS:
            B[t] += float((w * (mx > t)).sum())
        k = mx > 0.4
        ys.append(y[k].astype(np.int8)); ms.append(mx[k]); ws.append(w[k])
    np.savez_compressed(os.path.join(HERE, "data", f"ref_{name}.npz"), y=np.concatenate(ys), max_ill=np.concatenate(ms),
                        w=np.concatenate(ws), **{f"B{t}": B[t] for t in TAUS})
    print(name, {t: B[t] for t in TAUS}, "kept", sum(len(m) for m in ms), "%.0fs" % (time.time() - t0), flush=True)
