"""Verify T398 certificate files (format v1). usage: python cert_run.py FILE... -> results_cert_<name>.json"""
import json
import os
import sys

import numpy as np

import verify as V

for f in sys.argv[1:]:
    c = np.load(f, allow_pickle=True)
    ok, ts = V.verify_certfile(f, return_times=True)
    st = c["stage"]
    rec = dict(file=os.path.basename(f), cells=len(ok), verified=int(ok.sum()), failed=[int(i) for i in np.where(~ok)[0]],
               by_stage={int(s): [int((st == s).sum()), int(ok[st == s].sum()), float(np.median(ts[st == s])) if (st == s).any() else None]
                         for s in np.unique(st)},
               t_total=float(ts.sum()), t_median=float(np.median(ts)), t_p90=float(np.percentile(ts, 90)))
    print(rec, flush=True)
    json.dump(rec, open("results_cert_%s.json" % os.path.basename(f)[:-4], "w"), indent=0)
