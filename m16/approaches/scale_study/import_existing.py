"""Import the models that already exist at grid points (T421): ARC's zoo (16,10), (32,10) seeds 0-4 (GPU stream) and
T392's k = 1 models (branch claude/t392-chunked-rnn; bit-identical to alg_zoo.training.train on CPU, checked in README).
usage: python import_existing.py   (needs data/zoo/2nd_argmax_<H>_10_s<s>.pth from storage.googleapis.com/arc-ml-public/alg/zoo)
"""
import json
import os
import subprocess

import torch as th

from common_io import save_model

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
for H in (16, 32):
    for s in range(5):
        url = f"https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_{H}_10_1073741824_{s}.pth"
        sd = th.load(os.path.join(ROOT, f"data/zoo/2nd_argmax_{H}_10_s{s}.pth"), map_location="cpu")
        save_model(f"m_{H}_10_zoo{s}", sd, H=H, T=10, seed=s, stream="zoo", provenance={"source": url})
for H, seeds in ((16, [0]), (24, [0, 1, 2]), (32, [0, 1, 2])):
    for s in seeds:
        p = f"m16/approaches/chunked_rnn/models/k1h{H}_s{s}.json"
        j = json.loads(subprocess.check_output(["git", "-C", ROOT, "show", f"origin/claude/t392-chunked-rnn:{p}"]))
        W = j["weights"]
        sd = {"rnn.weight_ih_l0": th.tensor(W["Wih"]), "rnn.weight_hh_l0": th.tensor(W["Whh"]),
              "linear.weight": th.tensor(W["Wout"])}
        save_model(f"m_{H}_10_cpu{s}", sd, H=H, T=10, seed=s, stream="cpu",
                   provenance={"source": f"T392 claude/t392-chunked-rnn:{p}", "wall_seconds": j["log"][-1][-1]})
