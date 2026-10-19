"""1e8-input MC truths per registry model (T393's compiled float64 sampler, seed crc32(tag)) -> truths.jsonl.
usage: python truths.py [worker_index n_workers]"""
import sys
import time
import zlib

import numpy as np

import common as Cm

Cm._check()
import sampler_t393 as S  # noqa: E402

N = 10 ** 8
wi, nw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
done = {r["tag"] for r in Cm.jl_read("truths.jsonl")}
todo = sorted((r for r in Cm.registry() if r["tag"] not in done), key=lambda r: (r["H"] ** 2 * r["T"], r["tag"]))
for r in todo[wi::nw]:
    m = Cm.load(r["tag"]); t0 = time.time()
    p = S.est_compiled(m, N, np.random.default_rng(zlib.crc32(r["tag"].encode())), chunk=1 << 20)
    row = {"tag": r["tag"], "H": r["H"], "T": r["T"], "acc": p, "se": float(np.sqrt(p * (1 - p) / N)), "n": N,
           "sha256_npz": Cm.sha_npz(r["tag"]), "secs": round(time.time() - t0, 1)}
    Cm.jl_add("truths.jsonl", row); print(row, flush=True)
