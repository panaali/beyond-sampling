"""T405 code checks (not scores; dev3_s528 is T372/T386's model): (1) frozen.t5 reproduces T386's stored T5/gl20 result on
dev3_s528 (est, 432-gradient); (2) the own sampler's base P on dev3_s528 E1(2.75) vs T372's truth 2.72e-9 +- 5e-11.
usage: python check_code.py"""
import json
import os
import sys

import numpy as np

import frozen
import sampler as S

m = frozen.dev3("dev3_s528"); ref = json.load(open(os.path.join(frozen.T386, "results", "cfg_dev3_s528_T5_gl20.json")))
r = frozen.t5(m, 2.75)
print("T5 s528 est", r["est"], "T386", ref["est"], "max |grad diff| / max |grad|",
      float(np.abs(r["grad"] - np.array(ref["grad"])).max() / np.abs(ref["grad"]).max()), flush=True)
pg, cdf = S.v2_table(2.75); rng = np.random.default_rng(405999); St = S.stack([m]); n = h = 0
while n < 1e7:
    x, p3 = S.draw(rng, 100000, 2.75, cdf); h += int((S.argmax_stack(St, x)[0] == p3).sum()); n += 100000
q = h / n; print("own sampler s528 E1(2.75): p_gap %.4e P %.4e +- %.1e (T372 2.72e-9 +- 5e-11)" % (pg, pg * q, pg * np.sqrt(q * (1 - q) / n)))
