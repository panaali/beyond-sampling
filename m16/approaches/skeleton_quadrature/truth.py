"""Reference accuracy of a DEV model by plain sampling (DIAGNOSIS ONLY). usage: truth.py MODEL [N_millions]"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import rv  # noqa: E402

name = sys.argv[1]
N = int(float(sys.argv[2]) * 1e6) if len(sys.argv) > 2 else 20_000_000
model = rv.dev_model(name)
rng = np.random.default_rng(4004)
ok = 0
for a in range(0, N, 200000):
    x = rng.standard_normal((200000, 10))
    ok += int((rv.run(model, x)[0].argmax(1) == np.argsort(-x, 1)[:, 1]).sum())
p = ok / N
out = dict(model=name, N=N, p=p, se=float(np.sqrt(p * (1 - p) / N)))
print(out)
json.dump(out, open(f"{__file__.rsplit('/', 1)[0]}/truth_{name}.json", "w"))
