"""D1 (diagnostic MC): oracle degree-d Hermite projection floor of the TRUE logits on the screen models."""
import json, sys
import numpy as np
from run import load_panel, oracle_floor, SCREEN

z, names, truths = load_panel()
ds = [int(v) for v in sys.argv[2:]]
nm = sys.argv[1]
i = names.index(nm)
model = (z["Wi"][i].astype(float), z["Wh"][i].astype(float), z["Wo"][i].astype(float))
out = []
for d in ds:
    r = oracle_floor(model, d, np.random.default_rng(29200 + 10 * i + d))
    r.update(name=nm, truth=truths[nm]["acc"]); out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(f"floor_{nm}.json", "w"), indent=1)
