"""T234 add-on: H = 16 accuracy for T = 2..10 on one consistent MC truth (1e8 float64 samples per model, seed 777).
Zoo models (alg_zoo.zoo_2nd_argmax(16, T, seed), seeds 0-4) for T in the zoo; zoo-code-trained models (train_zoo.py)
for T = 7, 9. usage: python ht_table.py [n] -> results/ht_table.json"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from evaluate import evaluate
n = int(float(sys.argv[1])) if len(sys.argv) > 1 else 10 ** 8
out = {}
import alg_zoo
for T in range(2, 11):
    models, names = [], []
    for s in range(5):
        try:
            m = alg_zoo.zoo_2nd_argmax(16, T, seed=s); sd = {k: v.detach().double().numpy() for k, v in m.state_dict().items()}
            models.append((sd["rnn.weight_ih_l0"][:, 0], sd["rnn.weight_hh_l0"], sd["linear.weight"])); names.append(f"zoo{s}")
        except Exception:
            f = os.path.join(HERE, "weights", f"zoo_code_H16T{T}_seed{s}.npz")
            if os.path.exists(f):
                z = np.load(f); models.append((z["Wi"], z["Wh"], z["Wo"])); names.append(f"zoocode{s}")
    if models:
        ev = evaluate(models, n)
        out[T] = {nm: e for nm, e in zip(names, ev)}
        print(T, " ".join("%s %.5f" % (nm, e["acc"]) for nm, e in zip(names, ev)),
              "mean %.5f" % np.mean([e["acc"] for e in ev]), flush=True)
json.dump(out, open(os.path.join(HERE, "results", "ht_table.json"), "w"), indent=1)
