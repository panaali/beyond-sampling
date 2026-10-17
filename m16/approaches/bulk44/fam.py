"""T165 part 1: bulk error families of zoo (H,T) models by MC (diagnostic only). usage: python fam.py H T [n_seqs]
Writes fam_H_T.json; prints <=10 lines."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "understand_m43"))
from zoo44 import zoo
H, T = int(sys.argv[1]), int(sys.argv[2])
out = {}
for s in range(5):
    Wi, Wh, Wo = zoo(H, T, s)
    x = np.random.default_rng(s).standard_normal((1_000_000, T)); h = np.zeros((len(x), H)); G = []
    for t in range(T):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0); G.append((h > 0) @ (1 << np.arange(H)))
    pred = (h @ Wo.T).argmax(1); o = np.argsort(-x, 1); tr = o[:, 1]; bad = pred != tr
    rank_of_pred = np.argmax(o == pred[:, None], 1)       # 0 = predicted the argmax, 2 = third, ...
    conf = np.zeros((T, T), int); np.add.at(conf, (tr[bad], pred[bad]), 1)
    gT = G[-1][bad]; vals, cnt = np.unique(gT, return_counts=True); top = np.argsort(-cnt)[:3]
    # which position holds the max / 2nd among errors
    fam = {"acc": float(1 - bad.mean()), "rank_of_pred": np.bincount(rank_of_pred[bad], minlength=T).tolist(),
           "conf_true_pred": conf.tolist(), "pred_freq": np.bincount(pred, minlength=T).tolist(),
           "final_gate_top": [[int(vals[i]), float(cnt[i] / bad.sum())] for i in top],
           "argmax_pos_err": np.bincount(o[bad, 0], minlength=T).tolist(), "Wi": Wi.tolist(), "Wh": Wh.tolist(), "Wo": Wo.tolist()}
    out[s] = fam
    print(f"s{s} acc {fam['acc']:.3f} predRank(err) {np.round(np.array(fam['rank_of_pred'])/bad.sum(),2)} predFreq {np.round(np.array(fam['pred_freq'])/len(x),2)} "
          f"err by true pos {conf.sum(1)*1.0/len(x)} topGate {fam['final_gate_top'][0]}")
json.dump(out, open(f"fam_{H}_{T}.json", "w"), indent=0)
