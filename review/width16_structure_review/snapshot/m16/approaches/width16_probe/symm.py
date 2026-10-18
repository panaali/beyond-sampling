"""T216: weight-symmetry scan of the target M_16,10 (alg_zoo.example_2nd_argmax()). For each neuron pair (i, j),
cosines of the incoming vectors u_i = [W_ih[i], W_hh[i, :]] and outgoing vectors v_i = [W_hh[:, i], W_out[:, i]],
against +/- the other neuron's vector. Incoming rows are compared with the columns permuted to swap i <-> j
(a symmetry permutation must also permute the inputs it reads), and also unpermuted. Scale-invariant: the model has no
biases, so neuron i can be rescaled by a > 0 (u_i *= a, v_i /= a) without changing the function; cosines are invariant.
-> symm.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE]
import numpy as np
from alg_zoo import example_2nd_argmax
m = example_2nd_argmax().eval()
Wi = m.rnn.weight_ih_l0.detach().numpy()[:, 0].astype(float); Wh = m.rnn.weight_hh_l0.detach().numpy().astype(float)
Wo = m.linear.weight.detach().numpy().astype(float); H = len(Wi)
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
rows = []
for i in range(H):
    for j in range(i + 1, H):
        P = np.arange(H); P[[i, j]] = [j, i]
        ui, uj = np.r_[Wi[i], Wh[i]], np.r_[Wi[j], Wh[j]]
        uj_p = np.r_[Wi[j], Wh[j][P]]                      # neuron j's row with its reads of i, j swapped
        vi, vj = np.r_[Wh[:, i], Wo[:, i]], np.r_[Wh[:, j], Wo[:, j]]
        vj_p = np.r_[Wh[P, j], Wo[:, j]]                   # neuron j's outgoing column with targets i, j swapped
        rows.append(dict(i=i, j=j, cos_in=cos(ui, uj), cos_in_perm=cos(ui, uj_p), cos_out=cos(vi, vj), cos_out_perm=cos(vi, vj_p)))
keys = ("cos_in", "cos_in_perm", "cos_out", "cos_out_perm")
near = [r for r in rows if any(abs(r[k]) > 0.9 for k in keys)]
top = {k: sorted(rows, key=lambda r: -abs(r[k]))[:5] for k in keys}
both = sorted(rows, key=lambda r: -min(abs(r["cos_in_perm"]), abs(r["cos_out_perm"])))[:5]
dead = dict(zero_in=[int(i) for i in range(H) if np.linalg.norm(np.r_[Wi[i], Wh[i]]) < 1e-6],
            zero_out=[int(i) for i in range(H) if np.linalg.norm(np.r_[Wh[:, i], Wo[:, i]]) < 1e-6])
out = dict(H=H, n_pairs=len(rows), near_pairs_09=near, top5=top, top5_both_perm=both, dead=dead,
           max_abs={k: max(abs(r[k]) for r in rows) for k in keys})
json.dump(out, open(os.path.join(HERE, "symm.json"), "w"), indent=1)
print("max |cos|", out["max_abs"]); print("near pairs (|cos|>0.9, any measure):", [(r["i"], r["j"], {k: round(r[k], 3) for k in keys}) for r in near])
print("top by min(|in_perm|,|out_perm|):", [(r["i"], r["j"], round(r["cos_in_perm"], 3), round(r["cos_out_perm"], 3)) for r in both]); print(dead)
