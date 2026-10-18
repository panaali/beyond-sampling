"""T270 diagnosis (DRI checks a + b): for one k = 6 subtree, independent MC of the root cell and of every leaf region.
Samples x ~ N(0, I_6) (prefix inputs x_0..x_5), forward M_16,10, gate history = packbits(h_t > 0) for t = 1..6 (the
enumeration's region key). Root MC = samples whose history starts with the root's history; leaf MC = per region;
'unmatched' = samples inside the root whose full history is not an enumerated region (lost geometry).
usage: python mc6.py POS [n_samples] [seed] -> results/prefix6/mc6_<pos>.json"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import prefix6 as Q, prefix as P, gmass  # noqa: E402
pos = int(sys.argv[1]); N = int(float(sys.argv[2])) if len(sys.argv) > 2 else 10 ** 7
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2700000 + pos
model, front = Q.front_list(11); P._init(model, Q.K, True); item = front[pos]; t0 = time.time()
n, (hists, Rs, Is), o, s, sec = P._dfs(item)
cm = gmass.grouped(Rs, Is, len(hists), os.environ.get("DIAG_DEV", "cpu"), chunk=1024, tb=512); root = float(np.load(os.path.join(Q.OUT, "roots.npz"))["mass"][pos])
rh = np.frombuffer(item[4], np.uint8); L = len(rh); idx = {h: i for i, h in enumerate(hists)}
Wi, Wh, _ = model; rng = np.random.default_rng(seed); cnt = np.zeros(len(hists), np.int64); nroot = 0; unm = {}
done = 0
while done < N:
    m = min(10 ** 6, N - done); x = rng.standard_normal((m, Q.K)); h = np.zeros((m, 16)); H = []
    for t in range(Q.K):
        h = np.maximum(h @ Wh.T + x[:, t:t + 1] * Wi[None], 0); H.append(np.packbits(h > 0, axis=1))
    H = np.concatenate(H, 1); sel = (H[:, :L] == rh).all(1); nroot += int(sel.sum())
    for row in H[sel]:
        b = row.tobytes(); i = idx.get(b)
        if i is None:
            unm[b] = unm.get(b, 0) + 1
        else:
            cnt[i] += 1
    done += m
p = cnt / N; cmz = np.nan_to_num(cm); se = np.sqrt(np.maximum(np.maximum(cmz, p), 1e-300) / N); z = (p - cmz) / se
big = np.argsort(-np.abs(z))[:15]
r = dict(pos=pos, n=N, seed=seed, cells=n, regions=len(hists), root_mass=root, leaf_sum=float(cm.sum()),
         root_mc=nroot / N, root_mc_se=float(np.sqrt(nroot) / N), leaf_mc_sum=float(cnt.sum() / N),
         unmatched_mc=float(sum(unm.values()) / N), unmatched_patterns=len(unm),
         n_zero_or_nan=int((~(cm > 0)).sum()), mc_in_zero_or_nan=float(cnt[~(cm > 0)].sum() / N),
         mass_nan=int(np.isnan(cm).sum()), chi2_per_region=float(np.mean(z ** 2)), worst=[dict(i=int(i), mass=float(cm[i]), mc=float(p[i]), z=float(z[i]))
                                                         for i in big], sec=time.time() - t0)
json.dump(r, open(os.path.join(Q.OUT, f"mc6_{pos}.json"), "w"), indent=1)
print({k: v for k, v in r.items() if k != "worst"}); print(r["worst"][:5])
