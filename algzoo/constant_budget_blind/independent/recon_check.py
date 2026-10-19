"""Independent reconstruction check for T24: model acc, replacement acc, gap recovered, per-neuron R^2."""
import importlib.util, json, sys, os
import numpy as np, torch
from alg_zoo import zoo_2nd_argmax
torch.set_num_threads(1)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {'agsc': 8, 'gcap': 8, 'wfpn': 8, 'hnrt': 16, 'wduk': 16, 'yyuc': 16, 'dncx': 32, 'krvc': 32, 'sfbp': 32}
SEED, N, CH = 77123, 200_000, 20_000

def load(code):
    p = os.path.join(BASE, 'runs', code, 'labels.py')
    spec = importlib.util.spec_from_file_location(f'lab_{code}', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

out = {}
for code, H in RUNS.items():
    model = zoo_2nd_argmax(H, 10).eval()
    W = model.linear.weight.detach().double().numpy()  # (10,H)
    lab = load(code)
    rng = np.random.default_rng(SEED)  # same inputs for every run
    mc = rc = 0; sse = np.zeros(H); s1 = np.zeros(H); s2 = np.zeros(H)
    for s in range(0, N, CH):
        x = rng.standard_normal((CH, 10)).astype(np.float32)
        tgt = np.argsort(x, 1)[:, -2]
        with torch.no_grad():
            xt = torch.from_numpy(x)
            _, hn = model.rnn(xt.unsqueeze(-1))
            h = hn.squeeze(0).double().numpy()
            logits = model(xt).double().numpy()
        mc += (logits.argmax(1) == tgt).sum()
        xd = x.astype(np.float64)
        F = np.stack([np.asarray(lab.f(i, xd), float).reshape(-1) for i in range(H)], 1)
        rc += ((F @ W.T).argmax(1) == tgt).sum()
        sse += ((F - h) ** 2).sum(0); s1 += h.sum(0); s2 += (h ** 2).sum(0)
    sst = s2 - s1 ** 2 / N
    r2 = 1 - sse / sst
    ma, ra = mc / N, rc / N
    out[code] = dict(H=H, model_acc=float(ma), repl_acc=float(ra), gap=float((ra - .1) / (ma - .1)),
                     r2=[float(v) for v in r2], mean_r2=float(r2.mean()),
                     labels_cat={int(k): v[0] for k, v in lab.LABELS.items()})
    print(code, H, round(ma, 4), round(ra, 4), round(out[code]['gap'], 4), round(r2.mean(), 3), flush=True)
json.dump(dict(seed=SEED, N=N, runs=out), open(os.path.join(os.path.dirname(__file__), 'recon.json'), 'w'), indent=1)
