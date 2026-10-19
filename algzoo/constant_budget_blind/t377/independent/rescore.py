"""Independent rescore of T377 (written from scratch by checker; does not import project scoring code)."""
import importlib.util, json, os, sys
import numpy as np, torch
from alg_zoo import zoo_2nd_argmax, example_2nd_argmax

BASE = '/home/user/clean-neurons/algzoo/constant_budget_blind'
OUT = os.path.join(BASE, 't377/independent')
N, CHUNK = 200_000, 20_000
SEED_X, SEED_NOISE = 4242, 777

MODELS = {
    'M8':          (lambda: zoo_2nd_argmax(8, 10),            ['agsc', 'gcap', 'wfpn'], 'runs'),
    'M16':         (lambda: zoo_2nd_argmax(16, 10),           ['hnrt', 'wduk', 'yyuc'], 'runs'),
    'M32':         (lambda: zoo_2nd_argmax(32, 10),           ['dncx', 'krvc', 'sfbp'], 'runs'),
    'M16_s0':      (lambda: zoo_2nd_argmax(16, 10, seed=0),   ['bnic', 'rtiv'], 't377/runs'),
    'M16_s1':      (lambda: zoo_2nd_argmax(16, 10, seed=1),   ['briy', 'kzuy'], 't377/runs'),
    'M16_example': (lambda: example_2nd_argmax(),             ['pqid', 'phwb'], 't377/runs'),
    'M32_s1':      (lambda: zoo_2nd_argmax(32, 10, seed=1),   ['htmm', 'rdaq'], 't377/runs'),
    'M32_s2':      (lambda: zoo_2nd_argmax(32, 10, seed=2),   ['vbya', 'incg'], 't377/runs'),
    'M32_160call': (lambda: zoo_2nd_argmax(32, 10),           ['dqfw', 'vnna'], 't377/runs'),
}

def load_labels(path, name):
    spec = importlib.util.spec_from_file_location('lab_' + name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.f

torch.set_grad_enabled(False)
X = np.random.default_rng(SEED_X).standard_normal((N, 10))
y = np.argsort(-X, 1)[:, 1]
res = {'seed_x': SEED_X, 'seed_noise': SEED_NOISE, 'N': N, 'models': {}}
for mname, (ctor, codes, sub) in MODELS.items():
    model = ctor().double().eval()
    W = model.linear.weight.detach().numpy()
    H = W.shape[1]
    hs = []
    for s in range(0, N, CHUNK):
        xt = torch.from_numpy(X[s:s+CHUNK])
        out, _ = model.rnn(xt.unsqueeze(-1))
        hs.append(out[:, -1, :].numpy())
    h = np.concatenate(hs)
    logits_model = np.concatenate([model(torch.from_numpy(X[s:s+CHUNK])).numpy() for s in range(0, N, CHUNK)])
    acc_model = float((logits_model.argmax(1) == y).mean())
    accs = {}
    for code in codes:
        f = load_labels(os.path.join(BASE, sub, code, 'labels.py'), code)
        correct = 0
        for s in range(0, N, CHUNK):
            xc = X[s:s+CHUNK]
            P = np.stack([np.broadcast_to(np.asarray(f(i, xc), dtype=float), (len(xc),)) for i in range(H)], 1)
            correct += int(((P @ W.T).argmax(1) == y[s:s+CHUNK]).sum())
        acc_repl = correct / N
        accs[code] = {'acc_repl': acc_repl, 'gap': (acc_repl - 0.1) / (acc_model - 0.1)}
        print(mname, code, accs[code], flush=True)
    sd = h.std(0)
    rng = np.random.default_rng(SEED_NOISE)
    Fs = []
    for d in range(3):
        hn = h + rng.standard_normal(h.shape) * np.sqrt(0.1) * sd
        a = float(((hn @ W.T).argmax(1) == y).mean())
        Fs.append((a - 0.1) / (acc_model - 0.1))
    gbar = float(np.mean([v['gap'] for v in accs.values()]))
    res['models'][mname] = {'H': H, 'acc_model': acc_model, 'accounts': accs, 'g_bar': gbar,
                            'F_draws': Fs, 'F': float(np.mean(Fs))}
    print(mname, 'acc', acc_model, 'F', Fs, 'gbar', gbar, flush=True)
    del h, hs, logits_model
json.dump(res, open(os.path.join(OUT, 'stage1.json'), 'w'), indent=1)
