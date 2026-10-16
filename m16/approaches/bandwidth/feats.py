"""Feature sets and model families for predicting the band offset from (tie configuration, other inputs)."""
import numpy as np
import torch

CL = 0.3


def load(nm, big=False):
    R = dict(np.load(f"data/{nm}{'_big' if big else ''}.npz"))
    k = R["v"] > 0
    return {key: val[k] for key, val in R.items() if val.shape[:1] == k.shape}


def targets(R):
    return {"delta": np.clip(R["delta"], -CL, CL), "absdelta": np.clip(np.abs(R["delta"]), 0, CL), "mass": R["mass"]}


def sorted_lower(R):
    y = R["y"].copy(); n = len(y); r = np.arange(n)
    y[r, R["a"]] = np.inf; y[r, R["b"]] = np.inf; y[r, R["p1"]] = np.inf
    s = np.sort(y, 1)[:, :7][:, ::-1]           # 7 lower inputs, descending
    return s


def cell(R, kind):
    a, b, p1 = R["a"], R["b"], R["p1"]
    if kind == "ab":
        return a * 10 + b
    if kind == "abp":
        return (a * 10 + b) * 10 + p1
    return np.zeros(len(a), int)


def design(R, kind):
    n = len(R["v"]); one = np.ones((n, 1))
    if kind == "const":
        return one
    if kind == "roles":                 # v, max, 7 sorted lower values
        return np.column_stack([one, R["v"], R["o1"], sorted_lower(R)])
    if kind == "raw":                   # values by position
        return np.column_stack([one, R["y"]])
    if kind == "rawquad":
        y = R["y"]; iu = np.triu_indices(10)
        return np.column_stack([one, y, (y[:, :, None] * y[:, None, :])[:, iu[0], iu[1]]])
    raise ValueError(kind)


def fit_cells(Rtr, ytr, Rte, ckind, dkind, lam=1e-2):
    """Ridge-regularised linear model in design(dkind), separately per cell(ckind). Returns predictions, n_params."""
    Xtr, Xte = design(Rtr, dkind), design(Rte, dkind)
    ctr, cte = cell(Rtr, ckind), cell(Rte, ckind)
    pred = np.full(len(Xte), ytr.mean()); npar = 0
    # pooled fit as fallback / prior
    A = Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1]) * 1e-3; A[0, 0] -= lam * len(Xtr) * 1e-3
    wg = np.linalg.solve(A, Xtr.T @ ytr)
    for c in np.unique(ctr):
        i = ctr == c; j = cte == c
        X = Xtr[i]; p = X.shape[1]
        # ridge toward pooled coefficients, strength lam*p
        w = np.linalg.solve(X.T @ X + lam * p * np.eye(p) * max(1, 50 / i.sum()) * 10, X.T @ ytr[i] + lam * p * max(1, 50 / i.sum()) * 10 * wg)
        npar += p
        if j.any():
            pred[j] = Xte[j] @ w
    return pred, npar


def r2(p, t):
    return 1 - np.mean((t - p) ** 2) / np.var(t)


def onehot_inputs(R):
    n = len(R["v"]); E = np.eye(10)
    return np.column_stack([E[R["a"]], E[R["b"]], E[R["p1"]], R["y"], R["v"][:, None], R["o1"][:, None],
                            sorted_lower(R)]).astype(np.float32)


def mlp(Rtr, ytr, Rte, hidden=256, epochs=60, seed=0, wd=1e-5):
    torch.manual_seed(seed); torch.set_num_threads(2)
    Xtr = torch.tensor(onehot_inputs(Rtr)); Xte = torch.tensor(onehot_inputs(Rte))
    mu, sd = ytr.mean(), ytr.std()
    Y = torch.tensor(((ytr - mu) / sd).astype(np.float32))
    net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=wd)
    n = len(Xtr); bs = 512
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * ((n + bs - 1) // bs))
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = ((net(Xtr[idx])[:, 0] - Y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        p = net(Xte)[:, 0].numpy() * sd + mu
    return p, sum(q.numel() for q in net.parameters())
