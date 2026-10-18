"""T352: stacked QAT trainer for quantised M_16,10-architecture RNNs (zoo recipe, see README pre-registration).

K models trained at once, each with its own zoo init (seed), own N(0, I) data stream, activation code
(0 float relu, 1 binary {0,1} with hard-tanh STE, 2 int4 LSQ, 3 int8 LSQ) and input grid (0 continuous, 16 or 256
uniform levels on [-3, 3], label on the discretised inputs with ties broken by the lower index). AdamW per model
(torch formula), zoo LR schedule; the LSQ step s is a trained parameter (no weight decay).
usage: python qtrain.py TAG --spec name:seed:act:grid ... [--log2n 30] [--device cuda] -> TAG.pt (weights, s, meta)"""
import argparse, json, math, os, sys, time
import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "capacity_ceiling"))
import trainer as TR  # noqa: E402  (zoo_init, lr_mult)

H, T = 16, 10
LEVELS = {0: 0.0, 1: 0.0, 2: 15.0, 3: 255.0}                  # act code -> L - 1


class BinSTE(th.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g * (u.abs() <= 1).to(g.dtype)


class LSQ(th.autograd.Function):
    """q = s * clip(round(u / s), 0, Lm) per model; u (K, B, H), s (K,), Lm (K,)"""
    @staticmethod
    def forward(ctx, u, s, Lm):
        r = u / s[:, None, None]; q = th.minimum(th.clamp(th.round(r), min=0.0), Lm[:, None, None])
        ctx.save_for_backward(r, s, Lm)
        return q * s[:, None, None]

    @staticmethod
    def backward(ctx, g):
        r, s, Lm = ctx.saved_tensors; L3 = Lm[:, None, None]
        inside = (r >= 0) & (r <= L3)
        gu = g * inside.to(g.dtype)
        ds = th.where(inside, th.round(r) - r, th.where(r > L3, L3.expand_as(r), th.zeros_like(r)))
        scale = 1.0 / th.sqrt(th.tensor(float(r.shape[1] * r.shape[2]), device=r.device) * th.clamp(Lm, min=1.0))
        return gu, (g * ds).sum((1, 2)) * scale, None


def quant(u, act, s, Lm):
    """per-model activation: act (K,) long; returns (K, B, H)"""
    out = th.relu(u)
    if (act == 1).any():
        out = th.where((act == 1)[:, None, None], BinSTE.apply(u), out)
    if (act >= 2).any():
        out = th.where((act >= 2)[:, None, None], LSQ.apply(u, s, Lm), out)
    return out


def discretise(x, grid):
    """grid (K,) levels (0 = continuous); uniform levels on [-3, 3]"""
    if not (grid > 0).any():
        return x
    G = grid.to(x.dtype)[:, None, None]; step = 6.0 / th.clamp(G - 1, min=1)
    xq = th.clamp(th.round((x + 3.0) / step), min=0) ; xq = th.minimum(xq, G - 1) * step - 3.0
    return th.where((grid > 0)[:, None, None], xq, x)


def labels(x):
    """2nd-largest position, ties by the lower index (stable descending)"""
    key = x - th.arange(x.shape[-1], device=x.device, dtype=x.dtype) * 1e-7
    return key.topk(2, -1).indices[..., 1]


def forward(Wi, Wh, Wo, x, act, s, Lm):
    h = th.zeros(x.shape[0], x.shape[1], H, dtype=x.dtype, device=x.device)
    for t in range(T):
        h = quant(TR.RecMM.apply(h, Wh) + x[:, :, t:t + 1] * Wi[:, None, :], act, s, Lm)
    return TR.RecMM.apply(h, Wo)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--log2n", type=float, default=30); ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--data-seed", type=int, default=352)
    ap.add_argument("--steps", type=int, default=0, help="override number of steps (dry runs)")
    ap.add_argument("--warmup", type=int, default=2 ** 14)
    a = ap.parse_args(); dev = a.device
    spec = [dict(zip(("name", "seed", "act", "grid"), (f[0], int(f[1]), int(f[2]), int(f[3])))) for f in (s.split(":") for s in a.spec)]
    K = len(spec); B = a.batch; n_steps = a.steps or int(2 ** a.log2n) // B; warm = a.warmup
    inits = [TR.zoo_init(H, T, sp["seed"]) for sp in spec]
    P = [th.stack([i[j] for i in inits]).float().to(dev).requires_grad_() for j in range(3)]   # Wi, Wh, Wo
    act = th.tensor([sp["act"] for sp in spec], device=dev); grid = th.tensor([sp["grid"] for sp in spec], device=dev)
    Lm = th.tensor([LEVELS[sp["act"]] for sp in spec], device=dev)
    g = th.Generator(device=dev); g.manual_seed(a.data_seed)
    with th.no_grad():                                         # LSQ init from the first batch (float forward, init weights)
        x0 = th.randn(K, B, T, generator=g, device=dev); h = th.zeros(K, B, H, device=dev); us = []
        for t in range(T):
            u = th.bmm(h, P[1].transpose(1, 2)) + x0[:, :, t:t + 1] * P[0][:, None, :]; us.append(u); h = th.relu(u)
        mu = th.stack([th.relu(u).mean((1, 2)) for u in us]).mean(0)
        s0 = th.where(Lm > 0, 2 * mu / th.sqrt(th.clamp(Lm, min=1.0)), th.ones_like(mu))
    s = s0.clone().requires_grad_(); params = P + [s]
    m_ = [th.zeros_like(p) for p in params]; v_ = [th.zeros_like(p) for p in params]
    lr0, wd, b1, b2, eps = 0.04 / math.sqrt(H), 1e-3, 0.9, 0.999, 1e-8
    nst = th.full((K,), float(n_steps), device=dev); wu = th.full((K,), float(warm), device=dev)
    t0 = time.time(); log = []; lacc = th.zeros(2, K, device=dev); nlog = 0
    for step in range(1, n_steps + 1):
        x = discretise(th.randn(K, B, T, generator=g, device=dev), grid); y = labels(x)
        logits = forward(P[0], P[1], P[2], x, act, s, Lm)
        loss_k = th.nn.functional.cross_entropy(logits.reshape(K * B, T), y.reshape(-1), reduction="none").reshape(K, B).mean(1)
        for p in params:
            p.grad = None
        loss_k.sum().backward()
        with th.no_grad():
            lr = lr0 * TR.lr_mult(th.full((K,), float(step), device=dev), nst, wu)
            for i, p in enumerate(params):
                gr = p.grad; shp = (K,) + (1,) * (p.dim() - 1); lrs = lr.view(shp)
                if i < 3:
                    p.mul_(1 - lrs * wd)
                m_[i].mul_(b1).add_(gr, alpha=1 - b1); v_[i].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                mh = m_[i] / (1 - b1 ** step); vh = v_[i] / (1 - b2 ** step)
                p.sub_(lrs * mh / (vh.sqrt() + eps))
            s.clamp_(min=1e-6)
            lacc[0] += loss_k; lacc[1] += (logits.argmax(-1) == y).float().mean(1); nlog += 1
        if step % (16384 if n_steps >= 65536 else 500) == 0 or step == n_steps:
            log.append(dict(step=step, sec=time.time() - t0, loss=(lacc[0] / nlog).tolist(), acc=(lacc[1] / nlog).tolist()))
            lacc.zero_(); nlog = 0
            print(a.tag, step, "%.0fs" % (time.time() - t0), ["%.3f" % v for v in log[-1]["acc"][:8]], flush=True)
            th.save(dict(meta=dict(spec=spec, args=vars(a)), step=step, Wi=P[0].detach().cpu(), Wh=P[1].detach().cpu(),
                         Wo=P[2].detach().cpu(), s=s.detach().cpu(), log=log), os.path.join(HERE, f"{a.tag}.pt"))
    print(a.tag, "DONE", "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
