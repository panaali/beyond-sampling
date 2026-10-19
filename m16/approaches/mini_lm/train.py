"""Train one baseline (T433). Usage: python train.py <lang> <arch> <seed> [--steps 8000] [--eval]
Writes models/<lang>_<arch>_s<seed>.pt (state dict + cfg) and .json (training log + metrics)."""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from langs import LANGS, model_input
from models import build, n_params
import metrics

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang"); ap.add_argument("arch"); ap.add_argument("seed", type=int)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_n", type=int, default=1_000_000)
    ap.add_argument("--no_exact", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "models"))
    ap.add_argument("--eval_only", action="store_true", help="score an existing checkpoint")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(1000 + a.seed)
    lang = LANGS[a.lang]()
    model = build(a.arch, lang)
    emb = [p for n, p in model.named_parameters() if "tok" in n or "pos.weight" in n or n.startswith("U.")]
    rest = [p for n, p in model.named_parameters() if not any(p is e for e in emb)]
    opt = torch.optim.AdamW([{"params": rest, "weight_decay": 0.01}, {"params": emb, "weight_decay": 0.0}], lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    log, t0 = [], time.time()
    name = f"{a.lang}_{a.arch}_s{a.seed}"
    if a.eval_only:
        ck = torch.load(os.path.join(a.out, name + ".pt"), weights_only=True)
        model.load_state_dict(ck["state"])
        log = ck.get("loss_log", "not saved (eval_only run)")
    for step in range(0 if a.eval_only else a.steps):
        model.train()
        y = lang.sample(a.batch, rng)
        sc = torch.from_numpy(lang.annotate(y)["scored"]) if lang.name == "modsum" else None
        x, yt = torch.from_numpy(model_input(y, lang.BOS)), torch.from_numpy(y)
        lg = model(x)
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), yt.reshape(-1), reduction="none").view(yt.shape)
        loss = ce.mean() if sc is None else (ce * sc).sum() / sc.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == a.steps - 1:
            log.append((step, loss.item()))
    os.makedirs(a.out, exist_ok=True)
    if not a.eval_only:
        torch.save({"cfg": model.cfg, "lang": a.lang, "state": model.state_dict(), "loss_log": log,
                    "train_seconds": time.time() - t0}, os.path.join(a.out, name + ".pt"))
    res = dict(name=name, params=n_params(model), train_seconds=(ck.get("train_seconds") if a.eval_only else time.time() - t0), loss_log=log,
               steps=a.steps, batch=a.batch, lr=a.lr, flops_per_fpe_large=model.flops(lang.L))
    big = dict(n=lang.N) if a.lang == "modsum" else {}
    res["large"] = metrics.sampled(model, lang, a.eval_n, seed=7_000_000 + a.seed, **big)
    res["large_horizon"] = "n=20" if a.lang == "modsum" else "L=32"
    if not a.no_exact:
        t1 = time.time()
        res["small_exact"] = metrics.exact_small(model, lang)
        res["small_exact_seconds"] = time.time() - t1
        res["small_horizon"] = "n<=10 (n uniform)" if a.lang == "modsum" else "L=15"
        small_kw = dict(L=lang.T_small) if a.lang == "dyck2" else {}
        if a.lang == "modsum":  # sample the small law directly: n uniform on 1..10
            r = np.random.default_rng(8_000_000 + a.seed)
            y = np.concatenate([lang.sample(100_000, r, n=n) for n in range(1, lang.n_small + 1)])
            res["small_sampled_1e6"] = metrics.score(model, lang, y, np.full(len(y), 1.0 / len(y)))
        else:
            res["small_sampled_1e6"] = metrics.sampled(model, lang, 1_000_000, seed=8_000_000 + a.seed, **small_kw)
    json.dump(res, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
    print(name, json.dumps({k: res[k] for k in ("params", "train_seconds")}),
          {k: round(v, 6) for k, v in res["large"].items() if not k.endswith("_se")})


if __name__ == "__main__":
    main()
