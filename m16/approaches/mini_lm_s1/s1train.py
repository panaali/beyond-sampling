"""Train one S1 variant (T444). Same recipe as the harness train.py (AdamW 3e-3, cosine, batch 256, clip 1,
2 threads) plus: per-layer attention types, hard-attention training schedules, and a modsum curriculum.

python s1train.py <lang> <tag> <attn> <seed> [--sched st|anneal|mix] [--beta0 1 --beta1 30] [--hard_frac .25]
                  [--tol 1e-6] [--steps 8000] [--curric 0.5] [--eval_n 200000] [--exact]
anneal: softmax(beta*s), beta geometric beta0 -> beta1 over the first (1-hard_frac) of steps, then straight-through hard.
mix:    lam 0 -> 1 linearly over the first (1-hard_frac), a = (1-lam) softmax(beta1 s) + lam ST-hard; then ST.
curric (modsum): n ~ U{1..n_max(step)}, n_max linear from 2 to N over the first `curric` fraction of steps.
Writes models/<lang>_<tag>_s<seed>.pt and .json."""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mini_lm"))
from langs import LANGS, model_input  # noqa: E402
import metrics  # noqa: E402
from s1models import S1Transformer  # noqa: E402
from models import n_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang"); ap.add_argument("tag"); ap.add_argument("attn"); ap.add_argument("seed", type=int)
    ap.add_argument("--sched", default="st")
    ap.add_argument("--beta0", type=float, default=1.0); ap.add_argument("--beta1", type=float, default=30.0)
    ap.add_argument("--hard_frac", type=float, default=0.25)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--curric", type=float, default=0.0)
    ap.add_argument("--eval_n", type=int, default=200_000)
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "models"))
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(1000 + a.seed)
    lang = LANGS[a.lang]()
    model = S1Transformer(lang.V_in, lang.V_out, lang.L, attn=a.attn, tol=a.tol)
    emb = [p for n, p in model.named_parameters() if "tok" in n or "pos.weight" in n]
    rest = [p for n, p in model.named_parameters() if not any(p is e for e in emb)]
    opt = torch.optim.AdamW([{"params": rest, "weight_decay": 0.01}, {"params": emb, "weight_decay": 0.0}], lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / a.steps)))
    log, t0 = [], time.time()
    name = f"{a.lang}_{a.tag}_s{a.seed}"
    soft_steps = int(a.steps * (1 - a.hard_frac))
    for step in range(a.steps):
        model.train()
        if a.sched == "anneal" and step < soft_steps:
            model.set_train_mode("anneal", beta=a.beta0 * (a.beta1 / a.beta0) ** (step / soft_steps))
        elif a.sched == "mix" and step < soft_steps:
            model.set_train_mode("mix", beta=a.beta1, lam=step / soft_steps)
        else:
            model.set_train_mode("st", beta=a.beta1 if a.sched != "st" else 1.0)
        for b in model.blocks:  # sparsemax layers keep beta = 1 (scale is learned)
            if b.attn.endswith("sparse") or b.attn == "sparsemax":
                b.beta = 1.0
        if a.lang == "modsum" and a.curric > 0:
            nmax = min(lang.N, 2 + int((lang.N - 2) * step / (a.curric * a.steps)))
            nn_ = rng.integers(1, nmax + 1, a.batch)
            y = lang._build(rng.integers(0, lang.m, (a.batch, lang.N)), nn_)
        else:
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
            log.append((step, round(loss.item(), 5)))
    model.eval()
    betas = [b.beta for b in model.blocks]
    os.makedirs(a.out, exist_ok=True)
    torch.save({"cfg": model.cfg, "lang": a.lang, "state": model.state_dict(), "betas": betas, "loss_log": log,
                "args": vars(a), "train_seconds": time.time() - t0}, os.path.join(a.out, name + ".pt"))
    res = dict(name=name, args=vars(a), params=n_params(model), train_seconds=time.time() - t0, loss_log=log,
               flops_per_fpe_large=model.flops(lang.L))
    big = dict(n=lang.N) if a.lang == "modsum" else {}
    t1 = time.time()
    res["large"] = metrics.sampled(model, lang, a.eval_n, seed=7_000_000 + a.seed, **big)
    res["large_horizon"] = "n=20" if a.lang == "modsum" else "L=32"
    if a.exact:
        res["small_exact"] = metrics.exact_small(model, lang)
    res["eval_seconds"] = time.time() - t1
    json.dump(res, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
    L = res["large"]
    print(name, f"train {res['train_seconds']:.0f}s acc {L['acc']:.5f} ill {L['illegal_mass']:.2e} "
          f"B0.5 {L['B0.5']:.2e} B0.9 {L['B0.9']:.2e} n {L['n_seq']}", flush=True)


if __name__ == "__main__":
    main()
