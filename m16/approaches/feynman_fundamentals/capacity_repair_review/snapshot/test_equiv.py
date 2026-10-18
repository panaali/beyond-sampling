"""T234 check: batched Trainer == alg_zoo training loop (same init, same data, same AdamW/LambdaLR) step by step."""
import functools, sys
import torch as th
import trainer as TR
from alg_zoo.architectures import DistRNN
from alg_zoo.training import lr_mult_schedule

H, T, seeds, steps, B = 16, 10, [0, 3], 300, 64
n_train = steps * B * 4            # schedule mid-way through the decay
warm = 50
data = [th.randn(len(seeds), B, T, generator=th.Generator().manual_seed(1000 + i)) for i in range(steps)]
it = iter(data)
TR.sample = lambda K, B_, T_, gen, dev, *a: next(it)
tr = TR.Trainer(H, T, seeds, n_train, warmup=warm, batch=B, graph=False)
for _ in range(steps):
    tr.step()
worst = 0.0
for k, s in enumerate(seeds):
    th.manual_seed(s); m = DistRNN(hidden_size=H, seq_len=T)
    opt = th.optim.AdamW(m.parameters(), lr=0.04 / H ** 0.5, weight_decay=1e-3)
    sch = th.optim.lr_scheduler.LambdaLR(opt, functools.partial(lr_mult_schedule, n_steps=n_train // B,
                                                                warmup_steps=warm, decay_min_mult=0.01))
    for i in range(steps):
        x = data[i][k]; y = x.argsort(-1)[..., -2]
        loss = th.nn.functional.cross_entropy(m(x), y); loss.backward(); opt.step(); sch.step(); opt.zero_grad()
    ref = [m.rnn.weight_ih_l0.detach()[:, 0], m.rnn.weight_hh_l0.detach(), m.linear.weight.detach()]
    d = max(float((a - b[k]).abs().max()) for a, b in zip(ref, tr.params))
    scale = max(float(a.abs().max()) for a in ref)
    print("seed", s, "max |param diff| after", steps, "steps: %.2e (param scale %.2f)" % (d, scale))
    worst = max(worst, d)
sys.exit(0 if worst < 1e-4 else 1)
