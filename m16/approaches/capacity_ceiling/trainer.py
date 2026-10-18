"""T234: batched, faithful re-implementation of alg_zoo.training.train for DistRNN on 2nd_argmax.

K independent models are stacked in one set of tensors (Wi (K,H), Wh (K,H,H), Wo (K,T,H)); each model has its own
init (copied from DistRNN after torch.manual_seed(seed), i.e. bitwise the zoo init), its own data (fresh N(0,1)
batches, or a hard-case mixture), its own n_train / lr schedule, and AdamW applied elementwise (torch.optim.AdamW
formula, per-model lr), so the models never interact. Defaults = zoo: lr 0.04/sqrt(H), wd 1e-3, betas (0.9,0.999),
eps 1e-8, batch 1024, warm-up 2^14 steps, exponential decay to 0.01x at mid-run then flat.
On CUDA the whole step (data, forward, backward, update) is captured in a CUDA graph.

Data mixtures (option c): per batch, a fraction f_blind of inputs from the blind region (<= 1 positive coordinate,
drawn as |z| with random signs conditioned on that event) and f_tie of near-tie inputs (2nd and 3rd largest within
gap g); the rest plain N(0, I). Loss = cross-entropy on the true 2nd-argmax label.
"""
import math

import torch as th


def zoo_init(H, T, seed):
    from alg_zoo.architectures import DistRNN
    th.manual_seed(seed)
    m = DistRNN(hidden_size=H, seq_len=T)
    return (m.rnn.weight_ih_l0.detach()[:, 0].clone(), m.rnn.weight_hh_l0.detach().clone(),
            m.linear.weight.detach().clone())


def lr_mult(step, n_steps, warmup, floor=0.01):
    """alg_zoo.training.lr_mult_schedule (vectorised over models: n_steps, warmup tensors)"""
    w = th.clamp(step / warmup, max=1.0)
    d = th.clamp(floor ** (2 * step / n_steps), min=floor)
    return w * d


def forward(Wi, Wh, Wo, x):
    """x (K, B, T) -> logits (K, B, T); h_t = relu(Wh h_{t-1} + Wi x_t), h_0 = 0 (nn.RNN, no bias)."""
    K, B, T = x.shape
    h = th.zeros(K, B, Wi.shape[1], dtype=x.dtype, device=x.device)
    for t in range(T):
        h = th.relu(th.bmm(h, Wh.transpose(1, 2)) + x[:, :, t:t + 1] * Wi[:, None, :])
    return th.bmm(h, Wo.transpose(1, 2))


def sample(K, B, T, gen, device, f_blind=0.0, f_tie=0.0, gap=0.05):
    x = th.randn(K, B, T, generator=gen, device=device)
    nb = int(round(f_blind * B)); nt = int(round(f_tie * B))
    if nb:
        # blind region: <= 1 positive coordinate. Draw |z|, choose which coordinate (if any) is positive:
        # P(none positive) : P(exactly one) = 1 : T (both events equally likely per configuration of signs)
        z = th.randn(K, nb, T, generator=gen, device=device).abs()
        which = th.randint(0, T + 1, (K, nb), generator=gen, device=device)      # T = none positive
        sgn = -th.ones(K, nb, T, device=device)
        sgn.scatter_(2, which.clamp(max=T - 1)[..., None], 1.0)
        sgn = th.where((which == T)[..., None], -th.ones_like(sgn), sgn)
        x[:, :nb] = z * sgn
    if nt:
        # near-tie: move the 3rd largest to within U(0, gap) below the 2nd largest
        y = th.randn(K, nt, T, generator=gen, device=device)
        v, idx = y.sort(-1, descending=True)
        u = th.rand(K, nt, generator=gen, device=device) * gap
        v[..., 2] = v[..., 1] - u
        y.scatter_(2, idx, v)
        x[:, nb:nb + nt] = y
    return x


class Trainer:
    def __init__(self, H, T, seeds, n_train, lr_base=0.04, wd=1e-3, warmup=2 ** 14, batch=1024, device="cpu",
                 f_blind=0.0, f_tie=0.0, gap=0.05, data_seed=0, lr_scale=None, floor=0.01, graph=None):
        K = len(seeds); self.K, self.H, self.T, self.B = K, H, T, batch; self.device = device
        inits = [zoo_init(H, T, s) for s in seeds]
        self.Wi = th.stack([i[0] for i in inits]).to(device).requires_grad_()
        self.Wh = th.stack([i[1] for i in inits]).to(device).requires_grad_()
        self.Wo = th.stack([i[2] for i in inits]).to(device).requires_grad_()
        self.params = [self.Wi, self.Wh, self.Wo]
        self.m = [th.zeros_like(p) for p in self.params]; self.v = [th.zeros_like(p) for p in self.params]
        nt = th.as_tensor(n_train, dtype=th.float64).reshape(-1).expand(K) if not hasattr(n_train, "__len__") \
            else th.as_tensor(n_train, dtype=th.float64)
        self.n_steps = (nt // batch).to(device)
        self.warm = th.as_tensor(warmup, dtype=th.float64).expand(K).to(device) if not hasattr(warmup, "__len__") \
            else th.as_tensor(warmup, dtype=th.float64).to(device)
        scale = th.ones(K, dtype=th.float64) if lr_scale is None else th.as_tensor(lr_scale, dtype=th.float64)
        self.lr0 = (lr_base / H ** 0.5 * scale).to(device); self.floor = floor
        self.wd = wd; self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.gen = th.Generator(device=device); self.gen.manual_seed(data_seed)
        self.mix = (f_blind, f_tie, gap)
        self.step_t = th.zeros((), dtype=th.float64, device=device)     # completed steps (shared by all models)
        self.loss = th.zeros(K, device=device); self.acc = th.zeros(K, device=device)
        self.graph = None
        if graph is None:
            graph = device != "cpu"
        if graph:
            self._capture()

    def _step_body(self):
        x = sample(self.K, self.B, self.T, self.gen, self.device, *self.mix)
        y = x.argsort(-1)[..., -2]
        logits = forward(self.Wi, self.Wh, self.Wo, x)
        lossk = th.nn.functional.cross_entropy(logits.reshape(-1, self.T), y.reshape(-1), reduction="none")
        lossk = lossk.reshape(self.K, self.B).mean(1)
        grads = th.autograd.grad(lossk.sum(), self.params)
        with th.no_grad():
            self.loss.copy_(lossk.detach())
            self.acc.copy_((logits.argmax(-1) == y).float().mean(1))
            # LambdaLR applies lr_lambda(step) where step = number of scheduler.step() calls so far
            # (step 0 for the first update): lr used at update u (0-based) = lr0 * lr_mult(u)
            mult = lr_mult(self.step_t, self.n_steps, self.warm, self.floor)
            active = (self.step_t < self.n_steps).to(th.float64)               # finished models stop updating
            lr = (self.lr0 * mult * active).to(self.Wi.dtype)
            t = self.step_t + 1
            bc1 = 1 - self.b1 ** t; bc2 = 1 - self.b2 ** t
            for p, g, m, v in zip(self.params, grads, self.m, self.v):
                shp = (-1,) + (1,) * (p.dim() - 1)
                lrk = lr.reshape(shp); act = active.to(p.dtype).reshape(shp)
                p.mul_(1 - lrk * self.wd)
                m.copy_(m + act * (1 - self.b1) * (g - m))
                v.copy_(v + act * ((self.b2 * v + (1 - self.b2) * g * g) - v))
                denom = (v.sqrt() / bc2.sqrt().to(p.dtype)).add_(self.eps)
                p.add_(-(lrk / bc1.to(p.dtype)) * m / denom)
            self.step_t.add_(1)

    def _capture(self):
        s = th.cuda.Stream(); s.wait_stream(th.cuda.current_stream())
        state = [p.detach().clone() for p in self.params] + [x.clone() for x in self.m + self.v]
        with th.cuda.stream(s):
            for _ in range(3):                      # warm-up (allocations) then restore the state
                self._step_body()
        th.cuda.current_stream().wait_stream(s)
        with th.no_grad():
            for dst, src in zip(self.params + self.m + self.v, state):
                dst.copy_(src)
            self.step_t.zero_()
        self.gen.manual_seed(self.gen.initial_seed())
        self.graph = th.cuda.CUDAGraph()
        if hasattr(self.graph, "register_generator_state"):          # custom generator must be registered (torch>=2.4)
            self.graph.register_generator_state(self.gen)
        else:                                                         # fallback: graph-safe default CUDA generator
            th.cuda.manual_seed(self.gen.initial_seed()); self.gen = None
        with th.cuda.graph(self.graph):
            self._step_body()

    def step(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self._step_body()

    def state(self):
        return {f"p{i}": t.detach().cpu() for i, t in enumerate(self.params)} | \
               {f"m{i}": t.cpu() for i, t in enumerate(self.m)} | {f"v{i}": t.cpu() for i, t in enumerate(self.v)} | \
               {"step": self.step_t.cpu()}

    def load_state(self, st, data_seed):
        """in-place restore (keeps the captured graph valid); data stream reseeded from (data_seed, step)"""
        with th.no_grad():
            for i, t in enumerate(self.params):
                t.copy_(st[f"p{i}"])
            for i, t in enumerate(self.m):
                t.copy_(st[f"m{i}"])
            for i, t in enumerate(self.v):
                t.copy_(st[f"v{i}"])
            self.step_t.copy_(st["step"])
        seed = data_seed * 1000003 + int(st["step"])
        if self.gen is not None:
            self.gen.manual_seed(seed)
        else:
            th.cuda.manual_seed(seed)

    def weights(self):
        return [p.detach().cpu().double().numpy() for p in self.params]
