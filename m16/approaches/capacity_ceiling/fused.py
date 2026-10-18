"""T234 fused training kernel (Triton): one program per stacked model, N training steps per launch.

Per step and model: batch of B = 1024 sequences in tiles of BB; x ~ N(0,1) from in-kernel Philox keyed by
(data_seed, model, step, sample, t) (or loaded from memory in test mode); label = 2nd argmax by comparisons; forward
h_t = relu(Wh h_{t-1} + Wi x_t) for t < 10 with exact fp32 dots (input_precision="ieee"); cross-entropy over the 10
logits; BPTT with the batch gradient reduced inside the program; AdamW (torch formula) with the zoo LR schedule
(linear warm-up, exponential decay to the floor at mid-run) computed in-kernel. Readout Wo is padded to 16 rows.
State per model: Wi (16), Wh (16x16), Wo (16x16, rows >= 10 zero), Adam moments; the step counter is the host's."""
import math

import torch as th
import triton
import triton.language as tl

H = 16; TP = 16; T = tl.constexpr(10)


@triton.jit
def _train(Wi, Wh, Wo, mWi, mWh, mWo, vWi, vWh, vWo, lr0, nsteps, warm, bsz, fbl, fti, gap, lossacc, scratch, X,
           start, n_iter, data_seed, wd, b1, b2, eps, floor, B: tl.constexpr, BB: tl.constexpr,
           USE_X: tl.constexpr, DUMP: tl.constexpr):
    mdl = tl.program_id(0)
    r = tl.arange(0, 16)
    rb = tl.arange(0, BB)
    pw = mdl * 256 + r[:, None] * 16 + r[None, :]          # [i, j] row-major 16x16 of this model
    pwT = mdl * 256 + r[None, :] * 16 + r[:, None]         # transposed view [j, i]
    pi = mdl * 16 + r
    sbase = mdl * (T * BB * 16)
    lr_0 = tl.load(lr0 + mdl); ns = tl.load(nsteps + mdl).to(tl.float32); wu = tl.load(warm + mdl).to(tl.float32)
    Bm = tl.load(bsz + mdl)                                   # this model's batch (multiple of BB)
    nbl = (tl.load(fbl + mdl) * Bm.to(tl.float32) + 0.5).to(tl.int32)      # hard-case counts per batch
    nti = (tl.load(fti + mdl) * Bm.to(tl.float32) + 0.5).to(tl.int32)
    gp = tl.load(gap + mdl)
    for it in range(n_iter):
        step = start + it
        wi = tl.load(Wi + pi); wh = tl.load(Wh + pw); whT = tl.load(Wh + pwT); wo = tl.load(Wo + pw); woT = tl.load(Wo + pwT)
        gwi = tl.zeros([16], tl.float32); gwh = tl.zeros([16, 16], tl.float32); gwo = tl.zeros([16, 16], tl.float32)
        lsum = 0.0; csum = 0.0
        for tile in range(Bm // BB):
            samp = tile * BB + rb
            if USE_X:
                x = tl.load(X + ((mdl * B + samp[:, None]) * 16 + r[None, :]), mask=r[None, :] < T, other=float("-inf"))
            else:
                key = data_seed * 1048576 + step                              # Philox key = (data_seed, step)
                off = (mdl * B + samp[:, None]) * 16 + r[None, :]              # B = max batch; < 2^30 for <= 4096 models
                x = tl.randn(key, off)
                # blind region (samples < nbl): |z| with at most one positive coordinate (which = 10: none)
                which = (tl.randint(key ^ 0x2545F491, mdl * B + samp) % 11).to(tl.int32)
                sg = tl.where(r[None, :] == which[:, None], 1.0, -1.0)
                isb = samp[:, None] < nbl
                x = tl.where(isb, tl.abs(x) * sg, x)
                x = tl.where(r[None, :] < T, x, float("-inf"))
                # near-tie (nbl <= samples < nbl + nti): 3rd largest moved to U(0, gap) below the 2nd largest
                ist = (samp >= nbl) & (samp < nbl + nti)
                j1 = tl.argmax(x, axis=1); y = tl.where(r[None, :] == j1[:, None], float("-inf"), x)
                j2 = tl.argmax(y, axis=1); v2 = tl.max(y, axis=1); y = tl.where(r[None, :] == j2[:, None], float("-inf"), y)
                j3 = tl.argmax(y, axis=1)
                u = tl.rand(key ^ 0x68E31DA4, mdl * B + samp)
                x = tl.where(ist[:, None] & (r[None, :] == j3[:, None]), (v2 - u * gp)[:, None], x)
            i1 = tl.argmax(x, axis=1)
            x2 = tl.where(r[None, :] == i1[:, None], float("-inf"), x)
            lab = tl.argmax(x2, axis=1)
            xz = tl.where(r[None, :] < T, x, 0.0)
            if DUMP:                                                   # test mode: write data + labels, no training
                tl.store(X + ((mdl * B + samp[:, None]) * 16 + r[None, :]), tl.where(r[None, :] == 15, lab.to(tl.float32)[:, None], xz))
            h = tl.zeros([BB, 16], tl.float32)
            for t in tl.static_range(T):
                xt = tl.sum(tl.where(r[None, :] == t, xz, 0.0), axis=1)
                pre = tl.dot(h, whT, input_precision="ieee") + xt[:, None] * wi[None, :]
                h = tl.maximum(pre, 0.0)
                tl.store(scratch + sbase + t * BB * 16 + rb[:, None] * 16 + r[None, :], h)
            tl.debug_barrier()
            lg = tl.dot(h, woT, input_precision="ieee")
            lg = tl.where(r[None, :] < T, lg, float("-inf"))
            mx = tl.max(lg, axis=1)
            e = tl.exp(lg - mx[:, None]); se = tl.sum(e, axis=1)
            p = e / se[:, None]
            onehot = (r[None, :] == lab[:, None]).to(tl.float32)
            lsum += tl.sum(tl.log(se) + mx - tl.sum(tl.where(onehot > 0, lg, 0.0), axis=1), axis=0)
            csum += tl.sum((tl.argmax(lg, axis=1) == lab).to(tl.float32), axis=0)
            dlg = tl.where(r[None, :] < T, (p - onehot) / Bm.to(tl.float32), 0.0)
            gwo += tl.dot(tl.trans(dlg), h, input_precision="ieee")
            dh = tl.dot(dlg, wo, input_precision="ieee")
            for tt in tl.static_range(T):
                t = T - 1 - tt
                ht = tl.load(scratch + sbase + t * BB * 16 + rb[:, None] * 16 + r[None, :])
                if t > 0:
                    hp = tl.load(scratch + sbase + (t - 1) * BB * 16 + rb[:, None] * 16 + r[None, :])
                else:
                    hp = tl.zeros([BB, 16], tl.float32)
                dpre = tl.where(ht > 0, dh, 0.0)
                gwh += tl.dot(tl.trans(dpre), hp, input_precision="ieee")
                xt = tl.sum(tl.where(r[None, :] == t, xz, 0.0), axis=1)
                gwi += tl.sum(dpre * xt[:, None], axis=0)
                dh = tl.dot(dpre, wh, input_precision="ieee")
            tl.debug_barrier()
        # zoo LR schedule (LambdaLR: update u uses lr_mult(u)) and AdamW (torch formula), inactive after n_steps
        sf = step.to(tl.float32)
        mult = tl.minimum(sf / wu, 1.0) * tl.maximum(tl.exp(tl.log(floor) * 2.0 * sf / ns), floor)
        act = (sf < ns).to(tl.float32)
        lr = lr_0 * mult * act
        tcount = sf + 1.0
        bc1 = 1.0 - tl.exp(tcount * tl.log(b1)); bc2 = 1.0 - tl.exp(tcount * tl.log(b2))
        # Wi
        m = tl.load(mWi + pi); v = tl.load(vWi + pi)
        m = m + act * (1.0 - b1) * (gwi - m); v = v + act * ((b2 * v + (1.0 - b2) * gwi * gwi) - v)
        wi = wi * (1.0 - lr * wd) - (lr / bc1) * m / (tl.sqrt(v) / tl.sqrt(bc2) + eps)
        tl.store(mWi + pi, m); tl.store(vWi + pi, v); tl.store(Wi + pi, wi)
        m = tl.load(mWh + pw); v = tl.load(vWh + pw)
        m = m + act * (1.0 - b1) * (gwh - m); v = v + act * ((b2 * v + (1.0 - b2) * gwh * gwh) - v)
        wh = wh * (1.0 - lr * wd) - (lr / bc1) * m / (tl.sqrt(v) / tl.sqrt(bc2) + eps)
        tl.store(mWh + pw, m); tl.store(vWh + pw, v); tl.store(Wh + pw, wh)
        m = tl.load(mWo + pw); v = tl.load(vWo + pw)
        m = m + act * (1.0 - b1) * (gwo - m); v = v + act * ((b2 * v + (1.0 - b2) * gwo * gwo) - v)
        wo = wo * (1.0 - lr * wd) - (lr / bc1) * m / (tl.sqrt(v) / tl.sqrt(bc2) + eps)
        tl.store(mWo + pw, m); tl.store(vWo + pw, v); tl.store(Wo + pw, wo)
        tl.store(lossacc + mdl * 2, tl.load(lossacc + mdl * 2) + lsum / Bm.to(tl.float32))
        tl.store(lossacc + mdl * 2 + 1, tl.load(lossacc + mdl * 2 + 1) + csum / Bm.to(tl.float32))
        tl.debug_barrier()


class FusedTrainer:
    """stacked models with zoo inits; step(n) runs n training steps in one launch"""
    def __init__(self, seeds, n_train, lr_scale=1.0, warmup=2 ** 14, batch=1024, data_seed=0, lr_base=0.04, wd=1e-3,
                 floor=0.01, device="cuda", BB=128, f_blind=0.0, f_tie=0.0, gap=0.05):
        import trainer as TR
        M = len(seeds); self.M = M; self.BB = BB; self.device = device
        vec = lambda v, dt: (th.as_tensor(v, dtype=dt) if hasattr(v, "__len__") else th.full((M,), v, dtype=dt)).contiguous().to(device)
        self.bsz = vec(batch, th.int32); self.B = int(self.bsz.max())
        self.fbl = vec(f_blind, th.float32); self.fti = vec(f_tie, th.float32); self.gap = vec(gap, th.float32)
        inits = [TR.zoo_init(H, T, s) for s in seeds]
        self.Wi = th.stack([i[0] for i in inits]).float().contiguous().to(device)
        self.Wh = th.stack([i[1] for i in inits]).float().contiguous().to(device)
        Wo = th.zeros(M, 16, 16); Wo[:, :T] = th.stack([i[2] for i in inits]); self.Wo = Wo.contiguous().to(device)
        self.m = [th.zeros_like(x) for x in (self.Wi, self.Wh, self.Wo)]; self.v = [th.zeros_like(x) for x in (self.Wi, self.Wh, self.Wo)]
        lrs = th.as_tensor(lr_scale, dtype=th.float32).expand(M) if not hasattr(lr_scale, "__len__") else th.as_tensor(lr_scale, dtype=th.float32)
        self.lr0 = (lr_base / H ** 0.5 * lrs).contiguous().to(device)
        nt = th.as_tensor(n_train, dtype=th.int64).expand(M) if not hasattr(n_train, "__len__") else th.as_tensor(n_train, dtype=th.int64)
        self.nsteps = (nt.to(device) // self.bsz.to(th.int64)).to(th.int32).contiguous()
        wu = th.as_tensor(warmup, dtype=th.int32).expand(M) if not hasattr(warmup, "__len__") else th.as_tensor(warmup, dtype=th.int32)
        self.warm = wu.contiguous().to(device)
        self.lossacc = th.zeros(M, 2, device=device)
        self.scratch = th.empty(M * T * BB * 16, device=device)
        self.step_n = 0; self.data_seed = data_seed; self.wd = wd; self.floor = floor

    def step(self, n=1, X=None, dump=None):
        _train[(self.M,)](self.Wi, self.Wh, self.Wo, *self.m, *self.v, self.lr0, self.nsteps, self.warm, self.bsz,
                          self.fbl, self.fti, self.gap, self.lossacc, self.scratch,
                          X if X is not None else (dump if dump is not None else self.Wi), self.step_n, n,
                          self.data_seed, self.wd, 0.9, 0.999, 1e-8, self.floor, B=self.B, BB=self.BB,
                          USE_X=X is not None, DUMP=dump is not None, num_warps=4)
        if dump is None:
            self.step_n += n

    def weights(self):
        return [self.Wi.double().cpu().numpy(), self.Wh.double().cpu().numpy(), self.Wo[:, :T].double().cpu().numpy()]
