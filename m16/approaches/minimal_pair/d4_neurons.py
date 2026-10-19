"""D4: neurons in the negative orthant. With x<0 only neurons with Wi<0 can be driven by the input at step 0;
later steps mix via Wh. Report #Wi<0, mean #active neurons per step on all-negative inputs vs generic inputs,
and the readout margin (rank-2 logit minus best other) on errors vs correct inputs."""
import numpy as np
import core as C
for T in (4, 5):
  for s in range(5):
    Wi, Wh, Wo = m = C.zoo(16, T, s); rng = np.random.default_rng(3)
    xn = -np.abs(rng.standard_normal((200_000, T))); xg = rng.standard_normal((200_000, T))
    def acts(x):
        h = np.zeros((len(x), 16)); A = []
        for t in range(T):
            h = np.maximum(h @ Wh.T + x[:, t:t+1] * Wi[None], 0); A.append((h > 0).sum(1).mean())
        return A, h
    An, hn = acts(xn); Ag, hg = acts(xg)
    alive_n = ((hn > 0).mean(0) > 0.01).sum()
    L = hn @ Wo.T; pred = L.argmax(1); tgt = np.argsort(xn, 1)[:, -2]
    mar = L[np.arange(len(L)), tgt] - np.where(np.eye(T, dtype=bool)[tgt], -np.inf, L).max(1)
    rel = mar / np.abs(L).max(1)
    print(f"T{T} s{s}: #Wi<0 {int((Wi<0).sum()):2d}  active/step neg {np.round(An,1)} gen {np.round(Ag,1)}  "
          f"final-state neurons alive (neg) {alive_n}  |  P(err) {np.mean(pred!=tgt):.3f}  median rel margin {np.median(rel):.3f}")
