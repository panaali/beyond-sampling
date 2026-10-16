"""What sets the band edge? For each line, compare gates just inside / just outside the edge t = delta.
If they differ, the edge is a gate breakpoint (which step/neuron); else a logit crossing inside a linear piece.
Also: number of gate breakpoints strictly between the tie and the edge."""
import sys
import numpy as np
import wtool as W
nm = sys.argv[1] if len(sys.argv) > 1 else "orig"
M = W.get_model(nm)
R = dict(np.load(f"data/{nm}.npz")); k = np.nonzero((R["v"] > 0) & (np.abs(R["delta"]) > 1e-6) & (np.abs(R["delta"]) < R["tau"] - 1e-6))[0][:20000]
y, a, b, dl = R["y"][k], R["a"][k], R["b"][k], R["delta"][k]
n = len(y); r = np.arange(n)
d = np.zeros_like(y); d[r, a] = 1 / W.SQ2; d[r, b] = -1 / W.SQ2
sg = np.sign(dl)
g_in = W.jac_at_tie(M, y + (dl - sg * 1e-7)[:, None] * d, a, b, +1, eps=0)["G"]
g_out = W.jac_at_tie(M, y + (dl + sg * 1e-7)[:, None] * d, a, b, +1, eps=0)["G"]
g_tie = W.jac_at_tie(M, y, a, b, +1, eps=1e-9 * 1)["G"]
diff = g_in != g_out
isgate = diff.reshape(n, -1).any(1)
print(nm, "lines", n, "edge is a gate breakpoint: %.3f" % isgate.mean())
nflip = (g_tie != g_in).reshape(n, -1).sum(1)
print("gate bits changed between tie and edge: dist", np.round(np.bincount(np.minimum(nflip, 6)) / n, 3))
_, t_, i_ = np.nonzero(diff)
idx = np.nonzero(diff)[0]
rel = t_ - b[idx]
print("edge-gate step relative to b:", {int(o): round(float(np.mean(rel == o)), 3) for o in np.unique(rel)})
cnt = np.bincount(i_, minlength=16) / max(1, len(i_))
print("edge-gate neuron share:", np.round(cnt, 2))
# among edges that are logit crossings: which logit takes over? (a<->b or a 3rd candidate)
L_out = W.forward(M, y + (dl + sg * 1e-6)[:, None] * d).argmax(1)
L_in = W.forward(M, y + (dl - sg * 1e-6)[:, None] * d).argmax(1)
print("inside-band model output: a %.3f b %.3f p1 %.3f other %.3f" % ((L_in == a).mean(), (L_in == b).mean(), (L_in == R['p1'][k]).mean(), 1 - ((L_in == a) | (L_in == b) | (L_in == R['p1'][k])).mean()))
