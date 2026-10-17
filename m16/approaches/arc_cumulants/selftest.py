"""Self-test (diagnosis only, uses random inputs): the layer stacks compute exactly the leaderboard's `correct`,
and the cell decomposition sums to the accuracy."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "m16")
import numpy as np
import arc_kprop as A
import leaderboard as LB

for (H, T) in [(16, 4), (16, 5), (16, 10)]:
    name, m = LB._base(H, T)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200_000, T))
    ok = LB.correct(m, x)
    s0 = np.concatenate([np.zeros((len(x), H)), x], 1)
    f = A.forward_layers(A.rnn_layers(m) + A.readout_uncond(m), s0)[:, 0]
    o = np.argsort(x, 1)
    cellsum = 0
    for p1 in range(T):
        for p2 in range(T):
            if p1 != p2:
                sel = (o[:, -1] == p1) & (o[:, -2] == p2)
                g = A.forward_layers(A.rnn_layers(m) + A.readout_cell(m, p2), s0[sel])[:, 0]
                cellsum += g.sum()
    print(H, T, name, "acc", ok.mean(), "uncond-stack mismatch", int((f != ok).sum()), "cell-stack acc", cellsum / len(x))

# T42 audit finding 1: exact logit ties must follow np.argmax (smallest index). Minimal model: all logits equal.
import itertools
m = (np.array([1.0]), np.array([[1.0]]), np.ones((3, 1)))
x = np.array(list(itertools.permutations([1.0, 2.0, 3.0])))
ok = LB.correct(m, x)
s0 = np.concatenate([np.zeros((len(x), 1)), x], 1)
f = A.forward_layers(A.rnn_layers(m) + A.readout_uncond(m), s0)[:, 0]
o = np.argsort(x, 1)
g = np.array([A.forward_layers(A.rnn_layers(m) + A.readout_cell(m, o[n, -2]), s0[n:n + 1])[0, 0] for n in range(len(x))])
print("tie test: per-input mismatches uncond", int((f != ok).sum()), "cell", int((g != ok).sum()), "of", len(x))
