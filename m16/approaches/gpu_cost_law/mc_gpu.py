"""T192/T193: independent truth by plain Monte Carlo on the GPU (float64 forward pass, torch Philox generator).
Shares no code with estT. usage: python mc_gpu.py H T name n_log10 [seed] -> results/mc/H{H}T{T}_{name}.json"""
import json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "understand_m43"), os.path.join(HERE, "..", "..")]
from estP import load_model

H, T, name, nl = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 12345
Wi, Wh, Wo = [torch.as_tensor(w, dtype=torch.float64, device="cuda") for w in load_model(H, T, name)]
g = torch.Generator(device="cuda"); g.manual_seed(seed)
n = int(10 ** nl); chunk = 1 << 24; ok = 0; done = 0; t0 = time.time()
while done < n:
    m = min(chunk, n - done)
    x = torch.randn(m, T, generator=g, device="cuda", dtype=torch.float64)
    h = torch.zeros(m, H, dtype=torch.float64, device="cuda")
    for t in range(T):
        h = torch.relu(h @ Wh.T + x[:, t:t + 1] * Wi[None])
    ok += int(((h @ Wo.T).argmax(1) == x.topk(2, 1).indices[:, 1]).sum()); done += m
p = ok / n
r = dict(H=H, T=T, model=name, n=n, correct=ok, acc=p, se=(p * (1 - p) / n) ** 0.5, seed=seed, sec=time.time() - t0)
os.makedirs(os.path.join(HERE, "results", "mc"), exist_ok=True)
json.dump(r, open(os.path.join(HERE, "results", "mc", f"H{H}T{T}_{name.replace(':', '_')}.json"), "w"), indent=1)
print(json.dumps(r))
