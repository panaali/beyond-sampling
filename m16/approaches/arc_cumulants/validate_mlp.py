"""Validation of the wrapper (diagnosis only; MC used for ground truth):
1. on a wide random He-initialised ReLU MLP our `propagate` reproduces ARC's own `mlp_kprop` and beats MC as in the paper;
2. identity pass-through layers keep Gaussian cumulants exact."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import arc_kprop as A
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, SIMPLE

torch.manual_seed(0)
n, L = int(sys.argv[1]) if len(sys.argv) > 1 else 64, 3            # 3 linear layers = 2 hidden layers
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=L).double()
Ws = [w.weight.detach().numpy() for w in mlp.Ws]
layers = [(W, None, [A.RELU] * n) for W in Ws[:-1]] + [(Ws[-1], None, None)]
rng = np.random.default_rng(0); N = 4_000_000; acc = np.zeros(n); acc2 = np.zeros(n)
for _ in range(N // 500_000):
    x = rng.standard_normal((500_000, n)); y = A.forward_layers(layers, x); acc += y.sum(0); acc2 += (y ** 2).sum(0)
mc = acc / N; var1 = acc2 / N - mc ** 2
print(f"n={n}, 2 hidden layers; MC truth with N={N}; variance-normalised MSE of sampling with S samples = 1/S")
for K in (1, 2, 3):
    ours = A.propagate(layers, {1: np.zeros(n), 2: np.eye(n)}, K)
    theirs = mlp_kprop(mlp, {1: torch.zeros(n, dtype=torch.float64), 2: torch.eye(n, dtype=torch.float64)}, k_max=K,
                       kind=SIMPLE, use_avg_metric=False)[1].to_tensor().numpy()
    vmse = np.mean((ours - mc) ** 2 / var1)
    print(f"K={K}: max|ours-mlp_kprop|={np.abs(ours-theirs).max():.2e}  variance-normalised MSE={vmse:.2e} (~ sampling with {1/vmse:.0f} samples; MC-truth floor {1/N:.1e})")
# identity pass-through: Gaussian with arbitrary covariance through 3 identity layers with random mixing
D = 8; C = rng.standard_normal((D, D)); S = C @ C.T / D
lay = [(rng.standard_normal((D, D)) / math.sqrt(D), None, [A.ID] * D) for _ in range(3)] + [(np.eye(D), None, None)]
mu = rng.standard_normal(D)
for K in (2, 3, 4):
    est = A.propagate(lay, {1: mu, 2: S}, K)
    M = np.eye(D)
    for W, _, _ in lay[:-1]:
        M = W @ M
    print(f"identity layers K={K}: max error of propagated mean {np.abs(est - M @ mu).max():.1e}")
