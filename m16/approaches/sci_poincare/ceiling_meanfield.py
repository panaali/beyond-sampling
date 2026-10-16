"""Ceiling test for the reduction 'full rank permutation (symbolic) + top-k values (continuous)':
replace each input of rank > k by the conditional expected order statistic given the k-th value
(computed model-free by numerical quadrature), keep the order. The reduced system's accuracy (estimated here
by sampling, diagnostic only) is what a perfect deductive integration of this reduction would return."""
import sys; sys.path.insert(0, "/home/user/clean-neurons/m16")
import numpy as np
from arc_eval import forward, Wi0, Wh0, Wo0
from scipy.special import ndtr, ndtri
from scipy.stats import beta
rng = np.random.default_rng(6); M = (Wi0, Wh0, Wo0)
N = 1_000_000
def expected_low(c, n):
    """E of order stats of n iid N(0,1) conditioned < c (descending), via median-ish: quantile at beta means."""
    # j-th largest of n uniforms on (0, F(c)) has mean (n+1-j)/(n+1) * F(c); map through ndtri (quantile approx)
    j = np.arange(1, n + 1); return ndtri(ndtr(c)[:, None] * (n + 1 - j)[None] / (n + 1))
for k in (2, 3, 4, 5, 6, 7):
    acc_t = acc_r = 0
    for ch in range(N // 200_000):
        x = rng.standard_normal((200_000, 10)); o = np.argsort(-x, 1); tgt = o[:, 1]
        ck = np.take_along_axis(x, o[:, k-1:k], 1)[:, 0]
        xr = x.copy(); np.put_along_axis(xr, o[:, k:], expected_low(ck, 10 - k), 1)
        acc_t += np.sum(forward(M, x).argmax(1) == tgt); acc_r += np.sum(forward(M, xr).argmax(1) == tgt)
    print(f"k={k}: true {acc_t/N:.4f} reduced {acc_r/N:.4f}  bias {(acc_r-acc_t)/N*100:+.2f} pp")
