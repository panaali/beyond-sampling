import numpy as np
def _r(z): return np.maximum(z, 0.0)
def feature(name, x):
    """Interpretable features of x (N,10). s0=max, s1=second largest, am=argmax position,
    M_k=max(x0..x_k), S_k=second largest of x0..x_k, g=s0-s1."""
    x = np.asarray(x, float)
    s = -np.sort(-x, 1); s0, s1 = s[:, 0], s[:, 1]; am = np.argmax(x, 1); g = s0 - s1
    M = np.maximum.accumulate(x, 1)
    def S(k): return -np.sort(-x[:, :k+1], 1)[:, 1]
    def argM(k): return np.argmax(x[:, :k+1], 1)
    PO = ((am % 2 == 1) & (am < 8)).astype(float); PE = ((am % 2 == 0) & (am < 8)).astype(float)
    po = (am % 2 == 1).astype(float); pe = 1 - po
    n = name
    if n == 's0': return s0
    if n.startswith('max(x'): a=int(n[5]); b=int(n[8]); return np.maximum(x[:, a], x[:, b])
    if n.startswith('min(x'): a=int(n[5]); b=int(n[8]); return np.minimum(x[:, a], x[:, b])
    if n.startswith('cap'): t=int(n[3:]); return np.minimum(x[:, t], M[:, t-1])
    if n.startswith('r(x') and n[4]=='-' and n[5]=='x': return _r(x[:, int(n[3])] - x[:, int(n[6])])
    if n == 'g': return g
    if n.startswith('M9-x'): return s0 - x[:, int(n[4:])]
    if n.startswith('Mpo'): k = int(n[3:]); return (argM(k) % 2 == 1) * M[:, k]
    if n.startswith('Mpe'): k = int(n[3:]); return (argM(k) % 2 == 0) * M[:, k]
    if n == 'E8M8': a = argM(8); return ((a % 2 == 0) & (a < 8)) * M[:, 8]
    if n == 'PO*g': return PO * g
    if n == 'PE*g': return PE * g
    if n == 'pe*g': return pe * g
    if n == 'po*g': return po * g
    if n == 'pe*s0': return pe * s0
    if n == 'pe*s1': return pe * s1
    if n == 'PE*s0': return PE * s0
    if n == 'PO*s0': return PO * s0
    if n.startswith('po*(M9-x'): return po * (s0 - x[:, int(n[8])])
    if n.startswith('pe*x'): return pe * x[:, int(n[4:])]
    if n.startswith('am6*x'): return (am == 6) * x[:, int(n[5:])]
    if n == 'am6*s1': return (am == 6) * s1
    if n.startswith('r(x') and n.endswith('-m2)'): return _r(x[:, int(n[3])] - s1)
    if n.startswith('r(m2-x'): return _r(s1 - x[:, int(n[6])])
    if n.startswith('r(-x'): return _r(-x[:, int(n[4])])
    if n.startswith('r(x') and '-M' in n:  # r(x_t - M_k)
        t = int(n[3]); k = int(n[n.index('-M')+2:-1]); return _r(x[:, t] - M[:, k])
    if n.startswith('r(x') and '-S' in n:
        t = int(n[3]); k = int(n[n.index('-S')+2:-1]); return _r(x[:, t] - S(k))
    if n.startswith('r(x'): return _r(x[:, int(n[3])])
    if n.startswith('xn'): t = int(n[2:]); return x[:, t] * (am != t)
    if n.startswith('M') and '-x' in n: k = int(n[1:n.index('-')]); t = int(n[n.index('-x')+2:]); return M[:, k] - x[:, t]
    if n.startswith('M') and '-S' in n: k = int(n[1:n.index('-')]); return M[:, k] - S(k)
    if n.startswith('M'): return M[:, int(n[1:])]
    if n.startswith('S'): return S(int(n[1:]))
    if n.startswith('x'): return x[:, int(n[1:])]
    raise KeyError(n)
