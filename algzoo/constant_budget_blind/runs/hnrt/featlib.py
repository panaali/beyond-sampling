import numpy as np
def feature(x, name, cache=None):
    """Compute a named interpretable feature of x (N,10)."""
    N = len(x)
    o = np.argsort(-x, 1); s = np.take_along_axis(x, o, 1)          # overall ranks
    o9 = np.argsort(-x[:, :9], 1); a = np.take_along_axis(x[:, :9], o9, 1)  # ranks within x0..x8
    p = o9[:, 0]; g8 = (p != 8).astype(float)
    def topk(t, k):
        return -np.sort(-x[:, :t + 1], 1)[:, k]
    if name.startswith('max{') or name.startswith('sec{'):
        S = [int(v) for v in name[4:].strip('{}').split(',')]
        srt = np.sort(x[:, S], 1)
        return srt[:, -1] if name.startswith('max') else srt[:, -2]
    if '@' in name:
        base, sel = name.split('@')
        S = [int(v) for v in sel.strip('{}').split(',')]
        gate = base.endswith('g')
        if gate: base = base[:-1]
        k = int(base[1:]) - 1
        if base[0] == 's':
            return s[:, k] * np.isin(o[:, k], S)
        if base[0] == 'a':
            return a[:, k] * np.isin(o9[:, k], S) * (g8 if gate else 1.0)
    if name in ('s1', 's2', 's3'): return s[:, int(name[1]) - 1].copy()
    if name in ('a1', 'a2', 'a3'): return a[:, int(name[1]) - 1].copy()
    if name in ('a1g', 'a2g', 'x9g'):
        v = {'a1g': a[:, 0], 'a2g': a[:, 1], 'x9g': x[:, 9]}[name]; return v * g8
    if name[0] == 'x': return x[:, int(name[1:])].copy()
    if name == 'up9': return np.maximum(x[:, 9] - a[:, 0], 0)
    if name[0] == 'P': return topk(int(name[1:]), 0)
    if name[0] == 'R': return topk(int(name[1:]), 1)
    if name[0] == 'T': return topk(int(name[1:]), 2)
    if name[0] == 'Q': return x[:, int(name[1:]):].max(1)
    if name[0] == 'M': return x[:, int(name[1:])::3].max(1)
    if name.startswith('new'):
        j = int(name[3:]); return np.maximum(x[:, j] - x[:, :j].max(1), 0)
    raise ValueError(name)
def design(x, names):
    return np.stack([feature(x, n) for n in names], 1)
