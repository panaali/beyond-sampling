import numpy as np
def basics(x):
    x = np.asarray(x, dtype=float); N = x.shape[0]
    M = np.maximum.accumulate(x, 1)
    G = np.zeros((N, 10)); g = np.zeros((N, 10)); u = np.zeros((N, 10)); T = np.zeros((N, 10))
    T[:, 0] = x[:, 0]
    for t in range(1, 10):
        G[:, t] = np.maximum(M[:, t-1] - x[:, t], 0)
        s = -np.sort(-x[:, :t+1], 1); T[:, t] = s[:, :2].mean(1)
        g[:, t] = np.maximum(T[:, t-1] - x[:, t], 0)
        u[:, t] = np.maximum(x[:, t] - T[:, t-1], 0)
    srt = -np.sort(-x, 1); s9 = -np.sort(-x[:, :9], 1)
    return dict(x=x, M=M, G=G, g=g, u=u, T=T, top2sum=srt[:, :2].sum(1), top=srt[:, 0], sec=srt[:, 1], top4_08=s9[:, :4].sum(1))
def design(i, b):
    x, G, g, u, T = b['x'], b['G'], b['g'], b['u'], b['T']
    if i == 0: return [G[:, 3:6].sum(1), T[:, 8], g[:, 8], g[:, 1:6].sum(1), b['top4_08']]
    if i == 1: return [g[:, 4:7].sum(1), x[:, 4:6].sum(1), g[:, 5], G[:, 2:7].sum(1), u[:, 7]]
    if i == 2: return [G[:, 3:10].sum(1), x[:, 7:9].sum(1), g[:, 1:8].sum(1), G[:, 4], g[:, 3:6].sum(1)]
    if i == 3: return [G[:, 3:6].sum(1), g[:, 7], x[:, 4], g[:, 5], x[:, 1:5].sum(1)]
    if i == 4: return [g[:, 8], g[:, 9]]
    if i == 5: return [T[:, 8], x[:, 9]]
    if i == 6: return [b['top'] - x[:, 4], b['M'][:, 4], G[:, 2], b['sec'] - x[:, 4], x[:, 8]]
    if i == 7: return [G[:, 4:7].sum(1), x[:, 9], g[:, 5], T[:, 8], T[:, 2]]
