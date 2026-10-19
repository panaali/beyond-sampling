import numpy as np, sys, json
from scipy.optimize import least_squares
relu = lambda z: np.maximum(z, 0)

def feats(x):
    F = {}
    p = x[:, :9]
    s = -np.sort(-p, 1)
    F['M8'], F['S8'], F['T8'] = s[:, 0], s[:, 1], s[:, 2]
    F['am8'] = np.argmax(p, 1)
    sa = -np.sort(-x, 1); o = np.argsort(-x, 1)
    F['M'], F['S'], F['T'] = sa[:, 0], sa[:, 1], sa[:, 2]
    F['am'], F['a2'] = o[:, 0], o[:, 1]
    F['x'] = x
    F['M7'] = x[:, :8].max(1)
    F['mean7'] = x[:, :8].mean(1)
    return F

T = {}
def tF(p, F, sg=1):
    b, c0, A, w, ph = p
    wt = c0 + A * np.cos(w * F['am8'] + ph)
    return b * relu(sg * (F['S8'] + wt * (F['M8'] - F['S8']) - F['x'][:, 9]))
T['F+'] = (tF, [12, 0.7, 0.4, 0.7, 0.0])
T['F-'] = (lambda p, F: tF(p, F, -1), [10, 0.9, 0.1, 0.7, 0.0])
def tF2(p, F):
    b, c0, A, w, ph = p
    wt = c0 + A * np.cos(w * F['am8'] + ph)
    return b * (F['S8'] + wt * (F['M8'] - F['S8']))
T['F2'] = (tF2, [18, 0.8, 0.2, 0.7, 0.0])
def tG(p, F, M='M8', S='S8', am='am8'):
    A, c, w, g, e = p
    gt = A * (1 - np.cos(w * (F[am] - c)))
    return relu(gt * (F[M] - F[S]) + g * F[S] + e)
T['G8'] = (tG, [10, 3, 0.73, 1, 0])
T['Gall'] = (lambda p, F: tG(p, F, 'M', 'S', 'am'), [10, 3, 0.73, 1, 0])

def fitone(y, F, tmpl, inits):
    fn, p0 = T[tmpl]
    best = None
    for q in inits:
        r = least_squares(lambda p: fn(p, F) - y, q, max_nfev=3000)
        if best is None or r.cost < best.cost: best = r
    return best.x


def addfeats(F):
    x = F['x']; o = np.argsort(-x, 1); sa = -np.sort(-x, 1)
    F['p'] = o[:, :3]; F['v'] = sa[:, :3]
    s7 = -np.sort(-x[:, :8], 1); F['S7'] = s7[:, 1]
def mkK(P):
    w = 2 * np.pi / P
    def fn(p, F):
        if 'p' not in F: addfeats(F)
        A, B, c, al, e = p
        z = A * (1 - np.cos(w * (F['p'][:, 0] - c))) * F['v'][:, 0]
        for r in (1, 2):
            z = z + F['v'][:, r] * (al - B * (1 + np.cos(w * (F['p'][:, r] - c))))
        return relu(z + e)
    return fn
T['K8'] = (mkK(8), [8, 3, 3, -2, 0])
T['K9'] = (mkK(9), [8, 3, 3, -2, 0])
T['K86'] = (mkK(8.6), [8, 3, 3, -2, 0])
def t5(p, F):
    if 'p' not in F: addfeats(F)
    a, b, c, d, e = p
    return relu(a * F['x'][:, 8] + b * F['M7'] + c * F['S7'] + d * F['mean7'] + e)
T['n5'] = (t5, [12, -2, -2, -4, -3])
def t9(p, F):
    if 'p' not in F: addfeats(F)
    a, b, c, d, e = p
    return relu(a * F['x'][:, 8] + b * F['M7'] + c * F['S7'] + d * F['x'][:, 9] + e)
T['n9'] = (t9, [2, -1.5, -0.3, -0.2, -0.3])
def mkV2(P):
    w = 2 * np.pi / P
    def fn(p, F):
        if 'p' not in F: addfeats(F)
        A, B, c, al, e = p
        z = A * (1 - np.cos(w * (F['p'][:, 0] - c))) * F['v'][:, 0]
        z = z + al * (F['v'][:, 1] + F['v'][:, 2]) - B * (1 + np.cos(w * (F['p'][:, 1] - c))) * F['v'][:, 1]
        return relu(z + e)
    return fn
T['V2'] = (mkV2(8.6), [8, 3, 3, -2, 0])
def V3(p, F):
    if 'p' not in F: addfeats(F)
    A, B, c, al, w = p
    z = A * (1 - np.cos(w * (F['p'][:, 0] - c))) * F['v'][:, 0]
    for r in (1, 2):
        z = z + F['v'][:, r] * (al - B * (1 + np.cos(w * (F['p'][:, r] - c))))
    return relu(z)
T['V3'] = (V3, [8, 3, 3, -2, 0.73])
SETS = {}
def Sset(p, F):
    if 'p' not in F: addfeats(F)
    a1, b1, a2, b2, e = p
    sg = SETS['cur'][F['p']]
    z = F['v'][:, 0] * (a1 + b1 * sg[:, 0]) + (F['v'][:, 1:] * (a2 + b2 * sg[:, 1:])).sum(1)
    return relu(z + e)
T['S'] = (Sset, [5, 1, 0, 2, 0])
def W(p, F):
    if 'p' not in F: addfeats(F)
    A, c, g, d, e = p
    w = 2 * np.pi / 8.6
    z = A * (1 - np.cos(w * (F['p'][:, 0] - c))) * F['v'][:, 0] + g * (F['v'][:, 1] + F['v'][:, 2]) + d * F['x'][:, SETS['j']]
    return relu(z + e)
T['W'] = (W, [8, 3, -2, -2, 0])
def mkQ(M='M8', S='S8', Tt='T8', am='am8'):
    def fn(p, F):
        A, c, w, b, g = p
        return relu(A * (1 - np.cos(w * (F[am] - c))) * (F[M] - F[S]) + b * (F[S] - F[Tt]) + g * F[Tt])
    return fn
T['Q8'] = (mkQ(), [10, 3, 0.73, 6, 3])
T['Qg'] = (mkQ('M', 'S', 'T', 'am'), [10, 3, 0.73, 6, 3])
