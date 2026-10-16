"""Weight-derived threshold surrogate: per (p1,p2,p3) flip thresholds by vectorised bisection on canonical probes."""
import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, T
_ref = np.sort(np.random.default_rng(0).standard_normal((400_000, T)), 1)[:, ::-1]
EOS = _ref.mean(0)
TRIPLES = np.array([(a,b,c) for a in range(T) for b in range(T) for c in range(T) if len({a,b,c})==3])

def canon(seed=0):
    X = np.empty((len(TRIPLES), T))
    rs = np.random.default_rng(seed)
    for i,(a,b,c) in enumerate(TRIPLES):
        x = np.empty(T); x[a],x[b],x[c] = EOS[0],EOS[1],EOS[2]
        rest=[j for j in range(T) if j not in (a,b,c)]
        r=np.arange(3,T); rs.shuffle(r); x[rest]=EOS[r]; X[i]=x
    return X

def thresholds(model, iters=8, seed=0):
    """Returns tau23, tau12 (relative-gap thresholds) per triple, and evals used."""
    X = canon(seed); n = len(X); idx = np.arange(n)
    a,b,c = TRIPLES.T; v2 = X[idx,b]; evals = 0; out = []
    for kind in ("23","12"):
        def bad(g):
            Y = X.copy()
            if kind=="23": Y[idx,c] = v2*(1-g); wrong=c
            else: Y[idx,a] = v2/(1-g); wrong=a
            return forward(model,Y).argmax(1)==wrong
        lo=np.zeros(n); hi=np.full(n,0.95)
        for _ in range(iters):
            mid=(lo+hi)/2; bd=bad(mid); evals+=n
            lo=np.where(bd,mid,lo); hi=np.where(bd,hi,mid)
        out.append((lo+hi)/2)
    return out[0], out[1], evals
