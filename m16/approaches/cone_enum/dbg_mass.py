import sys; sys.path.insert(0, '/home/user/clean-neurons/m16'); sys.path.insert(0, '.')
import numpy as np
from cones import *
from common import Wi, Wh, Wo
model = (Wi, Wh, Wo); C = Counter()
rng = np.random.default_rng(5)
X = rng.standard_normal((4_000_000, 10))
x = rng.standard_normal(10)
key, A = cone_of(model, x, C); c, rho = cheb(A, C)
print('rho', rho, 'min A c', (A@c).min(), 'x in own cone', (A@x>=0).all(), (A@x).min())
ins = np.concatenate([X[k:k+200000][(X[k:k+200000] @ A.T >= 0).all(1)] for k in range(0,len(X),200000)])
print(len(ins))
s = ins@c; Y = ins - s[:,None]*c[None]
print('s', s.mean(), s.min(), '|y|/s', np.median(np.linalg.norm(Y,axis=1)/s))
ac=A@c; L = (-(Y@A.T)/ac).max(1); print('s>=L', (s>=L-1e-9).mean())
# proper preintegration MC with plain normal y
Z = rng.standard_normal((200000,10)); Yz = Z - (Z@c)[:,None]*c
Lz = (-(Yz@A.T)/ac).max(1); print('plain MC preint', norm.cdf(-Lz).mean())
