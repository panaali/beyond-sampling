import numpy as np
def _R(z): return np.maximum(z,0)
def base(x):
    y=x[:,:9]; o=np.argsort(-y,1,kind='stable'); s=np.take_along_axis(y,o,1)
    return dict(x=x,A=s[:,0],B=s[:,1],C=s[:,2],pA=o[:,0],pB=o[:,1],pC=o[:,2],m=np.maximum.accumulate(x,1))
def feat(b,spec):
    x=b['x']
    if spec in ('A','B','C'): return b[spec]
    if spec in ('Ap','Bp','Cp'): return b[spec[0]]*b['p'+spec[0]]/8.0
    if spec in ('Aq','Bq','Cq'): return b[spec[0]]*(8-b['p'+spec[0]])/8.0
    if spec=='x9': return x[:,9]
    if spec=='sum9': return x[:,:9].sum(1)
    if spec=='sum10': return x.sum(1)
    if '@' in spec:
        k,S=spec.split('@'); S=[int(q) for q in S.split(',')]
        return b[k]*np.isin(b['p'+k],S)
    if spec.startswith('m2_'):
        t=int(spec[3:]); return np.sort(x[:,:t+1],1)[:,-2]
    if spec.startswith('xjB+'): j=int(spec[4:]); return _R(x[:,j]-b['B'])
    if spec.startswith('Bxj+'): j=int(spec[4:]); return _R(b['B']-x[:,j])
    if spec.startswith('Cxj+'): j=int(spec[4:]); return _R(b['C']-x[:,j])
    if spec.startswith('gap'): t=int(spec[3:]); return _R(b['m'][:,t-1]-x[:,t])
    if spec.startswith('m'): return b['m'][:,int(spec[1:])]
    if spec.startswith('x'): return x[:,int(spec[1:])]
    raise ValueError(spec)
def model(x,specs,coefs):
    b=base(x); return _R(sum(c*feat(b,s) for s,c in zip(specs,coefs)))
