import numpy as np
def base(x):
    p=x[:,:9]; o=np.argsort(-p,1); s=np.sort(p,1)[:,::-1]
    D=dict(x=x,P8=s[:,0],Q8=s[:,1],R8=s[:,2],am=o[:,0],a2=o[:,1],P7=x[:,:8].max(1),
           Q7=np.sort(x[:,:8],1)[:,-2])
    return D
def feat(D,name):
    x=D['x']
    if name in D: return D[name]
    if name.startswith('x') and name[1:].isdigit(): return x[:,int(name[1:])]
    if name.startswith('d'):   # dK = xK - R8
        return x[:,int(name[1:])]-D['R8']
    if name.startswith('q'):   # qK = xK - Q8
        return x[:,int(name[1:])]-D['Q8']
    if name.startswith('AM'): S=[int(c) for c in name[2:]]; return np.isin(D['am'],S)*1.
    if name.startswith('A2'): S=[int(c) for c in name[2:]]; return np.isin(D['a2'],S)*1.
    if name.startswith('T2'): S=[int(c) for c in name[2:]]; return (np.isin(D['am'],S)|np.isin(D['a2'],S))*1.
    if name.startswith('ABS'): S=[int(c) for c in name[3:]]; return np.abs(x[:,S]).sum(1)
    if name=='G26': return np.maximum(D['P7']-np.maximum(D['Q7'],x[:,8]),0)
    raise KeyError(name)
