import numpy as np
def _r(a): return np.maximum(a,0)
def feat(name, x):
    """Evaluate a named feature. Prefix = x0..x8. Q1>=Q2>=Q3 its top values, am/as = positions of its largest/second."""
    p=x[:,:9]; Q=-np.sort(-p,1); s=np.argsort(-p,1)
    Q1,Q2,Q3=Q[:,0],Q[:,1],Q[:,2]; am,as_=s[:,0],s[:,1]
    t=name.split('|')
    k=t[0]
    if k=='Q1': return Q1
    if k=='Q2': return Q2
    if k=='Q3': return Q3
    if k=='x9': return x[:,9]
    if k=='x': return x[:,int(t[1])]
    if k=='rnew8': return _r(x[:,8]-x[:,:8].max(1))
    if k=='gap': return Q1-Q2
    if k=='rxQ3': j=int(t[1]); return _r(x[:,j]-Q3)
    if k=='rxQ2': j=int(t[1]); return _r(x[:,j]-Q2)
    if k in ('rx8S7','rS7x8','rP7x8'):
        f8=-np.sort(-x[:,:8],1); P7,S7=f8[:,0],f8[:,1]; x8=x[:,8]
        return {'rx8S7':_r(x8-S7),'rS7x8':_r(S7-x8),'rP7x8':_r(P7-x8)}[k]
    if k=='rx': return _r(x[:,int(t[1])])
    if k=='rQ2x': j=int(t[1]); return _r(Q2-x[:,j])
    if k=='rQ1x': j=int(t[1]); return _r(Q1-x[:,j])
    S=[int(c) for c in t[2]]
    if k=='am': ind=np.isin(am,S)
    elif k=='as': ind=np.isin(as_,S)
    elif k=='top2': ind=np.isin(am,S)|np.isin(as_,S)
    v=t[1]
    base={'Q1':Q1,'Q2':Q2,'Q3':Q3,'g12':Q1-Q2,'g23':Q2-Q3}[v]
    return ind*base
