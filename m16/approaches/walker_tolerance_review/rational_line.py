"""Independent exact-real line decomposition for small bias-free ReLU RNNs."""
from collections import defaultdict
from fractions import Fraction as F
import time
import mpmath as mp


def rational(x):
    return x if isinstance(x,F) else F.from_float(float(x))


def wrong_mass(model,y,d,lo,hi,negative_target,positive_target,max_pieces=10000,deadline=float('inf')):
    start=time.process_time(); counts=defaultdict(int)
    Wi,Wh,Wo=model
    Wi=list(map(rational,Wi)); Wh=[list(map(rational,row)) for row in Wh]
    Wo=[list(map(rational,row)) for row in Wo]
    y=list(map(rational,y)); d=list(map(rational,d)); lo,hi=rational(lo),rational(hi)
    H=len(Wi); cells=[(lo,hi,[(F(0),F(0)) for _ in Wi])]
    def guard():
        if time.process_time()>deadline:raise TimeoutError('exact line reference deadline')
        if len(cells)>max_pieces:raise RuntimeError('exact line cap: no partial result returned')
    for t in range(len(y)):
        guard(); nxt=[]
        for left,right,h in cells:
            z=[(sum((Wh[j][k]*h[k][0] for k in range(H)),F(0))+Wi[j]*y[t],
                sum((Wh[j][k]*h[k][1] for k in range(H)),F(0))+Wi[j]*d[t]) for j in range(H)]
            counts['affine_scalar_products']+=2*H*H+2*H
            cuts={left,right}
            for c,s in z:
                if s:
                    counts['gate_roots']+=1
                    root=-c/s
                    if left<root<right:cuts.add(root)
            cuts=sorted(cuts)
            for a,b in zip(cuts[:-1],cuts[1:]):
                mid=(a+b)/2
                hh=[(c,s) if c+s*mid>0 else (F(0),F(0)) for c,s in z]
                nxt.append((a,b,hh))
        cells=nxt; guard()
        counts['gate_cells_processed']+=len(cells)
    intervals=[]
    for left,right,h in cells:
        logits=[(sum((row[k]*h[k][0] for k in range(H)),F(0)),
                 sum((row[k]*h[k][1] for k in range(H)),F(0))) for row in Wo]
        counts['affine_scalar_products']+=2*len(Wo)*H
        cuts={left,right}
        if left<0<right:cuts.add(F(0))
        for i,(c,s) in enumerate(logits):
            for c2,s2 in logits[:i]:
                if s!=s2:
                    root=-(c-c2)/(s-s2); counts['readout_roots']+=1
                    if left<root<right:cuts.add(root)
        cuts=sorted(cuts)
        for a,b in zip(cuts[:-1],cuts[1:]):
            mid=(a+b)/2
            values=[c+s*mid for c,s in logits]
            winner=max(range(len(values)),key=lambda j:(values[j],-j))
            target=positive_target if mid>0 else negative_target
            intervals.append((a,b,winner,target))
    with mp.workdps(70):
        def phi(x):
            z=mp.mpf(x.numerator)/x.denominator
            return (1+mp.erf(z/mp.sqrt(2)))/2
        mass=sum((phi(b)-phi(a) for a,b,w,k in intervals if w!=k),mp.mpf(0))
        high_precision=str(mass); value=float(mass)
    counts['mass_intervals']=sum(w!=k for a,b,w,k in intervals)
    counts['final_intervals']=len(intervals)
    return dict(mass=value,mass_70digits=high_precision,counts=dict(counts),
                intervals=[dict(lo=str(a),hi=str(b),winner=w,target=k) for a,b,w,k in intervals],
                cpu_seconds=time.process_time()-start)
