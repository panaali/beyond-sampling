"""Independent exhaustive exact affine gate cells for small control networks."""
from fractions import Fraction as F
import itertools
import mpmath as mp

def frac(x):return F(float(x))
def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def interval(lo,hi,a,b,active):
    if b==0:return (lo,hi) if (a>0)==active else None
    root=-a/b;lower=(b>0)==active
    if lower:lo=root if lo is None else max(lo,root)
    else:hi=root if hi is None else min(hi,root)
    return None if lo is not None and hi is not None and lo>=hi else (lo,hi)

def exact_circle(model,u,v):
    wi,wh,wo=model;wi=list(map(frac,wi));wh=[list(map(frac,row)) for row in wh];wo=[list(map(frac,row)) for row in wo]
    u=list(map(frac,u));v=list(map(frac,v));h=len(wi);t=len(u);total=mp.mpf(0);cells=0;intervals=0
    def angle(x,sign):return sign*mp.pi/2 if x is None else mp.atan(mp.mpf(x.numerator)/x.denominator)
    with mp.workdps(70):
        for sign in (1,-1):
            ainput=[sign*x for x in u];binput=[sign*x for x in v]
            pieces=[(None,None,[F(0)]*h,[F(0)]*h)]
            for step in range(t):
                result=[]
                for lo,hi,a,b in pieces:
                    pre_a=[dot(row,a)+w*ainput[step] for row,w in zip(wh,wi)]
                    pre_b=[dot(row,b)+w*binput[step] for row,w in zip(wh,wi)]
                    for mask in itertools.product((False,True),repeat=h):
                        bounds=(lo,hi)
                        for aa,bb,on in zip(pre_a,pre_b,mask):
                            bounds=interval(*bounds,aa,bb,on)
                            if bounds is None:break
                        if bounds is not None:result.append((*bounds,[aa if on else F(0) for aa,on in zip(pre_a,mask)],[bb if on else F(0) for bb,on in zip(pre_b,mask)]))
                pieces=result
            for lo,hi,a,b in pieces:
                cells+=1;la=[dot(row,a) for row in wo];lb=[dot(row,b) for row in wo];roots=set()
                for aa,bb in ((ainput,binput),(la,lb)):
                    for i,j in itertools.combinations(range(t),2):
                        slope=bb[i]-bb[j]
                        if slope:
                            root=-(aa[i]-aa[j])/slope
                            if (lo is None or lo<root) and (hi is None or root<hi):roots.add(root)
                cuts=[lo]+sorted(roots)+[hi]
                for left,right in zip(cuts[:-1],cuts[1:]):
                    point=F(0) if left is None and right is None else right-1 if left is None else left+1 if right is None else (left+right)/2
                    labels=[aa+bb*point for aa,bb in zip(la,lb)];inputs=[aa+bb*point for aa,bb in zip(ainput,binput)]
                    winner=max(range(t),key=labels.__getitem__);target=sorted(range(t),key=lambda i:(inputs[i],i))[-2]
                    if winner==target:total+=angle(right,1)-angle(left,-1)
                    intervals+=1
        return float(total/(2*mp.pi)),dict(exact_gate_cells=cells,exact_intervals=intervals)
