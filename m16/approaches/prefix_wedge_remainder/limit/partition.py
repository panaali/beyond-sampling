"""Exact piecewise-affine limit partition and outward Gaussian CDFs."""
from fractions import Fraction as F
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from rational_bounds import I,pi_interval,atan_point,atan_interval,sqrt_point

def representatives(lo,hi):
    if lo is None and hi is None:return [F(-1),F(1)]
    if lo is None:return [hi-1,hi-2]
    if hi is None:return [lo+1,lo+2]
    assert lo<hi
    return [(2*lo+hi)/3,(lo+2*hi)/3]

def inside(x,lo,hi):return (lo is None or lo<x) and (hi is None or x<hi)

def values(a,b,w,r):
    h=[max(F(0),ai+bi*r) for ai,bi in zip(a,b)]
    scores=[sum(wi*hi for wi,hi in zip(row,h)) for row in w]
    winner=max(range(len(scores)),key=lambda i:scores[i])
    return h,scores,winner

def partition(a,b,w):
    a=list(map(F,a));b=list(map(F,b));w=[list(map(F,row)) for row in w]
    assert len(a)==len(b)==2 and len(w)==4 and all(len(r)==2 for r in w)
    gate_roots=sorted(set(-ai/bi for ai,bi in zip(a,b) if bi))
    endpoints=set(gate_roots);intervals=[]
    for lo,hi in zip([None]+gate_roots,gate_roots+[None]):
        test=representatives(lo,hi)[0]
        mask=[ai+bi*test>0 for ai,bi in zip(a,b)]
        outputs=[(sum(row[i]*a[i]*mask[i] for i in range(2)),
                  sum(row[i]*b[i]*mask[i] for i in range(2))) for row in w]
        roots=set()
        for i in range(4):
            for j in range(i):
                da,db=outputs[i][0]-outputs[j][0],outputs[i][1]-outputs[j][1]
                if db:
                    root=-da/db
                    if inside(root,lo,hi):roots.add(root)
        roots=sorted(roots);endpoints.update(roots)
        for l,r in zip([lo]+roots,roots+[hi]):
            samples=representatives(l,r);sample_data=[]
            for x in samples:
                h,scores,winner=values(a,b,w,x)
                assert [ai+bi*x>0 for ai,bi in zip(a,b)]==mask
                assert scores==[u+v*x for u,v in outputs]
                sample_data.append({'r':x,'hidden':h,'scores':scores,'winner':winner})
            assert sample_data[0]['winner']==sample_data[1]['winner']
            for i in range(4):
                for j in range(i):
                    da,db=outputs[i][0]-outputs[j][0],outputs[i][1]-outputs[j][1]
                    assert not db or not inside(-da/db,l,r)
            intervals.append({'lower':l,'upper':r,'mask':mask,'output_affines':outputs,
                              'winner':sample_data[0]['winner'],'representatives':sample_data})
    boundaries=[]
    for r in sorted(endpoints):
        h,scores,winner=values(a,b,w,r)
        boundaries.append({'r':r,'hidden':h,'scores':scores,'winner':winner,
                           'maximizers':[i for i,x in enumerate(scores) if x==scores[winner]]})
    return {'A':a,'B':b,'Wo':w,'gate_roots':gate_roots,'intervals':intervals,'boundaries':boundaries}

def dyadic(interval,bits=160):
    scale=1<<bits
    lo=(interval.lo.numerator*scale)//interval.lo.denominator
    hi=-((-interval.hi.numerator*scale)//interval.hi.denominator)
    out=I(F(lo,scale),F(hi,scale))
    assert out.lo<=interval.lo<=interval.hi<=out.hi
    assert interval.lo-out.lo<F(1,scale) and out.hi-interval.hi<F(1,scale)
    return out

def cdf(side,c):
    assert side in ['10','01']
    if c=='-inf':return I.point(0)
    if c=='+inf':return I.point(F(1,8) if side=='10' else F(3,8))
    c=F(c);pi=pi_interval()
    second=atan_interval(I.point(c)/sqrt_point(2+c*c))
    value=I.point(F(1,8))+(atan_point(c)+(-pi/4-second if side=='10' else pi/4+second))/(4*pi)
    maxmass=F(1,8) if side=='10' else F(3,8)
    value=I(max(F(0),value.lo),min(maxmass,value.hi))
    return dyadic(value)

def mass(parts,label=2):
    total=I.point(0);contributions=[];cache={}
    def f(side,x):
        key=(side,x)
        if key not in cache:cache[key]=cdf(side,x)
        return cache[key]
    for idx,row in enumerate(parts['intervals']):
        if row['winner']!=label:continue
        for side in ['10','01']:
            lo,hi=row['lower'],row['upper']
            if side=='10':hi=F(1) if hi is None else min(hi,F(1))
            else:lo=F(1) if lo is None else max(lo,F(1))
            if lo is not None and hi is not None and lo>=hi:continue
            lower=f(side,'-inf' if lo is None else lo)
            upper=f(side,'+inf' if hi is None else hi)
            raw=(upper-lower)/2
            contribution=I(max(F(0),raw.lo),min(F(1,24),raw.hi))
            total=total+contribution
            contributions.append({'interval_index':idx,'rank_side':side,'lower':lo,'upper':hi,
                                  'cdf_lower':lower,'cdf_upper':upper,'joint_mass':contribution})
    total=I(max(F(0),total.lo),min(F(1,12),total.hi))
    return {'joint_correctness_mass':total,'contributions':contributions,
            'distinct_cdf_calls':len(cache),'label':label,'sector_mass':F(1,12)}
