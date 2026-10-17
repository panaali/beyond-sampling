"""Exact projection of closed halfplanes onto their first coordinate."""
from fractions import Fraction as F

def project(rows):
    rows=[tuple(map(F,r)) for r in rows];positive=[];negative=[];reduced=[]
    for i,(a,b,c) in enumerate(rows):
        if b>0:positive.append(i)
        elif b<0:negative.append(i)
        else:reduced.append(dict(row=(a,c),sources=[(i,F(1))]))
    for i in positive:
        for j in negative:
            a,b,c=rows[i];A,B,C=rows[j]
            reduced.append(dict(row=(a/b-A/B,c/b-C/B),sources=[(i,1/b),(j,-1/B)]))
    low=None;high=None;bad=[]
    for i,r in enumerate(reduced):
        a,c=r['row']
        if a>0:low=max(low,-c/a) if low is not None else -c/a
        elif a<0:high=min(high,-c/a) if high is not None else -c/a
        elif c<0:bad.append(i)
    feasible=not bad and (low is None or high is None or low<=high)
    return dict(feasible=feasible,lower=low,upper=high,reduced=reduced,contradictions=bad)

def witness(rows,s):
    lower=[];upper=[]
    for a,b,c in rows:
        if b>0:lower.append(-(a*s+c)/b)
        elif b<0:upper.append(-(a*s+c)/b)
        elif a*s+c<0:raise ValueError('infeasible state')
    lo=max(lower) if lower else None;hi=min(upper) if upper else None
    if lo is not None and hi is not None:
        if lo>hi:raise ValueError('infeasible state')
        t=(lo+hi)/2
    else:t=lo if lo is not None else hi if hi is not None else F(0)
    assert all(a*s+b*t+c>=0 for a,b,c in rows)
    return t

def union(intervals):
    intervals=sorted(intervals,key=lambda p:p[0]);out=[]
    for lo,hi in intervals:
        assert lo is not None and lo>=0 and (hi is None or hi>=lo)
        if not out or (out[-1][1] is not None and lo>out[-1][1]):out.append([lo,hi])
        else:out[-1][1]=None if hi is None or out[-1][1] is None else max(hi,out[-1][1])
    return out

def complement(intervals):
    result=[];cursor=F(0)
    for lo,hi in union(intervals):
        if cursor<lo:result.append([cursor,lo])
        if hi is None:return result
        cursor=hi
    result.append([cursor,None]);return result
