"""Exact closed two-variable feasibility; never discard a zero-area set."""
from fractions import Fraction as F
from itertools import product

def dedup(poly):
    out=[]
    for p in poly:
        if not out or p!=out[-1]:out.append(p)
    if len(out)>1 and out[0]==out[-1]:out.pop()
    return out

def clip(poly,row):
    if not poly:return []
    value=lambda p:row[0]*p[0]+row[1]*p[1]+row[2]
    result=[];prev=poly[-1];fp=value(prev)
    for current in poly:
        fc=value(current)
        if (fp>=0)!=(fc>=0):
            t=fp/(fp-fc);result.append(tuple(a+t*(b-a) for a,b in zip(prev,current)))
        if fc>=0:result.append(current)
        prev=current;fp=fc
    return dedup(result)

def rectangle(a,b,c,d):
    if not a<b or not c<d:raise ValueError('strict compact rectangle needed')
    return [(a,c),(b,c),(b,d),(a,d)]

def area(poly):return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1])))/2 if poly else F(0)

def word_constraints(wi,wh,axis,first,middle,u):
    rows=[]
    for i,on in enumerate(first):
        sign=1 if on else -1;rows.append(tuple(sign*v for v in (wh[i][axis],wi[i],F(0))))
    for i,on in enumerate(middle):
        hs=sum(wh[i][j]*int(first[j])*wh[j][axis] for j in range(2));xs=sum(wh[i][j]*int(first[j])*wi[j] for j in range(2));sign=1 if on else -1
        rows.append(tuple(sign*v for v in (hs,xs,wi[i]*u)))
    return rows

def feasible_words(wi,wh,axis,slo,shi,u=F(1),limit=F(6)):
    out=[]
    for first,middle in product(product((False,True),repeat=2),repeat=2):
        rows=word_constraints(wi,wh,axis,first,middle,u);poly=rectangle(slo,shi,-limit,limit)
        for row in rows:poly=clip(poly,row)
        out.append(dict(first=first,middle=middle,constraints=rows,vertices=poly,feasible=bool(poly),area=area(poly)))
    return out

def serial(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x
