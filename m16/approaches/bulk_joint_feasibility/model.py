"""Exact gate and first-index readout semantics for strict open cones."""
from fractions import Fraction as F
from math import gcd

def primitive(row):
    g=0
    for v in row:g=gcd(g,abs(v))
    return tuple(v//g for v in row) if g else tuple(row)
def map_and_planes(history,wi,wh,wo,D):
    h=len(wi);T=len(wo);A=((0,)*T,)*h;scale=1;planes=[]
    for t in range(T):
        B=tuple(tuple(sum(wh[i][j]*A[j][q] for j in range(h))+(wi[i]*scale if q==t else 0) for q in range(T)) for i in range(h));gates=history[t*h:(t+1)*h]
        for i,row in enumerate(B):
            if any(row):planes.append(primitive(row if gates[i]=='1' else tuple(-x for x in row)))
            else:assert gates[i]=='0','identicallyzero gate mustbeOFF'
        A=tuple(row if gates[i]=='1' else (0,)*T for i,row in enumerate(B));scale*=D
    L=tuple(tuple(sum(row[i]*A[i][q] for i in range(h)) for q in range(T)) for row in wo)
    return list(dict.fromkeys(planes)),L

def rank_planes(top,second,T=4):
    pairs=[(top,second)]+[(second,j) for j in range(T) if j not in (top,second)]
    return [tuple(int(q==i)-int(q==j) for q in range(T)) for i,j in pairs]
def correct_planes(L,target):
    planes=[]
    for k in range(len(L)):
        if k==target:continue
        row=tuple(a-b for a,b in zip(L[target],L[k]))
        if not any(row):
            if k<target:return None
        else:planes.append(primitive(row))
    return planes

def wrong_planes(L,target,other):
    row=tuple(a-b for a,b in zip(L[other],L[target]))
    if not any(row):return [] if other<target else None
    return [primitive(row)]
def direct(wi,wh,wo,D,x):
    h=(F(0),)*len(wi);scale=1;history=''
    for z in x:
        v=tuple(sum(row[j]*h[j] for j in range(len(h)))+wi[i]*z*scale for i,row in enumerate(wh));history+=''.join('1' if a>0 else '0' for a in v);h=tuple(max(a,0) for a in v);scale*=D
    logits=[sum(a*b for a,b in zip(row,h)) for row in wo]
    return history,max(range(len(wo)),key=lambda j:logits[j])
