"""Floating LP proposes; exact rational Gordan witnesses decide."""
import time
START=time.process_time()
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
CALLS=0;MAXCALLS=240
class BudgetStop(Exception):pass
def lp(*args,**kwargs):
    global CALLS
    if CALLS>=MAXCALLS or time.process_time()-START>19:raise BudgetStop('frozen LP/CPU cap')
    CALLS+=1
    return linprog(*args,**kwargs,method='highs',options={'threads':1,'maxiter':10000,'time_limit':2.})
def exactsolve(M,b):
    n=len(M[0]);a=[list(map(F,row))+[F(v)] for row,v in zip(M,b)];r=0;pivots=[]
    for j in range(n):
        k=next((i for i in range(r,len(a)) if a[i][j]),None)
        if k is None:continue
        a[r],a[k]=a[k],a[r];v=a[r][j];a[r]=[x/v for x in a[r]]
        for i in range(len(a)):
            if i!=r:
                v=a[i][j];a[i]=[x-v*y for x,y in zip(a[i],a[r])]
        pivots.append(j);r+=1
    if any(not any(row[:-1]) and row[-1] for row in a) or len(pivots)!=n:return None
    result=[F(0)]*n
    for i,j in enumerate(pivots):result[j]=a[i][-1]
    return result
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def feasible(rows):
    before=CALLS;rows=[tuple(map(int,row)) for row in rows];d=len(rows[0]) if rows else 4
    if not rows:return dict(status='feasible',witness=['1']*d,lp_calls=0)
    for i,row in enumerate(rows):
        if not any(row):return dict(status='infeasible',dual={str(i):'1'},lp_calls=0,reason='strict zero inequality')
    scales=[max(abs(x) for x in row) for row in rows];exact=[[F(v,s) for v in row] for row,s in zip(rows,scales)];A=np.array(exact,float)
    primal=lp(np.zeros(d),A_ub=-A,b_ub=-np.ones(len(rows)),bounds=[(None,None)]*d)
    if primal.success:
        witness=[F(float(x)) for x in primal.x]
        if all(dot(row,witness)>0 for row in rows):return dict(status='feasible',witness=list(map(str,witness)),lp_calls=CALLS-before)
    dual=lp(np.zeros(len(rows)),A_eq=np.vstack((A.T,np.ones(len(rows)))),b_eq=np.r_[np.zeros(d),1.],bounds=(0,None))
    if dual.success:
        support=[i for i,v in enumerate(dual.x) if v>0]
        if 0<len(support)<=d+1:
            values=exactsolve([[exact[i][j] for i in support] for j in range(d)]+[[F(1)]*len(support)],[0]*d+[1])
            if values is not None and all(v>=0 for v in values):
                original=[v/scales[i] for i,v in zip(support,values)];normalizer=sum(original);original=[v/normalizer for v in original]
                if all(sum(v*rows[i][j] for i,v in zip(support,original))==0 for j in range(d)):
                    return dict(status='infeasible',dual={str(i):str(v) for i,v in zip(support,original)},lp_calls=CALLS-before)
    return dict(status='unknown',primal_status=int(primal.status),dual_status=int(dual.status),lp_calls=CALLS-before)
def verify(rows,result):
    if result['status']=='feasible':return all(dot(row,list(map(F,result['witness'])))>0 for row in rows)
    if result['status']=='infeasible':
        values={int(i):F(v) for i,v in result['dual'].items()}
        return sum(values.values())==1 and all(v>=0 for v in values.values()) and all(sum(v*rows[i][j] for i,v in values.items())==0 for j in range(len(rows[0])))
    return False
