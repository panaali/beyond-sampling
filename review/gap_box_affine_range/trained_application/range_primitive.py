"""Exact affine range on one labeled gap box. Core coordinates are physical x."""
from fractions import Fraction as F

def q(x):
    if type(x) not in (int,str,F):raise ValueError('exact integer/string/Fraction required')
    try:return F(x)
    except (ValueError,ZeroDivisionError) as e:raise ValueError('invalid rational') from e

def affine_range(box,coeff,constant,leader,second,gap):
    T=len(box)
    if T<3 or len(coeff)!=T or type(leader)is not int or type(second)is not int or not 0<=leader<T or not 0<=second<T or leader==second:raise ValueError('invalid dimensions or labels')
    if any(len(p)!=2 for p in box):raise ValueError('endpoint pairs required')
    b=[[q(v) for v in p] for p in box];a=list(map(q,coeff));c=q(constant);g=q(gap)
    if g<0 or any(l>u for l,u in b):raise ValueError('negative gap or reversed interval')
    rest=[r for r in range(T) if r not in (leader,second)]
    L=max(b[second][0],max(b[r][0]+g for r in rest));U=min(b[second][1],b[leader][1]-g)
    work=dict(input_rational_values=3*T+2,ordered_pairs=1,candidate_values=0,endpoint_completions=0,objective_products=0,objective_additions=0)
    premise='Selected exact-expression categories only; comparisons, sorting, Fraction internals, validation, storage and runtime are not fully priced.'
    if L>U:return dict(status='empty',second_interval=[L,U],candidates=[],range=None,witnesses=None,work=work,cost_scope=premise)
    K=sorted({L,U}|{v for v in [b[leader][0]-g]+[b[r][1]+g for r in rest] if L<=v<=U})
    rows=[];work['candidate_values']=len(K)
    for t in K:
        intervals=[]
        for r in range(T):
            if r==second:iv=[t,t]
            elif r==leader:iv=[max(b[r][0],t+g),b[r][1]]
            else:iv=[b[r][0],min(b[r][1],t-g)]
            assert iv[0]<=iv[1];intervals.append(iv)
        low=[iv[0] if z>=0 else iv[1] for z,iv in zip(a,intervals,strict=True)]
        high=[iv[1] if z>0 else iv[0] for z,iv in zip(a,intervals,strict=True)]
        values=[c+sum((z*x for z,x in zip(a,w,strict=True)),F(0)) for w in (low,high)]
        rows.append(dict(t=t,low=low,high=high,values=values));work['endpoint_completions']+=2;work['objective_products']+=2*T;work['objective_additions']+=2*(T+1)
    lo=min(rows,key=lambda r:r['values'][0]);hi=max(rows,key=lambda r:r['values'][1])
    assert len(K)<=T+1
    return dict(status='feasible',second_interval=[L,U],candidates=rows,range=[lo['values'][0],hi['values'][1]],witnesses=[lo['low'],hi['high']],work=work,cost_scope=premise)

def from_saved_v(box_v,coeff_v,constant,leader,second,gap_x):
    """v=2x: physical box halves, physical coefficients double; c unchanged."""
    return affine_range([[q(v)/2 for v in p] for p in box_v],[2*q(v) for v in coeff_v],constant,leader,second,gap_x)
