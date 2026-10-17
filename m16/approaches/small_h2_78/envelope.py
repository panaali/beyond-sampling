"""Exact rational readout envelope for nonnegative two-neuron hidden states."""
from fractions import Fraction as F
import numpy as np


def envelope(output):
    w=np.asarray(output,dtype=float)
    if w.ndim!=2 or w.shape[1]!=2 or not np.isfinite(w).all():
        raise ValueError('finite T by 2 output required')
    rows=[tuple(F(float(v)) for v in r) for r in w]
    T=len(rows);cuts={F(0)};arithmetic=0;comparisons=0
    for i,a in enumerate(rows):
        for b in rows[:i]:
            d=a[1]-b[1];arithmetic+=1
            if d:
                r=(b[0]-a[0])/d;arithmetic+=2;comparisons+=1
                if r>0:cuts.add(r)
    def winner(r):
        nonlocal arithmetic,comparisons
        values=[a+b*r for a,b in rows];arithmetic+=2*T;comparisons+=T-1
        return max(range(T),key=lambda i:values[i])
    cuts=sorted(cuts)
    points=[{'r':str(r),'winner':winner(r)} for r in cuts]
    cells=[]
    for l,r in zip(cuts,cuts[1:]):
        cells.append({'left':str(l),'right':str(r),'winner':winner((l+r)/2)})
        arithmetic+=2
    cells.append({'left':str(cuts[-1]),'right':None,'winner':winner(cuts[-1]+1)})
    arithmetic+=1;comparisons+=T-1
    axis=max(range(T),key=lambda i:rows[i][1])
    labels=sorted({0,axis,*[p['winner'] for p in points],*[p['winner'] for p in cells]})
    return {'coordinate':'h2/h1','cells':cells,'points':points,'axis_winner':axis,
            'zero_winner':0,'possible_labels':labels,'accuracy_upper_bound':len(labels)/T,
            'rational_arithmetic_operations':arithmetic,'rational_comparisons':comparisons,
            'sorting_cost_included':False}


def reset_certificate(model):
    wi,wh,wo=[np.asarray(w,dtype=float) for w in model];wi=wi.reshape(-1)
    if len(wi)!=2 or not (np.all(wi>0) and np.all(wh[0]<=0) and wh[1,1]<=0):
        return {'holds':False}
    q=[F(float(x)) for x in wi]
    scores=[sum(F(float(a))*b for a,b in zip(row,q)) for row in wo]
    return {'holds':True,'negative_negative_winner':0,
            'negative_negative_positive_winner':max(range(len(wo)),key=lambda i:scores[i]),
            'positive_ray_scores':[str(v) for v in scores]}


def nonvanishing_certificate(model):
    """Exact dyadic sign certificate for h_t != 0 after the first nonzero input.

    Opposite-sign input weights admit a strictly positive annihilator v. If
    v Wh has strictly positive coordinates, v(Wh h+Wi x)>0 for every h>=0,h!=0,
    so at least one next preactivation is positive for every real x.
    """
    wi,wh,wo=[np.asarray(w,dtype=float) for w in model];wi=wi.reshape(-1)
    q=[F(float(x)) for x in wi]
    if len(q)!=2 or q[0]*q[1]>=0:return {'holds':False}
    v=[abs(q[1]),abs(q[0])]
    assert sum(a*b for a,b in zip(v,q))==0
    coefficients=[sum(v[i]*F(float(wh[i,j])) for i in range(2)) for j in range(2)]
    if not all(c>0 for c in coefficients):return {'holds':False,'vWh':[str(c) for c in coefficients]}
    e=envelope(wo)
    labels=sorted({e['axis_winner'],*[r['winner'] for r in e['cells']],*[r['winner'] for r in e['points']]})
    return {'holds':True,'v':[str(x) for x in v],'vWh':[str(c) for c in coefficients],
            'nonzero_state_output_labels':labels,'accuracy_upper_bound':len(labels)/len(wo)}
