"""Exact cumulative pointwise allowances and final-step cone certificate."""
from fractions import Fraction as F

def propagate(wh,delta,remaining):
    for _ in range(remaining):delta=[sum(abs(a)*b for a,b in zip(row,delta)) for row in wh]
    return delta

def allowance(wh,wo,z,g,truth,remaining):
    delta=[max(F(0),v)-int(on)*v for v,on in zip(z,g)];assert all(v>=0 for v in delta)
    last=propagate(wh,delta,remaining)
    return [sum(abs(a-b)*d for a,b,d in zip(wo[truth],row,last)) for row in wo]

def decision(margins,errors,truth):
    others=[i for i in range(len(margins)) if i!=truth]
    if all(margins[i]-errors[i]>0 if i<truth else margins[i]-errors[i]>=0 for i in others):return 'correct'
    if any(margins[i]+errors[i]<=0 if i<truth else margins[i]+errors[i]<0 for i in others):return 'wrong'
    return 'unknown'

def final_cell(A,wo,g,truth,rays):
    # At the final step m_f is affine and e is convex/positively homogeneous.
    rows=[]
    for x in rays:
        z=[sum(a*b for a,b in zip(row,x)) for row in A];h=[v*int(on) for v,on in zip(z,g)];outputs=[sum(a*b for a,b in zip(row,h)) for row in wo]
        margins=[outputs[truth]-v for v in outputs];errors=allowance([],wo,z,g,truth,0);rows.append((margins,errors))
    others=[i for i in range(len(wo)) if i!=truth]
    correct=all(all(m[i]-e[i]>0 if i<truth else m[i]-e[i]>=0 for m,e in rows) for i in others)
    wrong=any(all(m[i]+e[i]<=0 if i<truth else m[i]+e[i]<0 for m,e in rows) for i in others)
    assert not(correct and wrong)
    return dict(status='correct' if correct else 'wrong' if wrong else 'unknown',vertex_margins_errors=rows)
