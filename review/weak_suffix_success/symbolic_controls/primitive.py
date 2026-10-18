"""Exact last-input partition; Phi expressions remain symbolic."""
from fractions import Fraction as F
from itertools import combinations


def rational(v):
    if type(v) not in (int, F):
        raise ValueError('exact int/Fraction required')
    if F(v).numerator.bit_length() > 128 or F(v).denominator.bit_length() > 128:
        raise ValueError('input bit cap')
    return F(v)


def validate(m):
    H, T = m['H'], m['T']
    if type(H) is not int or type(T) is not int or not 1 <= H <= 4 or not 3 <= T <= 4:
        raise ValueError('dimension')
    if len(m['Wi']) != H or len(m['h']) != H or len(m['Wh']) != H or len(m['Wo']) != T:
        raise ValueError('shape')
    if any(len(row) != H for row in m['Wh'] + m['Wo']):
        raise ValueError('matrix shape')
    wi = list(map(rational, m['Wi'])); h = list(map(rational, m['h']))
    wh = [list(map(rational, row)) for row in m['Wh']]
    wo = [list(map(rational, row)) for row in m['Wo']]
    a,b = map(rational, m['rank'][:2]); ia,ib = m['rank'][2:]
    if a <= b or any(type(i) is not int or not 0 <= i < T-1 for i in (ia,ib)) or ia == ib:
        raise ValueError('strict ranks/distinct prefix indices')
    if any(z < 0 for z in h):
        raise ValueError('negative hidden state')
    return H,T,wi,wh,wo,h,a,b,ia,ib


def sample(l,u):
    if l is None: return u-1 if u is not None else F(0)
    if u is None: return l+1
    return (l+u)/2


def inside(y,l,u):
    return (l is None or l < y) and (u is None or y < u)


def winner(values):
    return max(range(len(values)), key=values.__getitem__)


def cdf_expression(intervals):
    """Sufficient equality representation; no necessity/independence claim."""
    constant=F(0); terms={}
    for l,u in intervals:
        for v,sign,isupper in ((u,1,True),(l,-1,False)):
            if v is None:
                if isupper: constant += sign
            elif v == 0: constant += F(sign,2)
            elif v < 0:
                constant += sign; terms[-v] = terms.get(-v,0)-sign
            else: terms[v] = terms.get(v,0)+sign
    return {'constant':constant, 'terms':[[v,n] for v,n in sorted(terms.items()) if n]}


def partition(m):
    H,T,wi,wh,wo,h,a,b,ia,ib=validate(m)
    c=[sum((wh[k][j]*h[j] for j in range(H)),F(0)) for k in range(H)]
    gates=sorted(set(-c[k]/wi[k] for k in range(H) if wi[k]))
    ends=[None]+gates+[None]; pieces=[]; allroots=set(gates+[a,b]); pair_roots=0
    for l,u in zip(ends,ends[1:]):
        y=sample(l,u); mask=[int(c[k]+wi[k]*y>0) for k in range(H)]
        intercept=[sum((wo[o][k]*mask[k]*c[k] for k in range(H)),F(0)) for o in range(T)]
        slope=[sum((wo[o][k]*mask[k]*wi[k] for k in range(H)),F(0)) for o in range(T)]
        roots=[]; ties=[]
        for p,q in combinations(range(T),2):
            z=intercept[p]-intercept[q]; d=slope[p]-slope[q]
            if d:
                r=-z/d
                if inside(r,l,u): roots.append(r); allroots.add(r); pair_roots+=1
            elif not z: ties.append([p,q])
        pieces.append({'l':l,'u':u,'mask':mask,'intercept':intercept,'slope':slope,'roots':sorted(set(roots)),'permanent_ties':ties})
    roots=sorted(allroots); ends=[None]+roots+[None]; cells=[]; correct=[]
    for l,u in zip(ends,ends[1:]):
        y=sample(l,u); p=next(p for p in pieces if inside(y,p['l'],p['u']))
        logits=[c+d*y for c,d in zip(p['intercept'],p['slope'])]
        pred=winner(logits); truth=ib if y<b else (T-1 if y<a else ia)
        ok=pred==truth
        cells.append({'l':l,'u':u,'y':y,'logits':logits,'prediction':pred,'truth':truth,'correct':ok})
        if ok: correct.append([l,u])
    assert len(roots) <= H+2+(H+1)*T*(T-1)//2
    return {'c':c,'gate_roots':gates,'gate_pieces':pieces,'roots':roots,'cells':cells,'correct_intervals':correct,
            'symbolic':cdf_expression(correct),'counts':{'gate_cells':len(pieces),'pair_root_occurrences':pair_roots,
            'finite_breakpoints':len(roots),'intervals':len(cells),'constant_setup_dot_terms':H*H,
            'affine_readout_dot_terms':2*len(pieces)*T*H,'pair_checks':len(pieces)*T*(T-1)//2}}
