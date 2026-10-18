"""Exact local affine enclosures; no checkpoint, model library or floating arithmetic."""
from collections import Counter

class Ledger:
    def __init__(self):self.counts=Counter();self.peak_bits=0
    def add(self,k,n=1):self.counts[k]+=n
    def see(self,*values):
        self.peak_bits=max(self.peak_bits,*(abs(v).bit_length() for v in values))
    def report(self):return dict(named_exact_expressions=dict(self.counts),maximum_observed_integer_bits=self.peak_bits,scope='Selected explicit integer expressions/dot terms only. Integer algorithms, control, allocation, comparisons not explicitly named and serialization are not a complete FLOP or native price.')

def dot(w,v,C):
    assert len(w)==len(v)
    C.add('integer_multiply',len(w));C.add('integer_add',len(w));C.add('dot_calls');C.add('dot_terms',len(w))
    return sum(a*b for a,b in zip(w,v,strict=True))

def irange(w,intervals,C):
    C.add('sign_tests',len(w))
    low=[];high=[]
    for a,(l,u) in zip(w,intervals,strict=True):
        if a>=0:low.append(l);high.append(u)
        else:low.append(u);high.append(l)
    return dot(w,low,C),dot(w,high,C)

def validate(model,box):
    assert set(model)=={'wi','wh','wo','D','H','T'}
    H,T,D=model['H'],model['T'],model['D']
    assert all(type(q)is int for q in (H,T,D)) and H>=1 and T>=2 and D>0
    assert len(model['wi'])==H and len(model['wh'])==H and all(len(r)==H for r in model['wh'])
    assert len(model['wo'])==T and all(len(r)==H for r in model['wo'])
    assert all(type(v)is int for v in model['wi']) and all(type(v)is int for key in ('wh','wo') for r in model[key] for v in r)
    assert len(box)==T and all(len(b)==2 and all(type(v)is int for v in b) and b[0]<b[1] for b in box)

def remainder(L,U,k):
    assert L<0<U and k in (0,1,2)
    return (2*U,max(-L,U),-2*L)[k]

def certify(model,box,C=None,retain_trace=False):
    validate(model,box);C=C or Ledger();H,T,D=(model[k] for k in ('H','T','D'))
    A=[[0]*T for _ in range(H)];E=[(0,0)]*H;B=[(0,0)]*H;S=2;trace=[]
    for t in range(T):
        C.add('exact_division');unit=S//2;assert unit*2==S
        C.add('integer_multiply',2);P=S*D;nextS=2*P
        cols=list(zip(*A));newA=[];newE=[];newB=[];rows=[]
        for i in range(H):
            w=model['wh'][i];a=[dot(w,col,C) for col in cols]
            C.add('integer_multiply');C.add('integer_add');a[t]+=model['wi'][i]*unit
            l,u=irange(w,E,C);al,au=irange(a,box,C);C.add('integer_add',2);al+=l;au+=u
            bl,bu=irange(w,B,C);xl,xu=irange([model['wi'][i]*unit],[box[t]],C);C.add('integer_multiply');C.add('integer_add',2);bl+=xl;bu+=xu
            C.add('range_intersections');L=max(al,bl);U=min(au,bu);assert L<=U
            C.add('gate_tests')
            if L>=0:
                k=2;r=0;kind='on'
            elif U<=0:
                k=0;r=0;kind='off'
            else:
                C.add('crossing_gates');C.add('integer_subtract');width=u-l
                rs=[remainder(L,U,j) for j in (0,1,2)]
                C.add('integer_multiply',5);C.add('integer_add',3)
                k=min(range(3),key=lambda j:(j*width+rs[j],j));r=rs[k];kind='crossing'
            C.add('integer_multiply',T+4);C.add('integer_add')
            aa=[k*q for q in a];ee=(k*l,k*u+r);bb=(2*max(0,L),2*max(0,U))
            if kind=='off':assert aa==[0]*T and ee==(0,0) and bb==(0,0)
            newA.append(aa);newE.append(ee);newB.append(bb)
            C.add('gate_'+kind);C.add('slope_'+str(k));C.see(*a,l,u,L,U,*aa,*ee,*bb,nextS)
            if retain_trace:rows.append(dict(pre_A=a,pre_error=[l,u],pre_range=[L,U],post_A=aa,post_error=list(ee),post_range=list(bb),kind=kind,slope_multiplier=k,remainder_upper=r))
        A,E,B,S=newA,newE,newB,nextS
        if retain_trace:trace.append(dict(step=t,pre_denominator=P,post_denominator=S,rows=rows))
    margins={};cols=list(zip(*A))
    for a in range(T):
        for b in range(a):
            C.add('integer_subtract',H);w=[u-v for u,v in zip(model['wo'][a],model['wo'][b],strict=True)]
            coeff=[dot(w,col,C) for col in cols];l,u=irange(w,E,C);al,au=irange(coeff,box,C);C.add('integer_add',2);al+=l;au+=u
            bl,bu=irange(w,B,C);C.add('range_intersections');L=max(al,bl);U=min(au,bu);assert L<=U
            margins[f'{a},{b}']=[L,U];C.see(*coeff,l,u,L,U)
    C.add('integer_multiply');den=S*D
    statuses=[classify(margins,j,T,C) for j in range(T)]
    ans=dict(margin_denominator=den,margins=margins,statuses=statuses)
    if retain_trace:ans['trace']=trace
    return ans

def classify(margins,target,T,C=None):
    good=True;bad=False
    for other in range(T):
        if target==other:continue
        l,u=margins[f'{max(target,other)},{min(target,other)}']
        if target<other:l,u=-u,-l
        if C:C.add('readout_ownership_tests',2)
        if other<target:
            good=good and l>0;bad=bad or u<=0
        else:
            good=good and l>=0;bad=bad or u<0
    assert not (good and bad)
    return 'correct' if good else 'wrong' if bad else 'unknown'

def target_group(bin_ids):
    assert len(bin_ids)==4 and all(type(q)is int for q in bin_ids)
    levels=sorted(set(bin_ids),reverse=True);top=[i for i,q in enumerate(bin_ids) if q==levels[0]]
    if len(top)>=2:return top
    assert len(levels)>=2
    return [i for i,q in enumerate(bin_ids) if q==levels[1]]
