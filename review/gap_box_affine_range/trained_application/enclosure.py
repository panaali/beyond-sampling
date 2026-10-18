"""Exact physical-Fraction enclosure; no saved data or model libraries imported."""
from fractions import Fraction as F
from collections import Counter
from range_primitive import affine_range,q

class Ledger:
    def __init__(self):self.counts=Counter();self.numerator_bits=0;self.denominator_bits=0
    def add(self,key,n=1):self.counts[key]+=n
    def see(self,value):
        if isinstance(value,F):
            self.numerator_bits=max(self.numerator_bits,abs(value.numerator).bit_length())
            self.denominator_bits=max(self.denominator_bits,value.denominator.bit_length())
        elif isinstance(value,(list,tuple)):
            for x in value:self.see(x)
        elif isinstance(value,dict):
            for x in value.values():self.see(x)
    def report(self):return dict(selected_counts=dict(self.counts),observed_numerator_bits=self.numerator_bits,observed_denominator_bits=self.denominator_bits,scope='Selected source categories and observed Fraction components only; incomplete arithmetic/control/conversion/comparison/sorting/runtime price. No full FLOP or sampling comparison.')

def normalize_saved(constant,coeff,error,bounds,denominator):
    if type(denominator)is not int or denominator<=0:raise ValueError('positive integer denominator required')
    if len(coeff)<3 or len(error)!=2 or len(bounds)!=2:raise ValueError('normalization dimensions')
    c=q(constant);a=list(map(q,coeff));e=list(map(q,error));b=list(map(q,bounds))
    if e[0]>e[1] or b[0]>b[1]:raise ValueError('reversed normalized interval')
    return dict(constant=c/denominator,coeff=[2*x/denominator for x in a],error=[x/denominator for x in e],bounds=[x/denominator for x in b])

def validate(model,box):
    if set(model)!={'wi','wh','wo','D','H','T'}:raise ValueError('model keys')
    H,T,D=(model[k] for k in ('H','T','D'))
    if any(type(x)is not int for x in (H,T,D)) or H<1 or T<3 or D<=0:raise ValueError('model dimensions or denominator')
    if len(model['wi'])!=H or len(model['wh'])!=H or any(len(row)!=H for row in model['wh']) or len(model['wo'])!=T or any(len(row)!=H for row in model['wo']):raise ValueError('weight shape')
    if any(type(x)is not int for x in model['wi']) or any(type(x)is not int for name in ('wh','wo') for row in model[name] for x in row):raise ValueError('integer weight numerators required')
    if len(box)!=T or any(len(p)!=2 for p in box):raise ValueError('box shape')
    b=[[q(x) for x in p] for p in box]
    if any(l>u for l,u in b):raise ValueError('reversed box')
    return H,T,D,b

def classify(margins,target,T):
    good=True;bad=False
    for rival in range(T):
        if rival==target:continue
        l,u=margins[f'{max(target,rival)},{min(target,rival)}']
        if target<rival:l,u=-u,-l
        if rival<target:good=good and l>0;bad=bad or u<=0
        else:good=good and l>=0;bad=bad or u<0
    assert not(good and bad)
    return 'correct' if good else 'wrong' if bad else 'unknown'

def certify(model,box,domain=None,old_margins=None):
    H,T,D,b=validate(model,box);C=Ledger();C.add('box_rational_values',2*T)
    feasibility=None
    if domain is not None:
        if set(domain)!={'leader','second','gap'}:raise ValueError('domain keys')
        feasibility=affine_range(b,[0]*T,0,**domain);C.add('feasibility_range_calls');C.see(feasibility)
        if feasibility['status']=='empty':return dict(status='empty',feasibility=feasibility,trace=[],raw_margins=None,final_margins=None,old_margins=None,raw_statuses=None,statuses=None,accounting=C.report())
    wi=[F(v,D) for v in model['wi']];wh=[[F(v,D) for v in row] for row in model['wh']];wo=[[F(v,D) for v in row] for row in model['wo']]
    C.add('weight_rational_values',H+H*H+T*H);C.see([b,wi,wh,wo])
    def dot(a,x):
        C.add('dot_calls');C.add('dot_terms',len(a))
        return sum((z*v for z,v in zip(a,x,strict=True)),F(0))
    def irange(a,ivs):
        C.add('interval_sign_choices',len(a))
        return [dot(a,[l if z>=0 else u for z,(l,u) in zip(a,ivs,strict=True)]),dot(a,[u if z>=0 else l for z,(l,u) in zip(a,ivs,strict=True)])]
    def arange(c,a):
        if domain is None:
            C.add('whole_affine_range_calls');l,u=irange(a,b);return [c+l,c+u]
        C.add('restricted_affine_range_calls');answer=affine_range(b,a,c,**domain)
        assert answer['status']=='feasible'
        for k,n in answer['work'].items():C.add('range_'+k,n)
        C.see(answer);return answer['range']
    A=[[F(0)]*T for _ in range(H)];const=[F(0)]*H;E=[[F(0),F(0)] for _ in range(H)];B=[[F(0),F(0)] for _ in range(H)];trace=[]
    for t in range(T):
        cols=list(zip(*A));nextA=[];nextc=[];nextE=[];nextB=[];rows=[]
        for i,w in enumerate(wh):
            c=dot(w,const);a=[dot(w,col) for col in cols];a[t]+=wi[i];e=irange(w,E)
            al,au=arange(c,a);aff=[al+e[0],au+e[1]]
            bl,bu=irange(w,B);xl,xu=irange([wi[i]],[b[t]]);direct=[bl+xl,bu+xu]
            L=max(aff[0],direct[0]);U=min(aff[1],direct[1]);assert L<=U,'invalid preactivation intersection'
            C.add('range_intersections');C.add('gate_decisions')
            if L>=0:lam=F(1);r=F(0);kind='on'
            elif U<=0:lam=F(0);r=F(0);kind='off'
            else:
                choices=[F(0),F(1,2),F(1)];rs=[max(-v*L,(1-v)*U) for v in choices]
                k=min(range(3),key=lambda j:(choices[j]*(e[1]-e[0])+rs[j],choices[j]))
                lam=choices[k];r=rs[k];kind='crossing';C.add('crossing_slope_candidates',3)
            aa=[lam*z for z in a];cc=lam*c;ee=[lam*e[0],lam*e[1]+r];bb=[max(F(0),L),max(F(0),U)]
            record=dict(pre_constant=c,pre_A=a,pre_error=e,affine_range=aff,direct_range=direct,pre_range=[L,U],post_constant=cc,post_A=aa,post_error=ee,post_range=bb,slope=lam,remainder_upper=r,kind=kind)
            rows.append(record);C.see(record);C.add('gate_'+kind);C.add('slope_'+str(lam))
            nextA.append(aa);nextc.append(cc);nextE.append(ee);nextB.append(bb)
        trace.append(dict(step=t,rows=rows));A,const,E,B=nextA,nextc,nextE,nextB
    cols=list(zip(*A));raw={};margin_trace={}
    for i in range(T):
        for j in range(i):
            w=[a-bb for a,bb in zip(wo[i],wo[j],strict=True)];c=dot(w,const);a=[dot(w,col) for col in cols];e=irange(w,E)
            al,au=arange(c,a);aff=[al+e[0],au+e[1]];direct=irange(w,B)
            L=max(aff[0],direct[0]);U=min(aff[1],direct[1]);assert L<=U,'invalid readout intersection'
            key=f'{i},{j}';raw[key]=[L,U];margin_trace[key]=dict(constant=c,coeff=a,error=e,affine_range=aff,direct_range=direct)
            C.add('readout_contrasts');C.add('range_intersections');C.see([raw[key],margin_trace[key]])
    old=None;final={k:v[:] for k,v in raw.items()}
    if old_margins is not None:
        if set(old_margins)!=set(raw):raise ValueError('old margin contrast keys')
        old={}
        for k,v in old_margins.items():
            if len(v)!=2:raise ValueError('old margin interval')
            l,u=map(q,v)
            if l>u:raise ValueError('reversed old margin')
            old[k]=[l,u];final[k]=[max(l,raw[k][0]),min(u,raw[k][1])]
            assert final[k][0]<=final[k][1],'contradictory old/new certificate'
            C.add('fallback_intersections')
    C.see(final)
    return dict(status='feasible',feasibility=feasibility,trace=trace,margin_trace=margin_trace,raw_margins=raw,final_margins=final,old_margins=old,raw_statuses=[classify(raw,j,T) for j in range(T)],statuses=[classify(final,j,T) for j in range(T)],accounting=C.report())
