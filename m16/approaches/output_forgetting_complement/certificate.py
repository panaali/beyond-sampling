"""Exact rational affine-box certificate for suffix output ambiguity."""
from fractions import Fraction as F
from itertools import product,combinations
from collections import defaultdict
import time


def rational_model(model):return tuple(tuple(F(float(v)) for v in model[0])),tuple(tuple(F(float(v)) for v in row) for row in model[1]),tuple(tuple(F(float(v)) for v in row) for row in model[2])


def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))


def history(model,hidden,word,stats):
    wi,wh,wo=model;n=len(word);point=tuple(word)+(F(1),)
    state=[(F(0),)*n+(F(v),) for v in hidden];constraints=[];stable=True;gates=[]
    for t,x in enumerate(word):
        basis=tuple(F(int(i==t)) for i in range(n))+(F(0),)
        affine=[tuple(sum((wh[i][j]*state[j][k] for j in range(2)),F(0))+wi[i]*basis[k] for k in range(n+1)) for i in range(2)]
        new=[];mask=[]
        for i,row in enumerate(affine):
            value=dot(row,point);active=value>0;mask.append(active)
            signed=row if active else tuple(-v for v in row)
            if any(row):
                slack=dot(signed,point)
                if slack==0:stable=False
                constraints.append(dict(kind='gate',step=t,neuron=i,normal=signed,slack=slack))
            new.append(row if active else (F(0),)*(n+1))
        state=new;gates.append(mask);stats['affine_steps']+=1;stats['affine_scalar_products']+=2*2*(n+1)
    logits=[tuple(sum((row[j]*state[j][k] for j in range(2)),F(0)) for k in range(n+1)) for row in wo]
    scores=[dot(row,point) for row in logits];label=max(range(len(scores)),key=scores.__getitem__)
    for other in range(len(wo)):
        if other==label:continue
        margin=tuple(x-y for x,y in zip(logits[label],logits[other]));slack=dot(margin,point)
        if any(margin):
            if slack==0:stable=False
            constraints.append(dict(kind='readout',competitor=other,normal=margin,slack=slack))
        elif other<label:raise AssertionError('first-index ownership violated')
    stats['histories']+=1;stats['readout_affine_products']+=2*(n+1)*len(wo)
    return dict(initial_hidden=tuple(hidden),label=label,stable=stable,constraints=constraints,gates=gates)


def box_radius(histories):
    radius=F(1,4)
    for hist in histories:
        if not hist['stable']:return F(0)
        for c in hist['constraints']:
            slope=sum(map(abs,c['normal'][:-1]),F(0))
            if c['slack']<=0:return F(0)
            if slope:radius=min(radius,c['slack']/(2*slope))
    return radius


def find(model,seconds=15.,rescue=True):
    start=time.process_time();stats=defaultdict(int)
    states=[(F(0),F(0)),*[(F(s),F(0)) for s in (1,16,256)],*[(F(0),F(s)) for s in (1,16,256)]]
    for values in ((-1,1),(-2,-1,1,2)) if rescue else ((-1,1),):
        for word0 in product(values,repeat=4):
            if time.process_time()-start>seconds:raise TimeoutError('finite witness search CPU cap')
            if len(values)==4 and all(abs(x)==1 for x in word0):continue
            word=tuple(map(F,word0));stats['candidate_words']+=1
            histories=[history(model,s,word,stats) for s in states]
            for a,b in combinations(histories,2):
                stats['state_pair_tests']+=1
                if a['label']==b['label']:continue
                rho=box_radius((a,b))
                if rho:
                    R=max(map(abs,word))+rho;v=R*R/2;K=(v.numerator+v.denominator-1)//v.denominator
                    lower=(2*rho/F(3)**(K+1))**4
                    return dict(found=True,word=word,radius=rho,histories=[a,b],density_integer_K=K,exact_gaussian_mass_lower_bound=lower,stats=dict(stats),cpu_seconds=time.process_time()-start)
    return dict(found=False,stats=dict(stats),cpu_seconds=time.process_time()-start)


def direct(model,hidden,inputs):
    """No affine functions or certificate constraints used."""
    wi,wh,wo=model;state=list(hidden)
    for x in inputs:
        state=[max(F(0),wi[i]*x+sum((wh[i][j]*state[j] for j in range(2)),F(0))) for i in range(2)]
    scores=[sum((row[j]*state[j] for j in range(2)),F(0)) for row in wo]
    return max(range(len(scores)),key=scores.__getitem__)


def validate(model,result):
    if not result['found']:raise ValueError('no certificate to validate')
    count=0;minimum=None
    for signs in product((-1,1),repeat=4):
        inputs=tuple(c+s*result['radius'] for c,s in zip(result['word'],signs));point=inputs+(F(1),)
        for hist in result['histories']:
            label=direct(model,hist['initial_hidden'],inputs)
            if label!=hist['label']:raise AssertionError('direct vertex recurrence contradicts affine witness')
            for constraint in hist['constraints']:
                margin=dot(constraint['normal'],point)
                if margin<=0:raise AssertionError('open-box strict margin not retained')
                if margin<constraint['slack']/2:raise AssertionError('L1 radius bound failed')
                minimum=margin if minimum is None else min(minimum,margin)
            count+=1
    return dict(direct_vertex_recurrences=count,minimum_exact_margin=minimum)


def serial(value):
    if isinstance(value,F):return str(value)
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value
