"""Exact open prefix boxes extending the frozen four-suffix witness.

The Lipschitz proof certifies the whole boxes; vertices are a separate check.
No search or floating model evaluations are used.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import time
import mpmath as mp
from certificate import direct, serial

HERE = Path(__file__).resolve().parent


def rational_tree(x):
    return tuple(rational_tree(v) for v in x) if isinstance(x, list) else F(x)


def hidden_after(model, inputs):
    wi, wh, _ = model
    h = [F(0), F(0)]
    for x in inputs:
        h = [max(F(0), wi[i]*x + sum(wh[i][j]*h[j] for j in range(2))) for i in range(2)]
    return tuple(h)


def mm(q):
    return mp.mpf(q.numerator)/q.denominator


def interval_mass(c, r):
    lo, hi = mm(c-r), mm(c+r)
    s = mp.sqrt(2)
    if hi < 0:
        return (mp.erfc(-hi/s)-mp.erfc(-lo/s))/2
    if lo > 0:
        return (mp.erfc(lo/s)-mp.erfc(hi/s))/2
    return (mp.erf(hi/s)-mp.erf(lo/s))/2


def run():
    start = time.process_time()
    saved = json.loads((HERE/'pilot.json').read_text())
    model = rational_tree(saved['model'])
    wi, wh, wo = model
    rho = F(saved['radius'])
    word = tuple(map(F, saved['word']))
    prefixes = [(F(1,4),)+(F(0),)*5,
                (F(1,4),)+(F(0),)*4+(-F(256)/(-wi[0]),)]
    M = max(sum(map(abs, row)) for row in wh)
    L6 = max(map(abs, wi))*sum(M**k for k in range(6))
    epsilon = F(1,16)
    bounds, tie_checks = [], []
    for index, hist in enumerate(saved['histories']):
        assert hidden_after(model, prefixes[index]) == tuple(map(F, hist['initial_hidden']))
        gates = [c for c in hist['constraints'] if c['kind']=='gate']
        assert {(c['step'], c['neuron']) for c in gates} == set(product(range(4),range(2)))
        retained = {c['competitor'] for c in hist['constraints'] if c['kind']=='readout'}
        for other in range(len(wo)):
            if other == hist['label'] or other in retained:
                continue
            delta = tuple(a-b for a,b in zip(wo[hist['label']],wo[other]))
            assert all(not active or delta[i]==0 for i,active in enumerate(hist['gates'][-1]))
            assert other > hist['label']
            tie_checks.append(dict(history=index,competitor=other,final_support=hist['gates'][-1]))
        for c in hist['constraints']:
            if c['kind']=='gate':
                coefficient = M**(c['step']+1)*L6
            else:
                coefficient = sum(abs(a-b) for a,b in zip(wo[hist['label']],wo[c['competitor']]))*M**4*L6
            slack = F(c['slack'])
            suffix_loss = rho*sum(abs(F(v)) for v in c['normal'][:-1])
            assert suffix_loss <= slack/2
            if coefficient:
                epsilon = min(epsilon,slack/(4*coefficient))
            bounds.append(dict(history=index,kind=c['kind'],step=c.get('step'),
                               neuron=c.get('neuron'),competitor=c.get('competitor'),
                               slack=slack,suffix_loss=suffix_loss,prefix_coefficient=coefficient))
    assert 0 < epsilon <= F(1,16) and rho < F(1,4)
    for b in bounds:
        b['prefix_loss'] = b['prefix_coefficient']*epsilon
        b['remaining_margin'] = b['slack']-b['suffix_loss']-b['prefix_loss']
        assert b['remaining_margin'] >= b['slack']/4 > 0
    mp.mp.dps = 80
    boxes = []
    for i, prefix in enumerate(prefixes):
        center = prefix+word
        radii = (epsilon,)*6+(rho,)*4
        count = 0
        for signs in product((-1,1),repeat=10):
            x = tuple(c+s*r for c,s,r in zip(center,signs,radii))
            assert x[8]>x[0] and all(x[0]>v for k,v in enumerate(x) if k not in (0,8))
            assert direct(model,(F(0),F(0)),x)==saved['histories'][i]['label']
            count += 1
        exponent = 0
        for c,r in zip(center,radii):
            v = (abs(c)+r)**2/2
            exponent += (v.numerator+v.denominator-1)//v.denominator+1
        log_lower = 6*mp.log10(2*mm(epsilon))+4*mp.log10(2*mm(rho))-exponent*mp.log10(3)
        mass = mp.fprod(interval_mass(c,r) for c,r in zip(center,radii))
        log_mass = mp.log10(mass)
        assert log_mass > log_lower
        boxes.append(dict(prefix=prefix,label=saved['histories'][i]['label'],target=0,
                          correct=saved['histories'][i]['label']==0,vertex_checks=count,
                          gaussian_lower_bound='(2*epsilon)^6*(2*rho)^4/3^exponent',
                          exponent=exponent,log10_mass_lower_bound=str(log_lower),
                          gaussian_mass_80_digit=str(mass),log10_gaussian_mass=str(log_mass)))
    return dict(checkpoint_sha256=saved['checkpoint_sha256'],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                witness_sha256=hashlib.sha256((HERE/'pilot.json').read_bytes()).hexdigest(),
                M=M,L6=L6,epsilon=epsilon,rho=rho,bounds=bounds,tie_checks=tie_checks,
                boxes=boxes,total_cpu_seconds=time.process_time()-start)


if __name__=='__main__':
    result=run()
    (HERE/'prefix_support.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    print(json.dumps(serial({k:result[k] for k in ('epsilon','boxes','total_cpu_seconds')})))
