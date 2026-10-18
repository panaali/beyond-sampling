"""Frozen same-fixture derivative tail and spectral diagnostic."""
from collections import defaultdict
from pathlib import Path
import hashlib,importlib.util,json,sys,time
import mpmath as mp
import numpy as np
HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'plackett_precision_review'
PINS={'review.py':'74ab81048272c0d44fb0a60a214a134060cafb3c79df0a3c232e71d70c57943e',
      'rule_metadata.json':'ff1408c5e3cf5e9e435125608ffb39fe02504f3382f17a90fe978bb7862c2933',
      'fixture.json':'a4fee8f4bdc5d8ad5164831a141bc81f43c7b900fdb82b88c7f30603c5eaf600'}
for name,sha in PINS.items():assert hashlib.sha256((PRIOR/name).read_bytes()).hexdigest()==sha
spec=importlib.util.spec_from_file_location('pinned_t137',PRIOR/'review.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]


def derivative(C,t,stats):
    Rt=mp.eye(4)+t*(C-mp.eye(4));mp.cholesky(Rt);stats['positive_definite_checks']+=1
    Om=Rt**-1;stats['transformed_inversions']+=1
    parts=[];value=mp.mpf(0)
    for i,j in PAIRS:
        k,l=[q for q in range(4) if q!=i and q!=j]
        pc=-Om[k,l]/mp.sqrt(Om[k,k]*Om[l,l]);stats['partial_correlations']+=1
        assert -1<=pc<=1
        r=C[i,j];den=1-(t*r)**2;assert den>0
        value+=r*mp.asin(pc)/mp.sqrt(den);stats['arcsines']+=1;parts.append(str(pc))
    return dict(t=str(t),derivative=str(value/(4*mp.pi**2)),partial_correlations=parts)


def tail_bound(C,a,stats):
    value=mp.mpf(0)
    for i,j in PAIRS:
        r=abs(C[i,j]);assert r<1
        value+=mp.asin(r)-mp.asin(a*r);stats['arcsines']+=2
    return value/(8*mp.pi)


def cuts():
    metadata=json.loads((PRIOR/'rule_metadata.json').read_text())
    return [('last_node',prior.exact_float(metadata['rounded_rule'][-1][0]))]+[(f'one_minus_2pow_minus_{k}',1-mp.mpf(2)**(-k)) for k in (16,24,32)]


def controls():
    start=time.process_time();mp.mp.dps=80;stats=defaultdict(float);rows=[]
    for name,r,s in [('identity',mp.mpf(0),mp.mpf(0)),('independent_blocks',mp.mpf(1)/2,-mp.mpf(1)/3)]:
        C=mp.eye(4);C[0,1]=C[1,0]=r;C[2,3]=C[3,2]=s
        def prob(t):return (mp.mpf(1)/4+mp.asin(t*r)/(2*mp.pi))*(mp.mpf(1)/4+mp.asin(t*s)/(2*mp.pi))
        for label,a in cuts():
            got=derivative(C,a,stats)
            expected=r/(2*mp.pi*mp.sqrt(1-(a*r)**2))*(mp.mpf(1)/4+mp.asin(a*s)/(2*mp.pi))+s/(2*mp.pi*mp.sqrt(1-(a*s)**2))*(mp.mpf(1)/4+mp.asin(a*r)/(2*mp.pi))
            expected-=sum(C[i,j]/mp.sqrt(1-(a*C[i,j])**2) for i,j in PAIRS)/(8*mp.pi)
            difference=abs(mp.mpf(got['derivative'])-expected);assert difference<mp.mpf('1e-75')
            bound=tail_bound(C,a,stats);actual=prob(1)-prob(a)-sum(mp.asin(C[i,j])-mp.asin(a*C[i,j]) for i,j in PAIRS)/(8*mp.pi);assert abs(actual)<=bound
            stats['control_extra_arcsines']+=18
            rows.append(dict(name=name,cut=label,derivative_error=str(difference),exact_tail=str(actual),bound=str(bound)))
    return dict(rows=rows,work=dict(stats),cpu_seconds=time.process_time()-start,pins=PINS)


def fixture():
    start=time.process_time();prior.load_source()
    saved=json.loads((prior.SOURCE/'negative_mass_fixture.json').read_text());rays=np.asarray(saved['rays'])
    metadata=json.loads((PRIOR/'rule_metadata.json').read_text());old=json.loads((PRIOR/'fixture.json').read_text())
    rows=[]
    for precision in (80,100):
        mp.mp.dps=precision;stats=defaultdict(float);C=prior.correlation(rays,stats)
        eigenvalues=mp.eigsy(C,eigvals_only=True);stats['eigenproblems']+=1;assert min(eigenvalues)>0
        probes=[dict(name=f'source_node_{i}',**derivative(C,prior.exact_float(node[0]),stats)) for i,node in enumerate(metadata['rounded_rule'])]
        error=abs(mp.mpf(old['rows'][0]['arms']['exact_ray_C_rounded_rule']['mass'])-prior.exact_float(saved['positive']))
        tail_rows=[]
        for label,a in cuts():
            point=derivative(C,a,stats);B=tail_bound(C,a,stats);D=B+(1-a)*abs(mp.mpf(point['derivative']))
            tail_rows.append(dict(name=label,**point,bound=str(B),bound_over_rule_error=str(B/error),constant_extension_change_bound=str(D),extension_test_applicable=label=='last_node',extension_test_excludes_sole_endpoint=label=='last_node' and D<error))
        endpoint=derivative(C,mp.mpf(1),stats)
        rows.append(dict(precision=precision,eigenvalues=[str(v) for v in eigenvalues],condition_number=str(max(eigenvalues)/min(eigenvalues)),reference_error=str(error),nodes=probes,tails=tail_rows,endpoint=endpoint,work=dict(stats)))
    differences=[]
    for a,b in zip(rows[0]['eigenvalues'],rows[1]['eigenvalues']):differences.append(abs(mp.mpf(a)-mp.mpf(b)))
    for a,b in zip(rows[0]['tails'],rows[1]['tails']):
        for k in ('derivative','bound','constant_extension_change_bound'):differences.append(abs(mp.mpf(a[k])-mp.mpf(b[k])))
    maximum=max(differences);assert maximum<mp.mpf('1e-60')
    return dict(index=saved['index'],rows=rows,max_absolute_precision_difference=str(maximum),pins=PINS,cpu_seconds=time.process_time()-start)


if __name__=='__main__':
    name=sys.argv[1];assert name in ('controls','fixture')
    try:result=controls() if name=='controls' else fixture()
    except Exception as exc:
        (HERE/(name+'_failure.json')).write_text(json.dumps(dict(failure=repr(exc)),indent=2)+'\n');raise
    (HERE/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
