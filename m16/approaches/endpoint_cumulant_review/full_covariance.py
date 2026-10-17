"""Three-output Gaussian reference, independent of the Edgeworth endpoint."""
import time
START=time.process_time()
import hashlib,json,math,resource
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

HERE=Path(__file__).resolve().parent
TAU=64*np.finfo(float).eps


def normal_mass(lo,hi):
    if lo>=hi:return 0.
    return float(ndtr(-lo)-ndtr(-hi) if lo>=0 else ndtr(hi)-ndtr(lo))


def orthant(mu,V,strict,eps,stats):
    mu=np.asarray(mu);V=np.asarray(V)
    assert np.all(np.diag(V)>=0),'negative difference variance'
    random=[]
    for i in range(2):
        if V[i,i]==0:
            if mu[i]<0 or (mu[i]==0 and strict[i]):return 0.
        else:random.append(i)
    if not random:return 1.
    if len(random)==1:
        i=random[0];return float(ndtr(mu[i]/math.sqrt(V[i,i])))
    sd=np.sqrt(np.diag(V));a,b=mu/sd;rho=float(V[0,1]/sd[0]/sd[1])
    assert abs(rho)<=1+TAU,('invalid correlation',rho)
    if 1-abs(rho)<=TAU:
        stats['singular_limits']+=1
        if abs(rho)>1:stats['outside_unit_corrections'].append(rho)
        else:stats['singular_orthant_error_bounds'].append(math.acos(abs(rho))/(2*math.pi))
        return float(ndtr(min(a,b)) if rho>0 else normal_mass(-a,b))
    scale=math.sqrt((1-rho)*(1+rho))
    def integrand(z):return math.exp(-z*z/2)/math.sqrt(2*math.pi)*float(ndtr((b+rho*z)/scale))
    cuts=[-float(a)]
    if rho and -b/rho>cuts[0]:cuts.append(float(-b/rho))
    cuts.append(math.inf);total=0.
    for lo,hi in zip(cuts,cuts[1:]):
        out=quad(integrand,lo,hi,epsabs=eps,epsrel=eps,limit=300,full_output=1)
        if len(out)>3:raise RuntimeError(out[3])
        value,err,info=out
        total+=value;stats['quad_calls']+=1;stats['quad_evaluations']+=info['neval'];stats['quad_error_estimate_sum']+=err
    return total


def winners(mu,cov,eps=1e-12):
    mu=np.asarray(mu,float);cov=np.asarray(cov,float)
    assert mu.shape==(3,) and cov.shape==(3,3)
    stats=dict(quad_calls=0,quad_evaluations=0,quad_error_estimate_sum=0.,singular_limits=0,
        outside_unit_corrections=[],singular_orthant_error_bounds=[])
    q=[]
    for j in range(3):
        competitors=[k for k in range(3) if k!=j]
        D=np.eye(3)[j]-np.eye(3)[competitors]
        q.append(orthant(D@mu,D@cov@D.T,[k<j for k in competitors],eps,stats))
    stats['sum_residual']=sum(q)-1
    return dict(q=q,stats=stats)


def evaluate(K):
    mu=np.array(K['1']);cov=np.array(K['2'])
    coarse=winners(mu,cov,1e-10);fine=winners(mu,cov)
    scales=[winners(mu*s,cov*s*s) for s in [1e-8,1e8]]
    return dict(coarse=coarse,fine=fine,scales=scales,
        rule_sensitivity=float(np.max(abs(np.array(coarse['q'])-fine['q']))),
        scale_sensitivity=max(float(np.max(abs(np.array(x['q'])-fine['q']))) for x in scales))


def main():
    files={name:json.loads((HERE/name).read_text()) for name in ['controls.json','trained_t3.json','gate_mixture.json']}
    controls=files['controls.json'];trained=files['trained_t3.json'];mixture=files['gate_mixture.json']
    rows=[]
    for case in controls['cases']:
        result=evaluate(case['cumulants']);rows.append(dict(name=case['name'],exact_law_winners=case['exact_winners'],**result))
    # Explicit known Gaussian controls; scalar Z with shifted affine logits.
    analytic=[]
    for name,mu,cov,truth in [
        ('all_zero_tie',[0,0,0],np.zeros((3,3)),[1,0,0]),
        ('duplicate_top',[1,1,0],np.zeros((3,3)),[1,0,0]),
        ('opposite_lines',[.5,0,-.5],np.outer([1,0,-1],[1,0,-1]),[float(ndtr(.5)),0,float(ndtr(-.5))]),
        ('same_slope',[0,1,2],np.ones((3,3)),[0,0,1])]:
        result=evaluate({'1':mu,'2':cov});err=float(np.max(abs(np.array(result['fine']['q'])-truth)))
        assert err<1e-10;analytic.append(dict(name=name,truth=truth,error=err,**result))
    for row in rows[:2]:assert np.max(abs(np.array(row['fine']['q'])-row['exact_law_winners']))<1e-10
    trained_results=[]
    for group in trained['endpoint_cases']:
        acc=0.;detail=[]
        for case in group['conditional']:
            result=evaluate(case['cumulants']);j=case['target'];acc+=case['law_mass']*result['fine']['q'][j]
            if group['atom_separated'] and j==0:acc+=trained['geometry']['hzero_atom_joint'][0]
            detail.append(dict(target=j,**result))
        trained_results.append(dict(atom_separated=group['atom_separated'],accuracy=acc,bias=acc-mixture['truth'],cases=detail))
    acc=0.;detail=[]
    for case in mixture['components']:
        if case['fixed_winner'] is not None:acc+=case['mass']*(case['fixed_winner']==case['target']);continue
        result=evaluate(case['cumulants']);acc+=case['mass']*result['fine']['q'][case['target']]
        detail.append(dict(gate_cell=case['gate_cell'],order=case['order'],mass=case['mass'],**result))
    reset=(rows[3]['fine']['q'][0]+rows[3]['fine']['q'][1]+rows[2]['fine']['q'][2])/3
    out=dict(complete=True,controls=rows,analytic_controls=analytic,reset_accuracy=reset,reset_bias=reset-1/3,
        trained=trained_results,mixture=dict(accuracy=acc,bias=acc-mixture['truth'],cases=detail),
        source_pins={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in files},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cost_note='All four rule/scale evaluations counted in cases. Prior moments/geometry remain charged; no certified scalar FLOP accounting.')
    assert out['cpu_seconds_including_imports']<18
    (HERE/'full_covariance.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'reset_accuracy':reset,'trained':[{k:v for k,v in x.items() if k!='cases'} for x in trained_results],
        'mixture_accuracy':acc,'mixture_bias':acc-mixture['truth'],'cpu':out['cpu_seconds_including_imports']}))


if __name__=='__main__':main()
