"""Independent arrangement-vertex optimizer and positive probability controls."""
import hashlib,itertools,json,math,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
import torch
from .reset import coefficient,masses
HERE=Path(__file__).parent

def relu(x):return max(x,F(0))
def advance(wi,wh,h,x):return tuple(relu(sum(wh[i][j]*h[j] for j in (0,1))+wi[i]*x) for i in (0,1))
def vertices(wi,wh,x):
    # Independent 2D line arrangement: axes and the two actual gate planes.
    lines=[(F(1),F(0),F(0)),(F(0),F(1),F(0))]+[(wh[i][0],wh[i][1],-wi[i]*x) for i in (0,1)]
    out={(F(0),F(0))}
    for (a,b,e),(c,d,f) in itertools.combinations(lines,2):
        determinant=a*d-b*c
        if determinant:
            u,v=(e*d-b*f)/determinant,(a*f-e*c)/determinant
            if u>=0 and v>=0:out.add((u,v))
    return out

def main():
    start=time.process_time();out=dict(synthetic_cases=0,vertex_evaluations=0,reset_evaluations=0,probability_checks=[],models=[],random_model_inputs=0)
    for w in ((1,1),(2,1),(1,3)):
        wi=tuple(map(F,w))
        for a,b,c,d in itertools.product((0,1,2),repeat=4):
            wh=((F(-a),F(-b)),(F(c),F(-d)));result=coefficient(wi,wh)
            for x in map(F,(-2,0,1,3)):
                choices=vertices(wi,wh,x);values=[]
                for old in choices:
                    h=advance(wi,wh,old,x);values.append((F(c)*h[0]-F(d)*h[1],old));out['vertex_evaluations']+=1
                maximum,witness=max(values);assert maximum==result['gamma']*max(x,0)
                boundary=-result['kappa']*max(x,0)
                for old in choices|{(F(0),F(10000)),(F(10000),F(0)),(F(10000),F(10000))}:
                    for y in (boundary,boundary-1):
                        assert advance(wi,wh,advance(wi,wh,old,x),y)==(0,0);out['reset_evaluations']+=1
                assert advance(wi,wh,advance(wi,wh,witness,x),boundary+F(1,1024))[1]>0
                out['synthetic_cases']+=1
    phi=lambda x:math.exp(-x*x/2)/math.sqrt(2*math.pi)
    for k in (0.,.1,1.,3.,100.,math.inf):
        result=masses(k,8)
        if k==0:s2,s3=.5,.25;calls=0
        else:
            counts=[0]
            def fun(x):counts[0]+=1;return phi(x)*float(ndtr(x/k))
            negative=quad(fun,-np.inf,0,epsabs=1e-12)[0]
            def positive(x):counts[0]+=1;return phi(x)*float(ndtr(k*x))
            pos=quad(positive,0,np.inf,epsabs=1e-12)[0]
            s2=.5+negative;s3=pos+.5*negative;calls=counts[0]
        e2=abs(s2-masses(k,2)['no_reset_mass']);e3=abs(s3-masses(k,3)['no_reset_mass'])
        assert max(e2,e3)<1e-11 and abs(result['total_mass']-1)<1e-14
        out['probability_checks'].append(dict(kappa=str(k),n2_error=e2,n3_error=e3,quadrature_callbacks=calls,**result))
    assert masses(math.inf,8)['no_reset_mass']==55/256
    assert masses(0.,8)['no_reset_mass']==1/128
    checkpoint=Path('/workspace/clean-neurons/work/checkpoints/H2_T8_seed0.pth');state=torch.load(checkpoint,weights_only=True,map_location='cpu')
    base=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),base[1],base[2])
    out['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        row=dict(sigma=sigma,model_sha256=hashlib.sha256(b''.join(np.asarray(w,dtype='<f8').tobytes() for w in model)).hexdigest(),weights=[w.tolist() for w in model])
        try:
            exact=coefficient(model[0],model[1]);row.update(kappa=str(exact['kappa']),kappa_float=float(exact['kappa']),gamma=str(exact['gamma']),
                rational_operation_proxy=exact['rational_operation_proxy'],max_integer_bits=exact['max_integer_bits'],mass=masses(exact['kappa'],8))
            wi=tuple(F.from_float(float(x)) for x in model[0]);wh=tuple(tuple(F.from_float(float(x)) for x in w) for w in model[1])
            for scale in (F(1,2**40),F(2**40)):
                ss=(scale,1/scale);wis=tuple(ss[i]*wi[i] for i in (0,1));whs=tuple(tuple(ss[i]*wh[i][j]/ss[j] for j in (0,1)) for i in (0,1))
                assert coefficient(wis,whs)['kappa']==exact['kappa']
            choices=vertices(wi,wh,F(1));maxval=max(wh[1][0]*advance(wi,wh,old,F(1))[0]+wh[1][1]*advance(wi,wh,old,F(1))[1] for old in choices)
            assert maxval==exact['gamma'];row['exact_scaling_and_vertex_checks']=True
        except ValueError as exc:row['unsupported']=str(exc)
        out['models'].append(row)
    if time.process_time()-start>18:raise TimeoutError('20CPU cycle exceeded')
    out.update(complete=True,cpu_seconds=time.process_time()-start)
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(cpu_seconds=out['cpu_seconds'],synthetic_cases=out['synthetic_cases'],models=[{k:r.get(k) for k in ('sigma','kappa_float','unsupported')}|({'no_reset_mass':r['mass']['no_reset_mass']} if 'mass'in r else {}) for r in out['models']]),indent=2))
if __name__=='__main__':main()
