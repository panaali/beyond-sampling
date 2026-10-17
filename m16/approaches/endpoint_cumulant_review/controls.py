"""True conditional moment laws supplied to the unmodified ARC endpoint."""
import time
START=time.process_time()
import dataclasses,hashlib,json,math,resource,sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
import torch

HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/t40_sources')
LOG=json.loads((HERE/'source_log.json').read_text())
for row in LOG['sources']:assert hashlib.sha256((SOURCE/row['path']).read_bytes()).hexdigest()==row['sha256']
sys.path.insert(0,str(SOURCE/'src'))
from mlp_kprop.harmonic import HTensor
from mlp_kprop.max_endpoint.argmax import argmax_endpoint_estimate
from mlp_kprop.max_endpoint.quadrature import QuadratureCfg


def jsonable(x):
    if isinstance(x,torch.Tensor):return x.detach().numpy().tolist()
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x


def gaussian(name,cov):
    cov=np.array(cov);assert np.linalg.eigvalsh(cov).min()>0
    target=[]
    for i in range(3):
        j,k=[j for j in range(3) if j!=i];a=np.eye(3)[i]-np.eye(3)[j];b=np.eye(3)[i]-np.eye(3)[k]
        rho=(a@cov@b)/math.sqrt((a@cov@a)*(b@cov@b));target.append(.25+math.asin(rho)/(2*math.pi))
    return dict(name=name,moments={1:np.zeros(3),2:cov,3:np.zeros((3,)*3),4:np.zeros((3,)*4)},
                exact_winners=target,reference='zero-mean Gaussian two-difference arcsine',moment_integral_calls=0)


def reset(name,median):
    raw=[1.];errors=[];neval=0;started=time.process_time()
    for degree in range(1,5):
        def integrand(z):
            p,q=ndtr(z),ndtr(-z);density=(6*p*q if median else 1.5*(p*p+q*q))*math.exp(-z*z/2)/math.sqrt(2*math.pi)
            return z**degree*density
        result=quad(integrand,0,np.inf,epsabs=1e-12,epsrel=1e-12,full_output=1)
        assert len(result)==3,'quad convergence failure'
        value,error,info=result;raw.append(value);errors.append(error);neval+=info['neval']
    m1,m2,m3,m4=raw[1:]
    k=[m1,m2-m1*m1,m3-3*m1*m2+2*m1**3,m4-4*m1*m3-3*m2*m2+12*m1*m1*m2-6*m1**4]
    c=np.array([-1.,.25,1.]);moments={}
    outer=np.array(1.)
    for degree in range(1,5):
        outer=np.multiply.outer(outer,c);moments[degree]=k[degree-1]*outer
    return dict(name=name,moments=moments,exact_winners=[.5,0.,.5],
        reference='symmetric exact conditional order-statistic law; ReLU atom half and positive ray half',
        raw_hidden_moments=raw,hidden_cumulants=k,moment_error_estimates=errors,
        moment_integral_calls=4,moment_integrand_evaluations=neval,moment_cpu_seconds=time.process_time()-started)


def main():
    cases=[gaussian('iid_gaussian',np.eye(3)),
           gaussian('correlated_gaussian',[[1,.4,-.2],[.4,.81,.1],[-.2,.1,.64]]),
           reset('reset_target_last',True),reset('reset_target_other',False)]
    out=dict(complete=False,cases=[],source_commit='357c45578a40cb1807cd09bdc28e17d3d0277d0e',
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for case in cases:
        K={k:HTensor(torch.tensor(v,dtype=torch.float64),r=0) for k,v in case['moments'].items()}
        row={k:v for k,v in case.items() if k!='moments'};row['cumulants']={k:v.tolist() for k,v in case['moments'].items()};row['rules']=[]
        for q in [64,128]:
            started=time.process_time();result=argmax_endpoint_estimate(K,QuadratureCfg(num_nodes=q))
            record=jsonable(dataclasses.asdict(result));record['cpu_seconds']=time.process_time()-started
            record['raw_winner_error']={k:(np.array(v)-case['exact_winners']).tolist() for k,v in record['q_raw'].items()}
            row['rules'].append(record)
            print(case['name'],q,record['q_raw'],flush=True)
            if time.process_time()-START>18:raise TimeoutError('frozen review cycle deadline')
        out['cases'].append(row)
        (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
    last=out['cases'][2];other=out['cases'][3];estimates=[]
    for level in range(2):
        for name,q in last['rules'][level]['q_raw'].items():
            o=other['rules'][level]['q_raw'][name]
            raw=(o[0]+o[1]+q[2])/3
            qp=last['rules'][level]['q_projected'][name];op=other['rules'][level]['q_projected'][name]
            estimates.append(dict(nodes=last['rules'][level]['info']['num_nodes'],method=name,raw_accuracy=raw,
                raw_bias=raw-1/3,projected_accuracy=(op[0]+op[1]+qp[2])/3,
                exact_accuracy=1/3,atom_and_ray_rescue_accuracy=1/3))
    out.update(complete=True,reset_accuracy=estimates,cpu_seconds_including_imports=time.process_time()-START,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        setup_note='CPU includes initial diagram/rule construction and all repeats; upstream flop proxies omit/approximate setup and special-function work.')
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
