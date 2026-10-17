"""Trained-checkpoint application; analytic tails, random outer diagnostic only."""
import hashlib
import json
import pathlib
import sys
import time
import numpy as np
import torch

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'feynman_fundamentals'))
from tail import conditional_probability,prefix_summary


def execute(model,x):
    wi,wh,wo=model
    h=np.zeros((len(x),len(wi)))
    for z in x.T:
        h=np.maximum(h@wh.T+z[:,None]*wi,0)
    return (h@wo.T).argmax(axis=1)


def indicators(model,x):
    return execute(model,x)==np.argsort(x,axis=1)[:,-2]


def load(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    return tuple(d[key].numpy().astype(float).reshape(shape) for key,shape in (
        ('rnn.weight_ih_l0',(2,)),('rnn.weight_hh_l0',(2,2)),('linear.weight',(10,2))))


def main():
    start=time.process_time()
    checkpoint=pathlib.Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth')
    base=load(checkpoint)
    prefixes=np.random.default_rng(50110).standard_normal((32,8))
    output={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'tail_source_sha256':hashlib.sha256((HERE.parent/'feynman_fundamentals/tail.py').read_bytes()).hexdigest(),
        'classification':'analytic conditional tail; outer diagnostic sampling',
        'conditional_validation_inputs':0,'unconditional_reference_inputs':0,
        'outer_prefixes':96,'scale_prefixes':6,'prefix_recurrent_steps':816,'models':[]}
    for sigma in (0.,.0025,.02):
        noise=np.random.default_rng(1)
        model=tuple(w+sigma*np.sqrt(np.mean(w*w))*noise.standard_normal(w.shape) for w in base)
        rows=[]
        for i,prefix in enumerate(prefixes):
            h,top=prefix_summary(model,prefix)
            clock=time.process_time()
            try:
                p,stats=conditional_probability(model,h,top,8)
                row={'prefix':i,'probability':p,'stats':stats,'tail_cpu':time.process_time()-clock}
                if i<4:
                    rng=np.random.default_rng(50120+i)
                    x=np.column_stack((prefix[None,:]/np.linalg.norm(prefix)*np.sqrt(rng.chisquare(8,size=32768))[:,None],rng.standard_normal((32768,2))))
                    y=indicators(model,x)
                    observed=float(y.mean());se=float(np.sqrt(observed*(1-observed)/len(y)))
                    row['validation']={'draws':len(y),'estimate':observed,'se':se,'z':(p-observed)/se if se else None}
                    output['conditional_validation_inputs']+=len(y)
                if i<2:
                    scaled=(model[0]*1e-16,model[1],model[2]*1e-16)
                    sh,st=prefix_summary(scaled,prefix)
                    sp,ss=conditional_probability(scaled,sh,st,8)
                    row['scale_difference']=sp-p;row['scale_stats']=ss
                rows.append(row)
            except ArithmeticError as error:
                rows.append({'prefix':i,'failure':str(error),'tail_cpu':time.process_time()-clock})
        rng=np.random.default_rng(50130)
        x=rng.standard_normal((65536,10))
        clock=time.process_time();y=indicators(model,x);mc_cpu=time.process_time()-clock
        output['unconditional_reference_inputs']+=len(y)
        good=[r['probability'] for r in rows if 'failure' not in r]
        result={'sigma':sigma,'cases':rows,'mc_reference':float(y.mean()),
            'mc_reference_se':float(np.sqrt(y.var(ddof=1)/len(y))),
            'mc_cpu':mc_cpu,'failures':sum('failure' in r for r in rows),
            'tail_cpu_total':sum(r['tail_cpu'] for r in rows)}
        if len(good)==32:
            variance=float(np.var(good,ddof=1));mean=float(np.mean(good))
            result.update(outer_estimate=mean,outer_se=float(np.sqrt(variance/32)),
                conditional_variance=variance,
                equal_count_variance_gain=float(y.var(ddof=1)/variance) if variance else None,
                cpu_per_tail_vs_full_forward=(result['tail_cpu_total']/32)/(mc_cpu/len(y)))
        output['models'].append(result)
    output['cpu_seconds_excluding_imports']=time.process_time()-start
    (HERE/'trained_tail_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='models'}))
    for m in output['models']:
        print(json.dumps({k:v for k,v in m.items() if k!='cases'}))

if __name__=='__main__': main()
