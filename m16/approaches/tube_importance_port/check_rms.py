"""Frozen RMS proposal comparison; all pilots paid, no accuracy-truth tuning."""
import json,hashlib,time,resource,warnings,types,math
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import quad
from . import estimator as E
from ..feynman_incremental.check_estimator import CHECKPOINT
HERE=Path(__file__).parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));start=time.process_time()
    values=np.repeat([1.,2.,4.,8.],3);groups=np.repeat(np.arange(4),3);costs=np.ones(12);a=np.array([1.,2.,4.,8.]);mu=a.mean();synthetic={}
    for score in ('sd','rms'):
        q,_=E.fit_proposal(values,costs,groups,4,score_kind=score)
        second=sum(a[j]**2/(16*q[j]) for j in range(4));mean=sum(q[j]*a[j]/(4*q[j]) for j in range(4));assert abs(mean-mu)<1e-14
        synthetic[score]=dict(q=q.tolist(),mean=float(mean),variance=float(second-mu**2))
    expected=.05+.8*a/a.sum();assert np.max(abs(np.array(synthetic['rms']['q'])-expected))<1e-15
    assert synthetic['rms']['variance']<synthetic['sd']['variance']
    prior=json.loads((HERE/'results.json').read_text());state=torch.load(CHECKPOINT,weights_only=True,map_location='cpu')
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==prior['checkpoint_sha256']
    model=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
    out=dict(synthetic=synthetic,cases=[],accuracy_or_mse_claim=False)
    for idx,(name,budget,m) in enumerate((('checkpoint512',512,model),('checkpoint2048',2048,model),('zero_readout512',512,(model[0],model[1],np.zeros_like(model[2]))))):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='The balance properties of Sobol')
            value,cost,ledger=E.estimate(m,budget,score_kind='rms')
        old=prior['cases'][idx]['cold'];assert ledger['known_flops']==sum(ledger['components'].values())
        for l,r in zip(ledger['records'][:2],old['ledger']['records'][:2]):
            assert l['face_sha256']==r['face_sha256'] and l['mass_sha256']==r['mass_sha256']
        case=dict(name=name,estimate=value,cost=cost,ledger=ledger,sd_estimate=old['estimate'],difference_from_sd=value-old['estimate'])
        if name=='zero_readout512':
            geom=types.ModuleType('rms_constant_geometry');text=E.SOURCE.read_text();assert hashlib.sha256(text.encode()).hexdigest()==E.EXPECTED;exec(compile(text,str(E.SOURCE),'exec'),geom.__dict__)
            terms={}
            for kind,n in ledger['production_counts'].items():
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore',message='The balance properties of Sobol')
                    fc=getattr(geom,'faces'+kind)(n,q=np.array(ledger['proposals'][kind]['proposal']))
                rec=next(r for r in ledger['records'] if r['kind']==kind and not r['pilot']);assert E.digest(fc['x0'])==rec['face_sha256']
                rho=np.linalg.norm(fc['x0'],axis=1);lower=fc['lo']/(rho*math.sqrt(2));upper=np.maximum(fc['hi'],0)/(rho*math.sqrt(2));mass=[]
                for i,(lo,hi) in enumerate(zip(lower,upper)):
                    length=max(0.,hi-lo);v,_=quad(lambda t:(48/11.631728396567448)*length/(1+(lo+length*t)**2)**5,0,1,epsabs=2e-14,epsrel=2e-14)
                    mass.append(v*(int(fc['tpos'][i]!=0)+int(fc['tneg'][i]!=0)))
                terms['e'+kind]=float(fc['Z']*np.dot(fc['wt'],mass)/n)
            independent=1-sum(terms.values())-ledger['parts']['blind'];assert abs(independent-value)<1e-12
            case.update(independent_face_estimate=independent,independent_difference=independent-value,exact_truth=.1)
        out['cases'].append(case);print(json.dumps(dict(name=name,value=value,cost=cost,sd_difference=case['difference_from_sd'])),flush=True)
    out.update(complete=True,cpu_seconds=time.process_time()-start,maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (HERE/'rms_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
