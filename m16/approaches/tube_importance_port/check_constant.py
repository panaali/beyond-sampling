"""Independent positive radial integration for fixed first-index predictions."""
import hashlib,json,math,time,types,warnings,resource
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from .estimator import SOURCE,EXPECTED,digest
HERE=Path(__file__).parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));start=time.process_time()
    saved=json.loads((HERE/'results.json').read_text())['cases'][2]
    text=SOURCE.read_text();assert hashlib.sha256(text.encode()).hexdigest()==EXPECTED
    geom=types.ModuleType('independent_importance_geometry');exec(compile(text,str(SOURCE),'exec'),geom.__dict__)
    amplitude=48/11.631728396567448;cases=[]
    for mode in ('uniform','advice','cold'):
        expected=saved[mode];ledger=expected['ledger'];terms={};err=0;records={r['kind']:r for r in ledger['records'] if not r['pilot']}
        for kind,n in ledger['production_counts'].items():
            q=None if mode=='uniform' else np.load(SOURCE.with_name('q'+kind+'.npy')) if mode=='advice' else np.array(ledger['proposals'][kind]['proposal'])
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='The balance properties of Sobol')
                face=getattr(geom,'faces'+kind)(n,q=q)
            assert digest(face['x0'])==records[kind]['face_sha256']
            rho=np.linalg.norm(face['x0'],axis=1);lower=face['lo']/(rho*math.sqrt(2));upper=np.maximum(face['hi'],0)/(rho*math.sqrt(2));mass=[];bounds=[]
            for i,(lo,hi) in enumerate(zip(lower,upper)):
                length=max(0.,hi-lo)
                val,bnd=quad(lambda u:amplitude*length/(1+(lo+length*u)**2)**5,0,1,epsabs=2e-14,epsrel=2e-14)
                factor=int(face['tpos'][i]!=0)+int(face['tneg'][i]!=0)
                mass.append(val*factor);bounds.append(bnd*factor)
            terms['e'+kind]=float(face['Z']*np.dot(face['wt'],mass)/n)
            err+=float(face['Z']*np.dot(face['wt'],bounds)/n)
            assert abs(terms['e'+kind]-ledger['parts']['e'+kind])<1e-12
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='The balance properties of Sobol')
            xb=geom.blind_points(ledger['blind_points'])
        # Independent rank: count numbers strictly above each coordinate, then select rank1.
        second=np.array([next(j for j in range(10) if sum(float(z)>float(row[j]) for z in row)==1) for row in xb])
        terms['blind']=float(geom.P_BLIND*sum(j!=0 for j in second)/len(second))
        estimate=1-sum(terms.values());difference=estimate-expected['estimate'];assert abs(difference)<1e-12
        cases.append(dict(mode=mode,estimate=estimate,terms=terms,port_difference=difference,quadrature_reported_error=err,truth=.1,outer_rule_bias=estimate-.1))
    out=dict(complete=True,cases=cases,cpu_seconds=time.process_time()-start,network_calls=0,production_beta_calls=0,quadrature_error_is_not_rigorous_certificate=True)
    (HERE/'constant_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
