"""Independent positive constant-label radial face formula, no network walk."""
import json,math,time,types,warnings,hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
from .tube_estimator import SOURCE,EXPECTED
HERE=Path(__file__).parent

def main():
    start=time.process_time();mp.mp.dps=70;text=SOURCE.read_text()
    assert hashlib.sha256(text.encode()).hexdigest()==EXPECTED
    C=types.ModuleType('constant_tube_geometry');exec(compile(text,str(SOURCE),'exec'),C.__dict__)
    amplitude=mp.sqrt(2)*mp.sqrt(2)*24/mp.mpf('11.631728396567448');terms={};fingerprint=hashlib.sha256()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The balance properties of Sobol')
        for kind,n,builder in (('23',42,C.faces23),('12',20,C.faces12)):
            face=builder(n,q=None)
            for key in ('x0','d','lo','hi','tpos','tneg'):fingerprint.update(np.asarray(face[key],dtype='<f8').tobytes())
            radius=np.linalg.norm(face['x0'],axis=1);tau=np.maximum(face['hi'],0)/(radius*math.sqrt(2));tex=face['lo']/(radius*math.sqrt(2))
            aa=face['d'].argmax(1);bb=face['d'].argmin(1);masses=[]
            for a,b,lo,hi in zip(aa,bb,tex,tau):
                lo,hi=mp.mpf(float(lo)),mp.mpf(float(hi));d=hi-lo
                mass=amplitude*d*mp.quad(lambda u:(1+(lo+d*u)**2)**-5,[0,1]) if d>0 else mp.mpf(0)
                masses.append(float(mass)*(int(a!=0)+int(b!=0)))
            terms['e'+kind]=float(face['Z']*np.mean(masses))
        x=C.blind_points(30);fingerprint.update(np.asarray(x,dtype='<f8').tobytes())
    terms['blind']=float(C.P_BLIND*np.mean(np.argsort(x,axis=1)[:,-2]!=0))
    accuracy=1-sum(terms.values());saved=json.loads((HERE/'tube_results.json').read_text())['cases'][2]
    assert fingerprint.hexdigest()==saved['native']['ledger']['face_sha256']
    assert abs(accuracy-saved['native']['estimate'])<1e-12
    predicted_legacy=1-terms['blind'];assert predicted_legacy==saved['legacy']['estimate']
    out=dict(complete=True,terms=terms,independent_accuracy=accuracy,analytic_truth=.1,native_difference=accuracy-saved['native']['estimate'],
             predicted_legacy_tie_bug_accuracy=predicted_legacy,face_sha256=fingerprint.hexdigest(),cpu_seconds=time.process_time()-start,model_integrations=0)
    (HERE/'tube_constant_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
