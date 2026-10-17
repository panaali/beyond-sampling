"""Independent positive full/near face integrals for constant first-label output."""
import ast,hashlib,json,math,time,types,warnings
from pathlib import Path
import numpy as np
import mpmath as mp
from .laplace_estimator import ROOT,TUBE,noimports
HERE=Path(__file__).parent

def main():
    start=time.process_time();mp.mp.dps=70
    C=types.ModuleType('laplace_constant_geometry');exec(compile(TUBE.read_text(),str(TUBE),'exec'),C.__dict__)
    ns=dict(C.__dict__);ns['COMBO']=np.array(C.COMBO23)
    tree=ast.parse((ROOT/'lp.py').read_text());definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build23')
    exec(compile(ast.Module(body=[definition],type_ignores=[]),'<geometry>','exec'),ns);exec(noimports((ROOT/'lqt.py').read_text()),ns)
    amplitude=mp.mpf(48)/mp.mpf('11.631728396567448');near_means={};faces=hashlib.sha256()
    def integrate(lo,hi):
        lo,hi=mp.mpf(float(lo)),mp.mpf(float(hi));d=hi-lo
        return amplitude*d*mp.quad(lambda u:(1+(lo+d*u)**2)**-5,[0,1]) if d>0 else mp.mpf(0)
    def constant(m,fc,k0,unused):
        for key in ('x0','d','lo','hi','tpos','tneg'):faces.update(np.asarray(fc[key],dtype='<f8').tobytes())
        radius=np.linalg.norm(fc['x0'],axis=1);lo=fc['lo']/(radius*math.sqrt(2));hi=np.maximum(fc['hi'],0)/(radius*math.sqrt(2))
        a=fc['d'].argmax(1);b=fc['d'].argmin(1);kind='12' if np.all(fc['tpos']==b) else '23'
        full=[];near=[]
        for ai,bi,low,high in zip(a,b,lo,hi):
            coefficient=int(ai!=0)+int(bi!=0)
            full.append(float(integrate(low,high))*coefficient)
            near.append(float(integrate(low,min(high,k0/math.sqrt(2))))*coefficient)
        near_means[kind]=float(np.mean(near));return np.array(full),np.zeros(len(full))
    ns['F_hybrid']=constant
    def constant_forward(m,x):faces.update(np.asarray(x,dtype='<f8').tobytes());return np.zeros((len(x),10))
    ns['forward_from']=constant_forward
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The balance properties of Sobol')
        e23=ns['e23'](None,1,0,.04)[0];e12=ns['e12'](None,1,0,.04)[0];blind=ns['eblind'](None,25,0)[0]
    correct=1-e23-e12-blind;near=C.Z23*near_means['23']+C.Z12*near_means['12'];predicted_legacy=correct+near
    saved=json.loads((HERE/'laplace_results.json').read_text())['cases'][2]
    assert faces.hexdigest()==saved['native']['ledger']['face_sha256']
    assert abs(correct-saved['native']['estimate'])<1e-12
    assert abs(predicted_legacy-saved['legacy']['estimate'])<1e-12
    out=dict(complete=True,independent_accuracy=correct,analytic_truth=.1,near_wrong_mass=near,predicted_legacy_tail_only=predicted_legacy,
        native_difference=correct-saved['native']['estimate'],legacy_difference=predicted_legacy-saved['legacy']['estimate'],
        face_sha256=faces.hexdigest(),cpu_seconds=time.process_time()-start,model_integrations=0)
    (HERE/'laplace_constant_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
