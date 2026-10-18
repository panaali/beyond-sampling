"""Pairwise high-precision angular antiderivatives independently check sector sums."""
import argparse,hashlib,json,math,resource,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import mpmath as mp
from prefix_covariance import prefix2
from prefix_mean import means
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'rescue_protocol.json').read_text());mp.mp.dps=50

def m(x):
    x=F(x);return mp.mpf(x.numerator)/x.denominator

def pair_integral(a,b,c,d):
    a,b,c,d=map(m,(a,b,c,d));cuts=[-mp.pi/2,mp.pi/2]
    for x,y in ((a,b),(c,d)):
        if x==0 and y==0:return mp.mpf(0)
        phi=mp.atan2(y,x)
        for k in range(-2,3):
            z=phi+mp.pi/2+k*mp.pi
            if cuts[0]<z<mp.pi/2:cuts.append(z)
    cuts=sorted(set(cuts));value=mp.mpf(0)
    def primitive(t):
        return (a*c+b*d)*t/2+(a*c-b*d)*mp.sin(2*t)/4-(a*d+b*c)*mp.cos(2*t)/4
    for lo,hi in zip(cuts,cuts[1:]):
        mid=(lo+hi)/2
        if a*mp.cos(mid)+b*mp.sin(mid)>0 and c*mp.cos(mid)+d*mp.sin(mid)>0:value+=primitive(hi)-primitive(lo)
    return value/mp.pi

def reference(wi,wh):
    fw=list(map(F,wi));h=len(wi);aa=[]
    for sign in (1,-1):
        aa.append([sum((F(v)*max(sign*w,F(0)) for v,w in zip(row,fw)),F(0)) for row in wh])
    mu=[sum((mp.sqrt(m(a)**2+m(w)**2)+m(a) for a in (aa[0][i],aa[1][i])),mp.mpf(0))/(2*mp.sqrt(2*mp.pi)) for i,w in enumerate(fw)]
    raw=[[sum((pair_integral(a[i],fw[i],a[j],fw[j]) for a in aa),mp.mpf(0)) for j in range(h)] for i in range(h)]
    cov=[[raw[i][j]-mu[i]*mu[j] for j in range(h)] for i in range(h)]
    return np.array(list(map(float,mu))),np.array([[float(v) for v in row] for row in cov]),dict(mean=list(map(str,mu)),covariance=[[str(v) for v in row] for row in cov])

def fixtures():
    w=np.array([1.,-2.,.5,-.25]);I=np.eye(4)
    return [('zero',np.zeros(4),np.zeros((4,4))),('zero_recurrent',w,np.zeros((4,4))),('identity',w,I),('negative_identity',w,-I),('duplicates_opposites',np.array([1.,1.,-1.,-1.]),np.array([[1,0,0,0],[1,0,0,0],[0,0,1,0],[0,0,1,0.]])),('mixed_rational',w,np.array([[1,-2,3,-4],[-1,.5,-.25,2],[0,0,0,0],[1,1,-1,-1.]]))]

def load_model():
    import torch
    p=Path(P['weight_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==P['weight_sha256'];st=torch.load(p,map_location='cpu',weights_only=True)
    return st['rnn.weight_ih_l0'].numpy().astype(float)[:,0],st['rnn.weight_hh_l0'].numpy().astype(float)

def check(name,wi,wh):
    t=time.process_time();mu,cov,meta=prefix2(wi,wh);cpu=time.process_time()-t
    rm,rc,raw=reference(wi,wh);closed=means(wi,wh)[0]
    ok=bool(np.allclose(mu,rm,atol=P['moment_abs_tolerance'],rtol=P['moment_rel_tolerance']) and np.allclose(cov,rc,atol=P['moment_abs_tolerance'],rtol=P['moment_rel_tolerance']) and np.allclose(mu,closed,atol=P['moment_abs_tolerance'],rtol=P['moment_rel_tolerance']))
    row=dict(name=name,passed=ok,wi=wi.tolist(),wh=wh.tolist(),mean=mu.tolist(),covariance=cov.tolist(),reference=raw,max_mean_error=float(np.max(abs(mu-rm))),max_covariance_error=float(np.max(abs(cov-rc))),min_covariance_eigenvalue=float(np.linalg.eigvalsh(cov)[0]),kernel_cpu_seconds=cpu,metadata=meta)
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['controls','trained']);mode=ap.parse_args().mode;resource.setrlimit(resource.RLIMIT_CPU,(20,20));t=time.process_time()
    rows=[check(*x) for x in fixtures()] if mode=='controls' else [check('H16T4seed3',*load_model())]
    out=dict(rows=rows,complete=all(r['passed'] for r in rows),cpu_seconds=time.process_time()-t,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,qualification='High precision antiderivative comparison, not outward certificate; all failures saved.')
    (HERE/('covariance_'+mode+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}));assert out['complete']
if __name__=='__main__':main()
