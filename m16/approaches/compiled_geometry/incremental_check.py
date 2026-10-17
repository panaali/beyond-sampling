"""Frozen cache comparison: exact controls and unchanged 32-plane workload."""
import hashlib,json,resource,time
import numpy as np
from controls import fixtures
from fraction_oracle import exact_circle
from native import walk
from benchmark import load,HERE,ROOT,legacy

def main():
    start=time.process_time();out={'controls':[],'cases':[]}
    for name,model in fixtures():
        t=len(model[2]);u=np.zeros(t);v=np.zeros(t);u[:3]=(1/3,2/3,2/3);v[:3]=(2/3,1/3,-2/3)
        for scale in (1.,1e-12,1e12):
            scaled=(model[0]*scale,model[1],model[2]*scale);exact,_=exact_circle(scaled,u,v)
            full,fs=walk(scaled,u[None],v[None]);inc,ins=walk(scaled,u[None],v[None],engine='incremental')
            row=dict(name=name,scale=scale,exact=exact,incremental=float(inc[0]),difference=abs(float(inc[0])-exact),full_difference=float(abs(inc[0]-full[0])),same_pieces=fs['gate_pieces']==ins['gate_pieces'])
            assert row['difference']<1e-12 and row['same_pieces'] and abs(ins['coverage'][0]-1)<1e-14
            out['controls'].append(row)
    out['caps']=[]
    for cap in (0,1):
        try:walk(next(fixtures())[1],u[:3][None],v[:3][None],cap,engine='incremental')
        except RuntimeError as exc:out['caps'].append({'cap':cap,'raised':str(exc)})
        else:raise AssertionError('partial integral returned')
    out['control_cpu']=time.process_time()-start
    g=np.random.default_rng(660001).standard_normal((32,10,2));q,r=np.linalg.qr(g);q*=np.sign(np.diagonal(r,axis1=1,axis2=2))[:,None,:];u,v=q[:,:,0],q[:,:,1]
    base,sha=load(HERE.parent/'face_surplus_review/data/H16_T10_original.pth');cases=[]
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        cases.append((f'H16_sigma{sigma}',model,sha))
    h2,sha=load(ROOT/'work/checkpoints/H2_T10_seed0.pth');cases.append(('H2_T10',h2,sha))
    for name,model,sha in cases:
        row={'name':name,'checkpoint_sha256':sha}
        for repeat in ('cold','warm'):
            tick=time.process_time();full,fs=walk(model,u,v);row[f'full_{repeat}_cpu']=time.process_time()-tick
            tick=time.process_time();inc,ins=walk(model,u,v,engine='incremental');row[f'incremental_{repeat}_cpu']=time.process_time()-tick
            tick=time.process_time();old,pieces,partial=legacy.walk(model,u,v);row[f'legacy_{repeat}_cpu']=time.process_time()-tick
            assert fs['gate_pieces']==ins['gate_pieces']
            assert np.max(abs(full-inc))<1e-12 and np.max(abs(old-inc))<1e-12
        row.update(full_difference=float(np.max(abs(full-inc))),legacy_difference=float(np.max(abs(old-inc))),incremental_values=inc.tolist(),full_counts=fs,incremental_counts=ins,
            warm_speed_vs_legacy=row['legacy_warm_cpu']/row['incremental_warm_cpu'],warm_speed_vs_full=row['full_warm_cpu']/row['incremental_warm_cpu'],coefficient_step_ratio=sum(fs['coefficient_steps'])/sum(ins['coefficient_steps']))
        out['cases'].append(row);out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out['source_sha256']=hashlib.sha256((HERE/'incremental.cpp').read_bytes()).hexdigest()
        (HERE/'incremental_results.json').write_text(json.dumps(out,indent=2)+'\n')
        print(name,'legacy speed',row['warm_speed_vs_legacy'],'full speed',row['warm_speed_vs_full'],'step ratio',row['coefficient_step_ratio'],flush=True)

if __name__=='__main__':main()
