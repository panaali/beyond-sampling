"""Exact common-factor rescue; old numerical failures stay visible."""
import hashlib,json,resource,time
import numpy as np
from controls import fixtures
from fraction_oracle import exact_circle
from benchmark import load,HERE,ROOT,legacy
from native import walk

def run():
    start=time.process_time();out={'controls':[],'cases':[],'caps':[]}
    for name,model in fixtures():
        t=len(model[2]);u=np.zeros(t);v=np.zeros(t);u[:3]=(1/3,2/3,2/3);v[:3]=(2/3,1/3,-2/3)
        for scale in (1.,1e-12,1e12):
            scaled=(model[0]*scale,model[1],model[2]*scale);truth,_=exact_circle(scaled,u,v);row=dict(name=name,scale=scale,truth=truth)
            for engine in ('factored','factored_mixed'):
                val,stats=walk(scaled,u[None],v[None],engine=engine);row[engine]=float(val[0]);assert abs(val[0]-truth)<1e-12
            out['controls'].append(row)
    for engine in ('factored','factored_mixed'):
        for cap in (0,1):
            try:walk(next(fixtures())[1],u[:3][None],v[:3][None],cap,engine=engine)
            except RuntimeError as exc:out['caps'].append(dict(engine=engine,cap=cap,error=str(exc)))
            else:raise AssertionError('partial circle returned')
    g=np.random.default_rng(660001).standard_normal((32,10,2));q,r=np.linalg.qr(g);q*=np.sign(np.diagonal(r,axis1=1,axis2=2))[:,None,:];u,v=q[:,:,0],q[:,:,1]
    base,sha=load(HERE.parent/'face_surplus_review/data/H16_T10_original.pth');cases=[]
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base);cases.append((f'H16_sigma{sigma}',model))
    cases.append(('H2_T10',load(ROOT/'work/checkpoints/H2_T10_seed0.pth')[0]))
    for name,model in cases:
        row={'name':name};old,os=walk(model,u,v,engine='stable')
        for repeat in ('cold','warm'):
            results={}
            for engine in ('factored','factored_mixed'):
                wall=time.perf_counter();tick=time.process_time();val,stats=walk(model,u,v,engine=engine);row[f'{engine}_{repeat}_cpu']=time.process_time()-tick;row[f'{engine}_{repeat}_wall']=time.perf_counter()-wall;results[engine]=(val,stats)
                assert np.max(abs(val-old))<1e-12
            wall=time.perf_counter();tick=time.process_time();pv,_,_=legacy.walk(model,u,v);row[f'legacy_{repeat}_cpu']=time.process_time()-tick;row[f'legacy_{repeat}_wall']=time.perf_counter()-wall
        a,ast=results['factored'];b,bst=results['factored_mixed'];delta=np.array(bst['gate_pieces'])-ast['gate_pieces'];same=bool(np.all(delta==0))
        if name.startswith('H16'):assert same
        row.update(paired_counts_match=same,piece_differences=delta.tolist(),old_count_delta=(np.array(ast['gate_pieces'])-os['gate_pieces']).tolist(),factored_counts=ast,mixed_counts=bst,values=b.tolist(),paired_integral_difference=float(np.max(abs(a-b))),old_stable_integral_difference=float(np.max(abs(old-b))),legacy_integral_difference=float(np.max(abs(pv-b))),warm_cpu_speed=row['legacy_warm_cpu']/row['factored_mixed_warm_cpu'],warm_wall_speed=row['legacy_warm_wall']/row['factored_mixed_warm_wall'])
        out['cases'].append(row);out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={engine:hashlib.sha256((HERE/f'{engine}.cpp').read_bytes()).hexdigest() for engine in ('factored','factored_mixed')})
        (HERE/'factored_results.json').write_text(json.dumps(out,indent=2)+'\n');print(name,row['warm_cpu_speed'],row['warm_wall_speed'],same,row['old_stable_integral_difference'],flush=True)
if __name__=='__main__':
    try:run()
    except Exception as exc:
        import traceback
        (HERE/'factored_failure.json').write_text(json.dumps({'exception':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise
