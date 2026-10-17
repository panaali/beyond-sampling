"""Frozen controls and matched native topology; no accuracy truth selection."""
import hashlib,importlib.util,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
OLD=Path('/workspace/work/recover-small_h2_56/m16/approaches/compiled_geometry')
sys.path.insert(0,str(OLD))
from controls import fixtures
from fraction_oracle import exact_circle
from benchmark import load
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module('old_reset_native',OLD/'native.py')
new=module('new_reset_native',HERE/'native.py')

def run():
    start=time.process_time();out={'controls':[],'caps':[],'cases':[]}
    cases=list(fixtures())
    cases.extend([('late_reset',(np.array([.7,1.3]),-np.eye(2),np.array([[1.,0.],[0.,1.],[-1.,-1.]]))),
                  ('zero_Wi',(np.array([0.,1.]),-np.eye(2),np.array([[1.,0.],[0.,1.],[-1.,-1.]])))])
    for name,model in cases:
        t=len(model[2]);u=np.zeros(t);v=np.zeros(t);u[:3]=(1/3,2/3,2/3);v[:3]=(2/3,1/3,-2/3)
        for scale in (1.,1e-12,1e12):
            scaled=(model[0]*scale,model[1],model[2]*scale);truth,oracle=exact_circle(scaled,u,v)
            row=dict(name=name,scale=scale,truth=truth,oracle=oracle)
            for engine in ('factored','factored_mixed'):
                values,stats=new.walk(scaled,u[None],v[None],engine=engine)
                row[engine]=dict(value=float(values[0]),stats=stats)
                assert abs(values[0]-truth)<1e-12
                assert abs(stats['coverage'][0]-1)<1e-14
            out['controls'].append(row)
    # A late input with V=0 exercises the constant branch after an explicit reset.
    model=cases[-2][1];u=np.array([0.,1.,0.]);v=np.array([1.,0.,0.])
    truth,oracle=exact_circle(model,u,v)
    row=dict(name='late_constant_input',truth=truth,oracle=oracle)
    for engine in ('factored','factored_mixed'):
        values,stats=new.walk(model,u[None],v[None],engine=engine)
        row[engine]=dict(value=float(values[0]),stats=stats);assert abs(values[0]-truth)<1e-12
    out['controls'].append(row)
    assert any(sum(r['factored']['stats']['late_zero_steps'])>0 for r in out['controls'] if r['name']=='late_reset')
    for engine in ('factored','factored_mixed'):
        for cap in (0,1):
            try:new.walk(model,u[None],v[None],cap,engine=engine)
            except RuntimeError as exc:out['caps'].append(dict(engine=engine,cap=cap,error=str(exc)))
            else:raise AssertionError('cap returned partial circle')
    out['control_cpu']=time.process_time()-start
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    g=np.random.default_rng(660001).standard_normal((32,10,2));q,r=np.linalg.qr(g)
    q*=np.sign(np.diagonal(r,axis1=1,axis2=2))[:,None,:];u,v=q[:,:,0],q[:,:,1]
    base,sha=load(OLD.parent/'face_surplus_review/data/H16_T10_original.pth');models=[]
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        models.append((f'H16_sigma{sigma}',model,sha))
    h2,sha=load(Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth'));models.append(('H2_T10',h2,sha))
    for name,model,sha in models:
        row=dict(name=name,checkpoint_sha256=sha,runs={})
        for repeat in ('cold','warm'):
            for version,lib in (('old',old),('reset',new)):
                for engine in ('factored','factored_mixed'):
                    tick=time.process_time();values,stats=lib.walk(model,u,v,engine=engine);elapsed=time.process_time()-tick
                    row['runs'][f'{version}_{engine}_{repeat}']=dict(cpu_seconds=elapsed,values=values.tolist(),stats=stats)
        a=row['runs']['reset_factored_warm'];b=row['runs']['reset_factored_mixed_warm'];c=row['runs']['old_factored_warm'];d=row['runs']['old_factored_mixed_warm']
        row['old_piece_delta']=(np.array(d['stats']['gate_pieces'])-c['stats']['gate_pieces']).tolist()
        row['new_piece_delta']=(np.array(b['stats']['gate_pieces'])-a['stats']['gate_pieces']).tolist()
        row['long_piece_change']=(np.array(a['stats']['gate_pieces'])-c['stats']['gate_pieces']).tolist()
        row['mixed_piece_change']=(np.array(b['stats']['gate_pieces'])-d['stats']['gate_pieces']).tolist()
        row['max_old_new_integral_difference']=max(float(np.max(np.abs(np.array(x['values'])-c['values']))) for x in row['runs'].values())
        assert row['max_old_new_integral_difference']<1e-12
        row['paired_counts_match']=not any(row['new_piece_delta'])
        out['cases'].append(row);out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={engine:hashlib.sha256((HERE/f'{engine}.cpp').read_bytes()).hexdigest() for engine in ('factored','factored_mixed')})
        (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
        print(name,'pieces',sum(c['stats']['gate_pieces']),sum(d['stats']['gate_pieces']),'->',sum(a['stats']['gate_pieces']),sum(b['stats']['gate_pieces']),'matched',row['paired_counts_match'],'difference',row['max_old_new_integral_difference'],flush=True)
        if out['cpu_seconds']>100:raise RuntimeError('bounded CPU cap reached')
    print('total CPU',out['cpu_seconds'],'control CPU',out['control_cpu'],flush=True)

if __name__=='__main__':
    try:run()
    except Exception as exc:
        (HERE/'failure.json').write_text(json.dumps({'exception':repr(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise
