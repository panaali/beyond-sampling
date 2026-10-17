"""Selected seed2: exact-mode geometry capture only, one durable attempt."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,time,resource
import numpy as np
HERE=Path(__file__).resolve().parent
class Budget(Exception):pass

def main():
    start=time.process_time();out=HERE/'capture.json';progress=HERE/'capture_progress.jsonl';folder=HERE/'parents'
    assert not out.exists() and not progress.exists() and not folder.exists()
    P=json.loads((HERE/'capture_protocol.json').read_text());assert P['source_gate']
    for path,sha in P['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    result=dict(complete=False,parents=[],protocol_sha256=hashlib.sha256((HERE/'capture_protocol.json').read_bytes()).hexdigest(),mass_evaluations=0,random_inputs=0,neural_forward_input_evaluations=0)
    def log(x):
        with progress.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
    def save():
        result.update(cpu=time.process_time()-start,process_cpu=time.process_time(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out)
    current=[];E=None;guards=None;main_ops=None;instances=[];parent_index=-1
    try:
        log(dict(stage='preflight_attempt'));folder.mkdir()
        # Prove exact-mode source compatibility before importing reviewed adapter.
        def functions(path):return {n.name:n for n in ast.parse(Path(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old,new=functions(P['old_engine']),functions(P['current_engine']);dump=lambda n:ast.dump(n,include_attributes=False)
        check={k:dump(v)==dump(new[k]) for k,v in old.items() if k not in ('orthant','_margin_split')};assert all(check.values())
        margin=copy.deepcopy(new['_margin_split']);removed=[]
        for i,n in enumerate(margin.body):
            if isinstance(n,ast.If) and ast.unparse(n.test)=='DIAG[0] is not None and (not exact)':removed.append(i)
        assert len(removed)==1
        assert ast.unparse(margin.body[removed[0]])=='if DIAG[0] is not None and (not exact):\n    for k in trig:\n        DIAG[0](c, A, g, k, t, model, truth)'
        margin.body.pop(removed[0]);assert dump(margin)==dump(old['_margin_split'])
        result['source_compatibility']=dict(functions=check,margin_split_after_only_nonexact_DIAG_removal=True,mass_callback_overridden=True,exact_mode_only=True)
        spec=importlib.util.spec_from_file_location('t193_reviewed_adapter',P['adapter']);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        tick=time.process_time();E,guards,*unused=adapter.load();result['adapter_load_cpu']=time.process_time()-tick
        import torch
        torch.set_num_threads(1);tick=time.process_time();state=torch.load(P['checkpoint'],map_location='cpu',weights_only=True)
        assert set(state)=={'rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'}
        wi=state['rnn.weight_ih_l0'].numpy().astype(float)[:,0];wh=state['rnn.weight_hh_l0'].numpy().astype(float);wo=state['linear.weight'].numpy().astype(float)
        assert wi.shape==(8,) and wh.shape==(8,8) and wo.shape==(4,8) and all(np.isfinite(x).all() for x in (wi,wh,wo))
        model=(wi,wh,wo);result['checkpoint_load_cpu']=time.process_time()-tick;result['model_hex']=[[[float(v).hex() for v in row] for row in x] if x.ndim==2 else [float(v).hex() for v in x] for x in model]
        result['checkpoint_sha256']=P['pins'][P['checkpoint']]
        base_ops=E.Ops
        class Meter(base_ops):
            def __init__(self):super().__init__();instances.append(self)
            def add(self,n):
                super().add(n)
                if time.process_time()-start>95:raise Budget('95CPU soft cap')
        E.Ops=Meter;E._HALVES.clear();E.STATS.update({k:0 for k in E.STATS});guards.clear();main_ops=Meter()
        def record(R,ops):
            assert R.shape==(4,4) and R.dtype==np.float64 and np.isfinite(R).all()
            current.append(R.copy());E.STATS['simplices']+=1;return 0.
        E.orthant=record;finish=E._finish
        def durable(c,g,A,t,model,truth,n_true,eps,ops,exact):
            nonlocal current,parent_index
            assert exact
            if t!=1:return finish(c,g,A,t,model,truth,n_true,eps,ops,exact)
            assert not current;parent_index+=1;idx=parent_index;before=dict(E.STATS);tick=time.process_time();log(dict(stage='parent_attempt',index=idx,truth=int(truth)))
            val=finish(c,g,A,t,model,truth,n_true,eps,ops,exact);assert val==0.
            rays=np.stack(current) if current else np.empty((0,4,4),dtype=np.float64)
            path=folder/('parent_%02d.npz'%idx);assert not path.exists();np.savez_compressed(path,rays=rays)
            row=dict(index=idx,truth=int(truth),N=len(current),stats_delta={k:E.STATS[k]-before[k] for k in E.STATS},archive=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cpu_including_archive=time.process_time()-tick,copied_float64_elements=16*len(current),ray_copy_bytes=128*len(current),archive_stacking_float64_elements=16*len(current))
            result['parents'].append(row);current=[];log(dict(stage='parent_complete',row=row));save();return val
        E._finish=durable;log(dict(stage='capture_start'));score=E.estT(model,main_ops,mode='exact');assert score==1.
        assert len(result['parents'])==96 and E.STATS['cells']==P['expected_leaves'] and sum(r['N'] for r in result['parents'])==E.STATS['simplices']
        result.update(complete=True,stats=dict(E.STATS),guard_counts=dict(guards),main_geometry_proxy=main_ops.n,auxiliary_proxy_instances=[x.n for x in instances if x is not main_ops],all_proxy_instances_sum=sum(x.n for x in instances),ignored_zero_callback_accuracy=score,N=sum(r['N'] for r in result['parents']),ray_copy_bytes=sum(r['ray_copy_bytes'] for r in result['parents']),parent_archive_count=96)
        log(dict(stage='capture_complete',N=result['N'],leaves=E.STATS['cells']));save()
    except BaseException as e:
        result['failure']=repr(e)
        if current:
            path=HERE/'partial_parent.npz';assert not path.exists();np.savez_compressed(path,rays=np.stack(current));result['partial_parent']=dict(index=parent_index,N=len(current),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if E is not None:result['partial_stats']=dict(E.STATS)
        if guards is not None:result['guard_counts']=dict(guards)
        result['partial_proxy_instances']=[x.n for x in instances];log(dict(stage='failure',error=repr(e)));save();raise
    print(json.dumps(dict(complete=True,N=result['N'],leaves=result['stats']['cells'],cpu=result['cpu'])))
if __name__=='__main__':main()
