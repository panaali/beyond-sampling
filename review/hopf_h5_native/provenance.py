"""Reconstruct only the original25 saved models; no forward/estimator/MC calls."""
from pathlib import Path
import ast,hashlib,json,resource,time,types
import numpy as np
import torch
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def model_hash(model):return hashlib.sha256(b''.join(w.astype('<f8').tobytes(order='C') for w in model)).hexdigest()
def function_tree(path):
    tree=ast.parse(Path(path).read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='noisy_copies'];assert len(tree.body)==1;return tree

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(19,20));dest=P/'models.json';assert not dest.exists();t=time.process_time()
    out=dict(complete=False,source_sha256=sha(__file__),protocol_sha256=sha(P/'protocol.json'),checkpoint_identity_sha256=sha(P/'checkpoint_identity.json'),models=[],checkpoints=[])
    def save():out.update(cpu=time.process_time()-t,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);dest.write_text(json.dumps(out,indent=2)+'\n')
    save()
    try:
        p=json.loads((P/'protocol.json').read_text())
        for mapping in (p['source_pins'],p['frozen_pins']):
            for path,h in mapping.items():assert sha(path)==h,path
        meta=json.loads((P/'checkpoint_identity.json').read_text());assert meta['complete'] and meta['protocol_sha256']==out['protocol_sha256']
        assert meta['source_sha256']==sha(P/'fetch.py')
        tree=function_tree(P/'frozen/noise.py');other=function_tree(Path(p['tracer_root'])/'noise_snapshot.py')
        dump=lambda x:ast.dump(x,include_attributes=False)
        assert dump(tree)==dump(other)
        out['noise_function_ast_sha256']=hashlib.sha256(dump(tree).encode()).hexdigest()
        noise=types.ModuleType('frozen_h5_noise');noise.np=np;exec(compile(tree,str(P/'frozen/noise.py')+' [noisy_copies only]','exec'),noise.__dict__)
        saved=json.loads((P/'frozen/ev_54.json').read_text());assert set(saved)==set(map(str,range(5)))
        assert [r['seed'] for r in meta['checkpoints']]==p['seeds']
        for row in meta['checkpoints']:
            seed=row['seed'];path=P/'checkpoints'/f'seed{seed}.pth';assert sha(path)==row['sha256']
            assert row['canonical_url']==p['canonical_urls'][seed]
            sd=torch.load(path,map_location='cpu',weights_only=True)
            expected={'rnn.weight_ih_l0':(5,1),'rnn.weight_hh_l0':(5,5),'linear.weight':(4,5)}
            assert set(sd)==set(expected)
            for k,shape in expected.items():assert tuple(sd[k].shape)==shape and sd[k].dtype==torch.float32
            arrays={k:v.detach().numpy().astype(float) for k,v in sd.items()}
            base=(arrays['rnn.weight_ih_l0'][:,0].copy(),arrays['rnn.weight_hh_l0'].copy(),arrays['linear.weight'].copy())
            out['checkpoints'].append(dict(seed=seed,sha256=row['sha256'],state_shapes={k:list(v.shape) for k,v in sd.items()},float64_hash=model_hash(base)))
            cases=[base]+noise.noisy_copies(base,4,.05,50);assert len(saved[str(seed)]['rows'])==5
            for copy,model in enumerate(cases):
                assert [a.shape for a in model]==[(5,),(5,5),(4,5)] and all(np.isfinite(a).all() for a in model)
                prior=saved[str(seed)]['rows'][copy]
                assert all(np.isfinite(prior[k]) for k in ('mc','ref','h64','h64_ops'))
                assert 0<=prior['mc']<=1 and 0<=prior['h64']<=1
                hits=round(prior['mc']*10000000);assert abs(prior['mc']*10000000-hits)<1e-7
                values=[a.tolist() for a in model];assert model_hash(tuple(np.array(v,dtype=np.float64) for v in values))==model_hash(model)
                out['models'].append(dict(seed=seed,copy=copy,model_sha256=model_hash(model),model=values,
                    saved_h64=prior['h64'],saved_h64_hex=float(prior['h64']).hex(),saved_legacy=prior['h64_ops'],
                    saved_mc=prior['mc'],mc_count_inferred_from_report=hits,mc_n=10000000,mc_rng=100+seed if copy==0 else 7,
                    approximate_h128=prior['ref']))
                save()
        assert len(out['models'])==len({r['model_sha256'] for r in out['models']})==25
        out.update(complete=True,numpy_version=np.__version__,torch_version=torch.__version__,
            scope='Canonical current object hashes/generations and original reconstruction law; historical per-model byte hashes were not retained. Original saved-output parity is a separate pilot dependency. MC counts inferred from stored ratios, not independently recounted. No forward/estimator/MC calls; CPU excludes imports.')
    except BaseException as e:out['failure']=repr(e);raise
    finally:save();print(json.dumps({k:out[k] for k in ('complete','cpu','max_rss_kib')}))
if __name__=='__main__':main()
