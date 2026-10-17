"""Fixed paired kernel timings; no accuracy truth or fitted CV coefficients."""
import argparse,hashlib,importlib.util,json,resource,time
from pathlib import Path
import numpy as np
from native import walk
HERE=Path(__file__).parent;ROOT=Path('/workspace/clean-neurons')
spec=importlib.util.spec_from_file_location('legacy_great_circle',ROOT/'m16/approaches/great_circle/gcirc.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)

def load(path):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True)
    arrays=[state[k].detach().numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')]
    return (arrays[0][:,0],arrays[1],arrays[2]),hashlib.sha256(path.read_bytes()).hexdigest()

def run(pilot):
    start=time.process_time();n=4 if pilot else 32;g=np.random.default_rng(660001).standard_normal((n,10,2));q,r=np.linalg.qr(g);q*=np.sign(np.diagonal(r,axis1=1,axis2=2))[:,None,:];u,v=q[:,:,0],q[:,:,1]
    base,sha=load(HERE.parent/'face_surplus_review/data/H16_T10_original.pth');cases=[]
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        cases.append((f'H16_sigma{sigma}',model,sha))
    h2,sha=load(ROOT/'work/checkpoints/H2_T10_seed0.pth');cases.append(('H2_T10',h2,sha))
    out=dict(planes=n,seed=660001,cases=[],preparation_cpu=time.process_time()-start)
    for name,model,sha in cases:
        row=dict(name=name,checkpoint_sha256=sha)
        tick=time.process_time();p,stats=walk(model,u,v);native=time.process_time()-tick
        row.update(native_values=p.tolist(),native_counts=stats,native_cpu=native)
        tick=time.process_time();old,pieces,partial=legacy.walk(model,u,v);python=time.process_time()-tick
        row.update(legacy_values=old.tolist(),legacy_pieces=pieces.tolist(),legacy_cpu=python,max_difference=float(np.max(abs(p-old))),speed_ratio=python/native)
        if not pilot:
            tick=time.process_time();again,againstats=walk(model,u,v);row['native_warm_cpu']=time.process_time()-tick
            tick=time.process_time();oldagain,piecesagain,_=legacy.walk(model,u,v);row['legacy_warm_cpu']=time.process_time()-tick
            row['warm_speed_ratio']=row['legacy_warm_cpu']/row['native_warm_cpu'];row['native_repeat_difference']=float(np.max(abs(p-again)))
            row['legacy_repeat_difference']=float(np.max(abs(old-oldagain)))
        out['cases'].append(row);out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (HERE/('pilot.json' if pilot else 'benchmark.json')).write_text(json.dumps(out,indent=2)+'\n')
        print(name,'difference',row['max_difference'],'speed',row['speed_ratio'],'native',native,'python',python,flush=True)
        if out['cpu_seconds']>100:raise RuntimeError('bounded phase CPU cap; partial retained')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args();run(args.pilot)
