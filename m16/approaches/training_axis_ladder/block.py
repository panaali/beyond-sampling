"""One fixed seed/noise block; raw rows checkpointed in schema order."""
import argparse,json,resource,time,traceback
from common import *
SCHEMA=['copy','truth','estimate','candidate_ops','reference_ops','correct_leaves','cut','resolved','cells','candidate_cpu','reference_cpu','model_sha256']

def save(path,out):
    metadata={k:v for k,v in out.items() if k!='rows'}
    text=json.dumps(metadata,separators=(',',':'))[:-1]+',"rows":[\n'+',\n'.join(json.dumps(r,separators=(',',':')) for r in out['rows'])+'\n]}\n'
    tmp=path.with_suffix('.tmp');tmp.write_text(text);tmp.replace(path)

def run(seed,noise):
    start=time.process_time();path=HERE/f'block_s{seed}_n{noise}.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('complete') and len(saved['rows'])==256:
            print('already complete',seed,noise,flush=True);return
        raise RuntimeError(f'partial block retained at {path}; inspect before resuming')
    base=load(29,seed);out=dict(power=29,training_seed=seed,noise_seed=noise,sigma=.05,n=256,schema=SCHEMA,base_sha256=model_hash(base),recipe=E75.RECIPE,rows=[],complete=False)
    try:
        for i,model in enumerate(noisy_copies(base,256,.05,noise)):
            candidate=estimate(model);exact=truth(model);st=candidate['stats'];p=exact['value']
            assert -1e-12<=p<=1+1e-12 and -1e-12<=candidate['value']<=1+1e-12
            out['rows'].append([i,p,candidate['value'],candidate['operations'],exact['operations'],exact['correct_leaves'],st['cut'],st['resolved'],st['cells'],candidate['cpu_seconds'],exact['cpu_seconds'],model_hash(model)])
            out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if (i+1)%32==0:save(path,out)
            if out['cpu_seconds']>100:raise RuntimeError('bounded block CPU threshold reached; partial retained')
        out['complete']=True;save(path,out)
        rows=np.asarray([r[:11] for r in out['rows']],float);e=rows[:,2]-rows[:,1];ratio=np.mean(e*e)*np.mean(rows[:,3])/(72*np.mean(rows[:,1]*(1-rows[:,1])))
        print(seed,noise,'ratio',ratio,'cpu',out['cpu_seconds'],flush=True)
    except Exception as exc:
        out['failure']=dict(exception=repr(exc),traceback=traceback.format_exc());save(path,out);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training-seed',type=int,required=True,choices=range(5));parser.add_argument('--noise-seed',type=int,choices=(20,21,22,23));args=parser.parse_args()
    for ns in ((args.noise_seed,) if args.noise_seed is not None else (20,21,22,23)):run(args.training_seed,ns)
