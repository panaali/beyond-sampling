"""Read-only progress, with incomplete or failed chunks never marked done."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 complete=[];partial=[];failed=[]
 for seed in range(5):
  for noise in range(20,24):
   for block in range(4):
    p=HERE/('pilot.json' if (seed,noise,block)==(0,20,0) else f'panel/s{seed}_n{noise}_b{block}.json')
    if not p.exists():continue
    d=json.loads(p.read_text());r=dict(seed=seed,noise=noise,block=block,rows=len(d['rows']),cpu=d['cpu'])
    if d.get('failure'):failed.append(dict(r,failure=d['failure']))
    elif d.get('complete'):complete.append(r)
    else:partial.append(r)
 out=dict(complete_shards=len(complete),complete_noisy_models=sum(r['rows'] for r in complete),total_noisy_models=5120,completed_cpu=sum(r['cpu'] for r in complete),max_shard_cpu=max((r['cpu'] for r in complete),default=0),per_seed={str(s):sum(r['rows'] for r in complete if r['seed']==s) for s in range(5)},partial=partial,failed=failed)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
