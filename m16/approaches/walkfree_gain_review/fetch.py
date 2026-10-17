"""Fetch explicit public T2 checkpoints; no gain table or feature computation."""
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path

ROOT=Path('/workspace/work/t45_weights')
OUT=Path(__file__).with_name('checkpoint_manifest.json')
BASE='https://storage.googleapis.com/arc-ml-public/alg/zoo'
KEYS=[(h,t,s) for h in [8,16,32] for t in [3,4,5,6,8,10] for s in range(5)]
ROOT.mkdir(parents=True,exist_ok=True)
start=time.monotonic()


def fetch(key):
    h,t,s=key;name=f'2nd_argmax_{h}_{t}_1073741824_{s}.pth';url=f'{BASE}/{name}';path=ROOT/name
    row=dict(H=h,T=t,seed=s,n_seqs=2**30,url=url,path=str(path),read_date='2026-10-17')
    try:
        if path.exists():data=path.read_bytes();row['cached']=True
        else:
            with urllib.request.urlopen(url,timeout=20) as response:data=response.read(2*1024*1024+1)
            if len(data)>2*1024*1024:raise RuntimeError('Checkpoint exceeds frozen2MiB bound')
            path.write_bytes(data);row['cached']=False
        row.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),status='available')
    except Exception as exc:row.update(status='unavailable',error=str(exc))
    return row


rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for row in pool.map(fetch,KEYS):
        rows.append(row)
        if len(rows)%15==0:print(f'Checked {len(rows)}/90 public checkpoints',flush=True)
assert sum(r.get('bytes',0) for r in rows)<16*1024*1024
out=dict(models=rows,total_bytes=sum(r.get('bytes',0) for r in rows),available=sum(r['status']=='available' for r in rows),
         wall_seconds=time.monotonic()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         note='Public explicit-seed model weights only; no gain table, feature evaluation or external code executed.')
OUT.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='models'},indent=2))
