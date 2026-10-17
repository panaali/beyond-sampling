"""Fixed existing H8 core native-source cost shard, preserving every site ledger."""
import time
START=time.process_time()
import gzip,hashlib,json,resource,sys,traceback
from pathlib import Path
import numpy as np
import torch
import trace as T
HERE=Path(__file__).resolve().parent

def main(zoo,noise):
 assert 0<=zoo<5 and noise in (80,81,82,83)
 folder=HERE/'panel';folder.mkdir(exist_ok=True);stem=f'H8_z{zoo}_n{noise}';output=folder/f'{stem}.json';archive=folder/f'{stem}.jsonl.gz'
 assert not output.exists() and not archive.exists(),'Preserve existing complete/partial shard'
 source=HERE.parent/'recipe112_recovery/direct_counts'/f'core_{stem}.json';old=json.loads(source.read_text());assert old['complete'] and len(old['rows'])==64
 path=Path('/workspace/work/t45_weights')/f'2nd_argmax_8_3_1073741824_{zoo}.pth';assert hashlib.sha256(path.read_bytes()).hexdigest()==old['checkpoint_sha256']
 s=torch.load(path,map_location='cpu',weights_only=True)
 base=(s['rnn.weight_ih_l0'].numpy().astype(float)[:,0],s['rnn.weight_hh_l0'].numpy().astype(float),s['linear.weight'].numpy().astype(float));rng=np.random.default_rng(noise)
 out=dict(zoo=zoo,noise=noise,H=8,complete=False,rows=[],checkpoint_sha256=old['checkpoint_sha256'],prior_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(T.SOURCE.read_bytes()).hexdigest(),trace_sha256=hashlib.sha256((HERE/'trace.py').read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sampler_numeric=351,archive=archive.name)
 def save():
  out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);output.write_text(json.dumps(out,indent=2)+'\n')
 with gzip.open(archive,'wt',compresslevel=6) as log:
  for i,saved in enumerate(old['rows']):
   model=tuple(w+.05*rng.standard_normal(w.shape) for w in base);sha=hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest();assert sha==saved['model_sha256']
   row=dict(copy=i,model_sha256=sha,truth=saved['truth'],saved_estimate=saved['saved_estimate'],saved_scalar_ops=saved['ops']);out['rows'].append(row)
   try:
    baseline=T.evaluate(model,False);traced=T.evaluate(model,True)
    for key in ('hex','legacy','stats','info'):assert baseline[key]==traced[key],(i,key)
    assert baseline['legacy']==saved['saved_ops'],(i,'legacy')
    assert baseline['stats']['cells']==saved['saved_cells'],(i,'cells')
    assert abs(baseline['value']-saved['saved_estimate'])<=1e-12,(i,'saved output')
    ledger=traced['ledger'];row.update(estimate=baseline['value'],ops=ledger['named_numeric_total'],integer=ledger['integer_control_total'],legacy=baseline['legacy'],cells=baseline['stats']['cells'],parity=True,totals=ledger['totals'])
    log.write(json.dumps(dict(copy=i,model_sha256=sha,baseline=baseline,traced=traced))+'\n');log.flush()
   except Exception as exc:
    row.update(failure=repr(exc),traceback=traceback.format_exc());save();raise
   if (i+1)%8==0:save()
   if time.process_time()-START>50:out['budget_stop']=True;save();return
 out.update(complete=True,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest());save()
 print(json.dumps(dict(shard=stem,complete=True,rows=len(out['rows']),numeric_total=sum(r['ops'] for r in out['rows']),cpu=out['cpu'])),flush=True)
if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))
