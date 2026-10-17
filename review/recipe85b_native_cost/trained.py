"""Two fixed existing T115 cases; execute only after source/control gate."""
import time
START=time.process_time()
import hashlib,json,sys,traceback,resource
from pathlib import Path
import numpy as np
import torch
import native as N
HERE=Path(__file__).resolve().parent
TRAIN=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery')

def main(seed):
 assert seed in (1,4);noise,copy=(20,0) if seed==1 else (23,255)
 pins=json.loads((HERE/'inventory.json').read_text())['pins']
 source=TRAIN/f'k30_s{seed}_noise{noise}.json';manifest=TRAIN/'continuation_manifest.json'
 for path in (source,manifest):assert hashlib.sha256(path.read_bytes()).hexdigest()==pins[str(path)]
 row=json.loads(source.read_text())['rows'][copy];record=json.loads(manifest.read_text())[f'checkpoint_{seed}'];path=Path(record['source']);assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
 sd=torch.load(path,weights_only=True,map_location='cpu');base=(sd['rnn.weight_ih_l0'].double().numpy()[:,0],sd['rnn.weight_hh_l0'].double().numpy(),sd['linear.weight'].double().numpy());rng=np.random.default_rng(noise)
 for _ in range(copy+1):model=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
 sha=hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest();assert sha==row['model_hash']
 plain=N.evaluate(model,False);traced=N.evaluate(model,True)
 for key in ('hex','legacy','stats','info'):assert plain[key]==traced[key],key
 saved=row['recipes']['r85b'];assert plain['value']==saved['value'] and plain['legacy']==saved['native_ops'] and plain['stats']['cells']==saved['cells']
 out=dict(seed=seed,noise=noise,copy=copy,model_sha256=sha,checkpoint_sha256=record['sha256'],saved=row,baseline=plain,traced=traced,pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,HERE/'native.py',Path(__file__),N.N.SOURCE)},sampler_numeric=111,cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 target=HERE/f'trained_seed{seed}.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(seed=seed,value=plain['value'],legacy=plain['legacy'],native=traced['ledger']['named_numeric_total'],sort_calls=traced['ledger']['sort_dispatch_calls'],cpu=out['cpu'])))
if __name__=='__main__':
 seed=int(sys.argv[1])
 try:main(seed)
 except Exception as exc:
  (HERE/f'trained_seed{seed}_failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
