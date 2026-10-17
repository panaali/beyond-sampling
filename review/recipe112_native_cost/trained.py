"""Exactly two preregistered source-native traces; run only after source gate."""
import time
START=time.process_time()
import hashlib,json,sys,traceback,resource
from pathlib import Path
import numpy as np
import torch
import trace as T
HERE=Path(__file__).resolve().parent

def main(H):
 assert H in (4,8)
 old=next(r for r in json.loads((HERE.parent/'recipe112_recovery/semantics.json').read_text())['models'] if r['H']==H)
 path=Path(old['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==old['checkpoint_sha256']
 s=torch.load(path,map_location='cpu',weights_only=True)
 base=(s['rnn.weight_ih_l0'].numpy().astype(float)[:,0],s['rnn.weight_hh_l0'].numpy().astype(float),s['linear.weight'].numpy().astype(float))
 rng=np.random.default_rng(89)
 for _ in range(old['copy']+1):model=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
 baseline=T.evaluate(model,False);traced=T.evaluate(model,True)
 for key in ('hex','legacy','stats','info'):assert baseline[key]==traced[key],key
 assert baseline['legacy']==old['saved']['charge']
 assert abs(baseline['value']-old['recipe'])<=1e-15
 rows=json.loads((HERE.parent/f'recipe112_recovery/counter_H{H}.json').read_text())['rows']
 scalar=next(r for r in rows if r['zoo']==old['zoo'] and r['copy']==old['copy'])
 out=dict(H=H,zoo=old['zoo'],noise=89,copy=old['copy'],checkpoint_sha256=old['checkpoint_sha256'],model_sha256=hashlib.sha256(b''.join(x.tobytes() for x in model)).hexdigest(),baseline=baseline,traced=traced,scalar_repaired_counter=scalar['repaired_counter'],source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (T.SOURCE,HERE/'trace.py',HERE/'base_snapshot.py',Path(__file__))},cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 target=HERE/f'trained_H{H}.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(H=H,value=baseline['value'],legacy=baseline['legacy'],scalar=out['scalar_repaired_counter'],native_numeric=traced['ledger']['named_numeric_total'],integer=traced['ledger']['integer_control_total'],cpu=out['cpu'])))
if __name__=='__main__':
 H=int(sys.argv[1])
 try:main(H)
 except Exception as exc:
  (HERE/f'trained_H{H}_failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
