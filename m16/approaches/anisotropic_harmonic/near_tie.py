"""Frozen H4T3 mechanism contrast with existing weights/exact reference."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from engine import estimate
HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/clean-neurons/m16/approaches/arc_m43_repro')
REVIEW=Path('/workspace/work/recover-small_h2_910/m16/approaches/harmonic_negative_review')
sys.path.insert(0,str(REVIEW))
from diagnose import evaluate
path=SOURCE/'m43_weights.npz'
arrays=np.load(path)
model=tuple(arrays[k].astype(float) for k in ('Wi','Wh','Wo'))
reference=json.loads((SOURCE/'results.json').read_text())['base']['exact']
start=time.process_time();deadline=start+18
result=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference=reference,rows=[])
for N in (8,16):
    p,s=estimate(model,N,4,seconds=deadline-time.process_time())
    row=dict(method='supported',estimate=p,error=p-reference,stats=s)
    result['rows'].append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
    (HERE/'near_tie_results.json').write_text(json.dumps(result,indent=2)+'\n')
s=evaluate(model,8,True,True,deadline)
result['rows'].append(dict(method='isotropic_corrected',D=8,estimate=s['estimate'],error=s['estimate']-reference,stats=s))
result['total_cpu_seconds']=time.process_time()-start
(HERE/'near_tie_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result['rows'][-1].items() if k!='stats'}))
