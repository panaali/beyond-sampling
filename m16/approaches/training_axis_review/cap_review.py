"""Non-mutating-output replay with explicit trace-loop exhaustion telemetry."""
import builtins,json,time
from pathlib import Path
import loaders as L
HERE=Path(__file__).parent
COUNTS=dict(calls=0,exhaustions=0)
def monitored_range(*args):
 if args!=(64,):return builtins.range(*args)
 COUNTS['calls']+=1
 def iterate():
  for i in builtins.range(64):yield i
  COUNTS['exhaustions']+=1
 return iterate()
L.FIB.range=monitored_range
start=time.process_time();out=dict(rows=[],complete=False)
def check(m,original,key):
 before=COUNTS.copy();v=L.fibre(m)
 row=dict(key=key,calls=COUNTS['calls']-before['calls'],exhaustions=COUNTS['exhaustions']-before['exhaustions'],value_change=v['value']-original['value'],cost_change=v['strict_ops']-original['strict_ops'],traces=v['traces'])
 out['rows'].append(row)
 assert row['value_change']==0 and row['cost_change']==0
for path in sorted(HERE.glob('*_k*_s*.json')):
 d=json.loads(path.read_text());assert d['complete'];base=L.load(d['k'],d['training_seed'])
 check(base,d['clean']['fibre'],path.name+':clean')
 for ns in d['noise_seeds']:
  models=L.noisy(base,ns,8,d['sigma'])
  for i,m in enumerate(models):
   original=next(r for r in d['rows'] if r['noise_seed']==ns and r['index']==i)
   check(m,original['fibre'],f'{path.name}:{ns}:{i}')
 out['cpu']=time.process_time()-start
 (HERE/'cap_results.json').write_text(json.dumps(out,indent=2)+'\n')
 print(path.name,len(out['rows']),'rows, exhaustion',COUNTS['exhaustions'],flush=True)
out.update(complete=True,cpu=time.process_time()-start,counts=COUNTS)
(HERE/'cap_results.json').write_text(json.dumps(out,indent=2)+'\n')
