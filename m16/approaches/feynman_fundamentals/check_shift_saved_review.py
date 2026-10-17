import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time,itertools
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope/shift_review');C=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census');start=time.process_time()
def read(p):return json.loads(p.read_text())
d=read(P/'controls.json');assert d['cases']==1152 and d['parity_failures']==d['bound_failures']==d['z_failures']==0
for name,h in d['pins'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
keys=set();types=set()
for r in d['rows']:
 assert r['parity'] and r['within'] and r['z_unchanged'];assert r['units']==sum(r['reference']['counts'].values())<=r['upper']=={3:35,4:44}[r['length']]
 for k,v in r['actual'].items():assert v==r['reference'][k] and (k=='ttype' or float(v).hex()==float(r['reference'][k]).hex())
 keys.add(tuple(r[k] for k in ('length','pp','deflated','config_id','pattern','oldtype')));types.add(r['actual']['ttype'])
assert keys==set(itertools.product((3,4),(0,1),range(4),range(8),('flat','qd','ramp'),(-6,-18,0))) and types==set(range(-12,0))
s=read(P/'contraction.json');outm={}
for name,h in s['pins'].items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==h
for mode in ('exact','recipe108'):
 groups={'geometry':Counter(),'cold_once':Counter()}
 for j in range(8):
  q=read(C/f'{mode}_{j}.json');cold=Counter()
  for phase in ('native','cold'):
   assert not q[phase]['overflow']
   for r in q[phase]['kernels']:
    if r['kernel']!='dlasq3':continue
    assert r['m'] in (1,2,3,4) and not r['query'] and not r['failures'];dest=groups['geometry'] if phase=='native' else cold;dest[r['m'],r['n']]+=r['calls']
  if not j:groups['cold_once']=cold
  else:assert cold==groups['cold_once']
 groups['total']=groups['geometry']+groups['cold_once']
 for key,c in groups.items():
  z=s['modes'][mode][key];assert sum(c.values())==z['q3_calls'];assert sum(v for (n,p),v in c.items() if n>=3)==z['q4_calls_upper'];assert sum({1:0,2:0,3:35,4:44}[n]*v for (n,p),v in c.items())==z['source_named_numeric_upper'];assert c==Counter({(r['entry_length'],r['pp']):r['q3_calls'] for r in z['rows']})
 outm[mode]=s['modes'][mode]['total']['source_named_numeric_upper']
out=dict(controls=1152,types=sorted(types),source_upper=outm,review_cpu=time.process_time()-start,scope='Every saved native/source parity and counted ledger plus all raw Q3 entry histograms reconstructed, cold once. Pinned source Q4 upper35/44; finite parity is not all-input compiled equivalence. Q3/Q5 and remaining rank/control/memory costs excluded.')
Path(__file__).with_name('shift_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
