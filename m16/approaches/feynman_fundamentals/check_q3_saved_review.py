import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope/shift_review');C=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census');start=time.process_time()
def read(p):return json.loads(p.read_text())
maxq=0;total=0
for name,expected in [('parent_controls.json',864),('retry_controls.json',16)]:
 d=read(P/name);assert d['cases']==expected and len(d['rows'])==expected and d['failures']==0
 for f,h in d['pins'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h
 for r in d['rows']:
  assert r['passed'] and r['parity'] and r['within'];a,b=r['reference'],r['native'];assert sum(a['counts'].values())<=a['upper']=={1:3,2:25,3:51,4:73}[r['length']]+16*a['children'].get('q5',0)
  for k in ['n0','pp','nfail','iteration','ndiv','ttype']:assert a[k]==b[k]
  for k in ['z','values']:assert len(a[k])==len(b[k]) and all(float(x).hex()==float(y).hex() for x,y in zip(a[k],b[k]))
  maxq=max(maxq,a['children'].get('q5',0));total+=1
assert total==880 and maxq==3
s=read(P/'parent_contraction.json');outm={}
for name,h in s['pins'].items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==h
for mode in ('exact','recipe108'):
 groups={'geometry':Counter(),'cold_once':Counter()}
 for j in range(8):
  q=read(C/f'{mode}_{j}.json');cold=Counter()
  for phase in ('native','cold'):
   assert not q[phase]['overflow']
   for r in q[phase]['kernels']:
    if r['kernel'] not in ('dlasq3','dlasq5','dlasq6'):continue
    assert not r['query'] and not r['failures'];dest=groups['geometry'] if phase=='native' else cold;dest[r['kernel'],r['m'],r['n']]+=r['calls']
  if not j:groups['cold_once']=cold
  else:assert cold==groups['cold_once']
 groups['total']=groups['geometry']+groups['cold_once']
 for key,c in groups.items():
  calls=Counter();base=0
  for (k,n,p),v in c.items():
   calls[k]+=v
   if k=='dlasq3':base+={1:3,2:25,3:51,4:73}[n]*v
  z=s['modes'][mode][key];assert z['q3_calls']==calls['dlasq3'] and z['q5_actual_calls']==calls['dlasq5'] and z['q6_actual_calls']==calls['dlasq6']==0;assert z['parent_base_upper']==base and z['parent_retry_upper']==16*calls['dlasq5'] and z['parent_source_upper']==base+16*calls['dlasq5']
 outm[mode]=s['modes'][mode]['total']['parent_source_upper']
out=dict(controls=880,max_q5_retries_observed=maxq,parent_source_upper=outm,review_cpu=time.process_time()-start,scope='All saved parent/native outputs and counted upper bounds, raw finite Q3/Q5/Q6 census and cold-once contractions checked. Shared native children test parent equivalence only; original864 no-retry limitation and16 added controls preserved. Pinned-source upper excludes child pricing, remaining rank/control/memory and all-input binary proof.')
Path(__file__).with_name('q3_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
