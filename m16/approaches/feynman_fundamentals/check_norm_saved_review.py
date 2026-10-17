import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib,math,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope/norm_review');C=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census');start=time.process_time()
def read(p):return json.loads(p.read_text())
d=read(P/'controls.json');proto=read(P/'controls_protocol.json');assert d['protocol_sha256']==hashlib.sha256((P/'controls_protocol.json').read_bytes()).hexdigest();assert len(d['rows'])==192 and d['failures']==0 and d['longdouble_mantissa']==63
keys={(r['n'],r['stride'],r['pattern']) for r in d['rows']};assert keys=={(n,s,p) for n in proto['lengths'] for s in proto['strides'] for p in proto['patterns']}
for r in d['rows']:
 n=r['n'];a,b=float.fromhex(r['actual_hex']),float.fromhex(r['expected_hex']);assert r['same'] and ((math.isnan(a) and math.isnan(b)) or r['actual_hex']==r['expected_hex']);assert r['count']==dict(mul=n if n>1 else 0,add=n+3 if n>1 else 0,sqrt=int(n>1),abs=int(n==1))
summary=read(P/'contraction.json');outm={}
for name,h in summary['pins'].items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==h
for mode in ('exact','recipe108'):
 group={'geometry':Counter(),'cold_once':Counter()};calls=Counter();shapes=Counter()
 for j in range(8):
  q=read(C/f'{mode}_{j}.json')
  for phase in ['native']+(['cold'] if j==0 else []):
   key='geometry' if phase=='native' else 'cold_once'
   for r in q[phase]['kernels']:
    if r['kernel']!='dnrm2':continue
    n,s,c=r['m'],r['n'],r['calls'];assert n>0 and s>0 and r['failures']==0;shapes[n,s]+=c;calls[key]+=c
    if n==1:group[key]['abs']+=c
    else:group[key].update(mul=n*c,add=(n+3)*c,sqrt=c)
 total=group['geometry']+group['cold_once'];calls['total']=calls['geometry']+calls['cold_once'];group['total']=total
 for key,c in group.items():
  z=summary['modes'][mode][key];assert c==Counter(z['categories']) and sum(c.values())==z['named_numeric_subset'] and calls[key]==z['calls']
 assert shapes==Counter({(r['n'],r['stride']):r['calls'] for r in summary['modes'][mode]['rows']});outm[mode]=dict(calls=calls['total'],numeric_subset=sum(total.values()),cold_once=sum(group['cold_once'].values()))
out=dict(saved_controls=192,modes=outm,review_cpu=time.process_time()-start,scope='Every saved length/stride/pattern control and all raw norm call shapes contracted independently, cold once. No norm/rank/model replay. Installed numeric subset uses x87 extended operations and named sqrt; historical provenance, control/memory/dispatch and parent/other-child costs remain separate.')
Path(__file__).with_name('norm_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
