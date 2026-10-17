import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope/binding_review');C=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census');start=time.process_time()
def read(p):return json.loads(p.read_text())
r=read(P/'results.json');direct=read(P/'controls_direct.json')['rows'];obs=read(P/'controls_observed.json')['rows'];assert len(direct)==len(obs)==12
for path,h in r['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
for a,b in zip(direct,obs):
 assert a=={k:v for k,v in b.items() if k!='records'};parents=sum(x[4] for x in b['records'] if x[0]==8);norm=sum(x[4] for x in b['records'] if x[0]==9);assert parents==1 and norm==(0 if a['n']==1 else (2 if a['alpha']==1e-310 and a['tiny'] else 1))
saved={(x['mode'],x['batch'],x['phase']):x for x in r['saved_phases']};assert len(saved)==32;tot={};contract=read(P/'contraction.json')
for mode in ('exact','recipe108'):
 active=norm=0
 for i in range(8):
  d=read(C/f'{mode}_{i}.json')
  for phase in ('cold','native'):
   q=d[phase];assert q['overflow']==0 and all(x['failures']==0 and x['calls']>0 for x in q['kernels']);a=sum(x['calls'] for x in q['kernels'] if x['kernel']=='dlarfg' and x['m']>1);n=sum(x['calls'] for x in q['kernels'] if x['kernel']=='dnrm2');assert a==n==saved[mode,i,phase]['active_reflectors']==saved[mode,i,phase]['norm_calls']
   if phase=='native' or i==0:active+=a;norm+=n
 z=contract['modes'][mode];assert active==z['active_reflectors']==z['dnrm2_calls'] and active*10==z['conditional_finite_exclusive_10'] and active*113==z['generic_exclusive_113'];tot[mode]=dict(active=active,conditional_exclusive=active*10,generic_exclusive=active*113)
text=(P/'dlarfg.disasm.txt').read_text();assert 'd3e8ff:' in text and 'd3eb01:' in text;assert all('86030' in line for line in text.splitlines() if 'd3e8ff:' in line or 'd3eb01:' in line)
out=dict(controls=12,complete_saved_phases=32,totals=tot,review_cpu=time.process_time()-start,scope='Installed norm callsites, finite passthrough controls and all saved phase equalities checked. No-rescale deduction requires historical complete observer/binding provenance; current binding trace cannot retroactively prove it. Generic113 retained; child costs and full native cost remain open.')
Path(__file__).with_name('reflector_binding_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
