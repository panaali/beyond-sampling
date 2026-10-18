import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib,math,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope/helper_review');C=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census');start=time.process_time()
def read(p):return json.loads(p.read_text())
def eq(a,b):return a==b or math.isnan(float.fromhex(a)) and math.isnan(float.fromhex(b))
d=read(P/'controls.json');assert not d['failures'] and d['protocol_sha256']==hashlib.sha256((P/'controls_protocol.json').read_bytes()).hexdigest();counts=Counter(r['kind'] for r in d['rows']);assert counts==dict(dlapy2=64,dlamch=20,dscal=320)
keys=set()
for r in d['rows']:
 assert r['same'];a,b=r['actual'],r['expected'];assert all(eq(x,y) for x,y in zip(a,b)) and len(a)==len(b) if isinstance(a,list) else eq(a,b)
 if r['kind']=='dlapy2':assert r['upper_named_units']==13;keys.add(('h',r['x'],r['y']))
 elif r['kind']=='dlamch':assert r['named_float_units']==0 and r['unpriced_character_control'];keys.add(('m',r['char']))
 else:assert r['upper_named_units']==r['n']+3;keys.add(('s',r['n'],r['stride'],r['alpha'],r['stress']))
assert len(keys)==404
summary=read(P/'contraction.json');norm=read(P.parent/'norm_review/contraction.json');outm={}
for name,h in summary['pins'].items():assert hashlib.sha256((C/name).read_bytes()).hexdigest()==h
for mode in ('exact','recipe108'):
 groups={k:Counter() for k in ('geometry','cold_once')};coldprev=None
 for j in range(8):
  q=read(C/f'{mode}_{j}.json');cold=Counter()
  for phase in ('native','cold'):
   for r in q[phase]['kernels']:
    if r['kernel']!='dlarfg' or r['m']<=1:continue
    assert r['n']>0 and not r['query'] and r['failures']==0
    dest=groups['geometry'] if phase=='native' else cold;dest[r['m'],r['n']]+=r['calls']
  if j==0:groups['cold_once']=cold
  else:assert cold==groups['cold_once']
 groups['total']=groups['geometry']+groups['cold_once']
 for key,rows in groups.items():
  a=sum(rows.values());s=sum((n-1)*v for (n,stride),v in rows.items());z=summary['modes'][mode][key];assert (a,s)==(z['active_reflectors'],z['scale_length_sum'])
  for label,hyp,sc in [('conditional_no_rescale',1,1),('generic_up_to20_rescales',2,21)]:
   v=z[label];assert v['dlapy2_upper']==13*hyp*a and v['dscal_upper']==sc*(s+3*a);assert v['helper_named_numeric_upper']==v['dlapy2_upper']+v['dscal_upper'];assert v['direct_dlamch_calls_upper']==2*a and v['internal_dlamch_calls_upper']==hyp*a
 a=summary['modes'][mode]['total']['active_reflectors'];z=summary['modes'][mode]['composed_named_subset'];n=norm['modes'][mode]['total']['named_numeric_subset'];assert z['norm_observed']==n
 for label,mult,sub in [('conditional',10,'conditional_no_rescale'),('generic',113,'generic_up_to20_rescales')]:
  h=summary['modes'][mode]['total'][sub]['helper_named_numeric_upper'];assert z[label+'_reflector_exclusive']==mult*a and z[label+'_helpers']==h and z[label+'_total_upper']==n+mult*a+h
 outm[mode]=z
out=dict(controls=dict(counts),modes=outm,review_cpu=time.process_time()-start,scope='All saved parity/control coverage and raw reflector call contractions checked independently, cold once. Conditional historical no-rescale/binding premise retained; generic2hypot/21scale is a source upper. Named subset excludes Q5, remaining rank work, character/control/memory and hardware costs. No native/model rerun.')
Path(__file__).with_name('helper_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
