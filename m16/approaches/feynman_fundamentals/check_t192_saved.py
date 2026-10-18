"""Independent scalar saved-record review; no author imports or numerical replay."""
import ast, hashlib, json, math, time
from pathlib import Path
from fractions import Fraction as F

ROOT=Path('/workspace/work/feynman_fundamentals')
D=ROOT/'work/t192_readonly/m16/approaches/gpu_cost_law'
SH=Path('/workspace/work/reviewer-small-publication')
OUT=ROOT/'m16/approaches/feynman_fundamentals/t192_saved_review.json'
def read(p):return json.loads(p.read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=3e-13,abs_tol=3e-14),(a,b)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time()
pins={str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()}
runs={};coverage=[]
for p in sorted((D/'results/runs').glob('*.json')):
 r=read(p); raw=[json.loads(l) for l in p.with_suffix('.jsonl').read_text().splitlines()]
 ids=[x['half'] for x in raw];assert len(ids)==len(set(ids))
 assert set(ids)==set(range(r['halves'])),p.name
 assert r.get('halves_run',r['halves'])==r['halves']
 for key in ['ops','simplices','cells','cut','resolved','ops_geometry','thin','sec_geometry']:
  if key in r:
   assert all(key in x for x in raw),(p.name,key)
   close(r[key],math.fsum(x[key] for x in raw))
 close(r['acc'],1-math.fsum(x['err'] for x in raw))
 checked_mass=0
 for x in raw:
  if 'ops_geometry' in x and 'thin' in x:
   n,thin=x['simplices'],x['thin'];assert 0<=thin<=n
   if r['T']==4:expected=thin*(144+1000*30)+(n-thin)*(144+440+128*234)
   elif r['T']==5:expected=thin*(125+1296*40)+(n-thin)*(125+1175+240+128*1340)
   else:raise AssertionError(r['T'])
   assert x['ops']-x['ops_geometry']==expected,(p.name,x['half'])
   checked_mass+=1
 key=(r['H'],r['T'],r['model'].replace(':','_'),r['mode']);runs[key]=r
 coverage.append({'file':p.name,'halves':len(raw),'mass_proxy_half_checks':checked_mass,'regions_saved':'regions' in r})
audits={};audit_rows=[]
for p in sorted((D/'results/audit').glob('*.json')):
 r=read(p);s=r['sites'];close(r['flops_complete'],math.fsum(v['full'] for v in s.values()))
 # Cold Ops instances are attributed but not included in the main estimator Ops object.
 charged=math.fsum(v['charged'] for v in s.values())
 close(r['ratio_complete_over_charged'],r['flops_complete']/r['ops_charged'])
 ex=runs[(r['H'],r['T'],r['model'],r['mode'])];assert ex['ops']==r['ops_charged']
 audit_rows.append({'file':p.name,'ratio_declared_model':r['ratio_complete_over_charged'],'cold_attribution_delta':charged-r['ops_charged'],'accuracy_delta_vs_gpu':r['acc']-ex['acc']})
 audits[(r['H'],r['T'],r['model'],r['mode'])]=r
mc={};mcrows=[]
for folder in ['mc','mc2']:
 for p in sorted((D/'results'/folder).glob('*.json')):
  r=read(p);assert type(r['correct']) is int and type(r['n']) is int and 0<=r['correct']<=r['n']
  assert r['acc']==r['correct']/r['n'];close(r['se'],math.sqrt(r['acc']*(1-r['acc'])/r['n']))
  key=(r['H'],r['T'],r['model'].replace(':','_'));ex=runs.get((*key,'exact'))
  row={'file':str(p.relative_to(D)),'seed':r['seed'],'n':r['n'],'has_exact':ex is not None}
  if ex:row['z_mc_minus_exact']=(r['acc']-ex['acc'])/r['se']
  mcrows.append(row);mc[(folder,*key)]=r
summary=[]
for r in read(D/'results/summary.json'):
 key=(r['H'],r['T'],r['model']);ex=runs[(*key,'exact')];assert r['exact']==ex['acc']
 rec=runs.get((*key,'recipe108'))
 if rec:
  e=F(rec['acc'])-F(ex['acc']);v=F(ex['acc'])*(1-F(ex['acc']))
  H,T=key[:2];fpe=F(6000*(T*(2*H*H+2*H)+2*H*T),10*(2*16*16+2*16)+2*16*10)
  close(r['mse_ratio_ops'],float(e*e*F(rec['ops'])/(v*fpe)))
  a=audits.get((*key,'recipe108'));cost=a['flops_complete'] if a else 1.37*rec['ops']
  assert r['flops_measured']==bool(a);close(r['recipe_flops'],cost)
  close(r['mse_ratio_flops'],float(e*e*F(cost)/(v*({4:682,5:853}[T]))))
  summary.append({'key':list(key),'legacy_ratio':r['mse_ratio_ops'],'declared_T196_ratio':r['mse_ratio_flops'],'uses_transferred_1_37':not bool(a)})
 if 'mc' in r:
  m=mc[('mc',*key)];assert r['mc']==m['acc'];close(r['z_mc'],(m['acc']-ex['acc'])/m['se'])
parity=[]
for seed in range(5):
 old=read(SH/f'm16/approaches/estT_fix/rescore/zoo{seed}.json')
 for mode in ['exact','recipe108']:
  cur=runs[(8,4,str(seed),mode)]
  for k in ['ops','cells','resolved']:assert cur[k]==old[mode][k]
  parity.append({'T':4,'seed':seed,'mode':mode,'delta':cur['acc']-old[mode]['acc']})
for mode in ['exact','recipe108']:
 old=read(SH/f'm16/approaches/estT_fix/run5/zoo0_{mode}.json');cur=runs[(8,5,'0',mode)]
 for k in ['ops','cells','resolved','cut']:assert cur[k]==old[k]
 parity.append({'T':5,'seed':0,'mode':mode,'delta':cur['acc']-old['acc'],'simplices_gpu':cur['simplices'],'simplices_historical':old['simplices'],'simplices_equal':cur['simplices']==old['simplices']})
base_same=(D/'estT_base.py').read_bytes()==(SH/'m16/approaches/estT_fix/estT.py').read_bytes()
result={'complete':True,'source_commit':'9d2feb68de935b41299357dbdc060ce5548a887e','pins':pins,'run_coverage':coverage,'audits':audit_rows,'mc':mcrows,'summary':summary,'historical_saved_parity':parity,'base_source_byte_equal_shared':base_same,'review_cpu_seconds':time.process_time()-start,'scope':'Saved arithmetic, source inspection and finite historical parity only. No numerical replay, no complete-cost bound, no geometry correctness proof or MC stream independence certification.'}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:len(result[k]) for k in ['run_coverage','audits','mc','summary','historical_saved_parity']}))
