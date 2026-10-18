"""Secondary independent saved coverage/aggregate review, no author imports."""
from pathlib import Path
import json,hashlib,math,time
from fractions import Fraction as F
import numpy as np
P=Path('/workspace/work/recover-small_h4_4/review/split_attribution_review');D=P/'frozen'
O=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def eq(a,b):assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=2e-13),(a,b)
start=time.process_time();r=load(P/'saved_recovery_results.json');g=load(P/'saved_recovery_gate.json')
assert r['gate_sha256']==sha(P/'saved_recovery_gate.json')
for p,h in g['pins'].items():assert sha(P/p)==h,p
manifest=load(P/'manifest.json')['records'];assert len(manifest)==121
missing=[x for x in manifest if x.get('missing')];assert len(missing)==2
for x in manifest:
 if not x.get('missing'):assert sha(P.parents[1]/x['frozen'])==x['sha256']
pool={3:[],4:[]};guard_total=[0,0,0];names=set()
for T in [3,4]:
 for s in range(5):
  name=f'H8_T{T}_s{s}';names.add(name)
  with np.load(D/'out'/f'{name}.npz',allow_pickle=False) as z,np.load(D/'out'/f'{name}_guard.npz',allow_pickle=False) as q:
   meta=json.loads(str(z['meta']));other=json.loads(str(q['meta']));n=len(z['t']);assert n==meta['resolved']==len(q['t'])
   ref=load(D/'ref'/f'{name}.json')
   for k in ['acc','ops','cells','resolved']:assert meta[k]==other[k]==ref[k]
   for k in ['t','k','frozen','sigma','truth','nrays','perm','a','con','mag']:assert np.array_equal(z[k],q[k])
   # absdE is the sum of absolute piece contributions, not abs(net dE).
   assert np.all(z['absdE']+1e-14>=np.abs(z['dE']))
   driver=np.argmax(z['con'],axis=1);rank=np.array([list(v).index(int(j))+1 for v,j in zip(z['perm'],driver,strict=True)])
   a=r['analysis'][name];assert a['n_splits']==n;eq(a['abs_mass'],math.fsum(z['absdE']));eq(a['net_mass'],math.fsum(z['dE']))
   for v in np.unique(rank):
    mask=rank==v;b=a['by']['rank'][str(v)];assert b['n']==sum(mask);eq(b['absdE'],math.fsum(z['absdE'][mask]))
   gs=q['guard'];counts=[int(sum(gs[:,2]>1)),int(sum(gs[:,3]>1)),int(sum((gs[:,2]>1)&(gs[:,3]<=1)))]
   for k,v in zip(['cum_above_one','individual_above_one','cumulative_only_above_one'],counts):assert r['guards'][name][k]==v
   guard_total=[a+b for a,b in zip(guard_total,counts)]
   pool[T].extend(zip(rank.tolist(),z['absdE'].tolist(),z['m_cell'].tolist()))
assert set(r['analysis'])==names
for T,rows in pool.items():
 a=r['pooled'][str(T)];n=len(rows);mass=math.fsum(x[1] for x in rows);assert a['n']==n;eq(a['absdE'],mass)
 thin=[x for x in rows if 0<=x[2]<1e-4]
 for k,v in {'rank4_effect_share':math.fsum(x[1] for x in rows if x[0]>=4)/mass,'thin_split_share':len(thin)/n,'thin_effect_share':math.fsum(x[1] for x in thin)/mass,'own_effect_below_1e6':sum(x[1]<1e-6 for x in rows)/n}.items():eq(a[k],v)
seen=set()
for item in r['pruning']:
 a=item['record'];assert a==load(D/'prune'/item['file']);key=(a['T'],a['seed'],a['test'],a['tau']);assert key not in seen;seen.add(key)
 assert a['flops']==2*a['ops'];T=a['T'];cost=F(6000*T*160,5760);N=F(a['flops'])/cost
 assert F(item['continuous_budget_exact'])==N
 assert sum(x[0] for x in a['by_step'].values())==a['pruned'];eq(math.fsum(x[1] for x in a['by_step'].values()),a['bound'])
 if T==3:
  truth=F(load(D/'ref/exact_H8_T3.json')[str(a['seed'])]);variance=truth*(1-truth)/N
  eq(item['point_squared_error_ratio'],(F(a['acc'])-truth)**2/variance);eq(item['reported_B_squared_ratio'],F(a['bound'])**2/variance)
assert len(seen)==52
progress=[json.loads(l) for l in (P/'saved_recovery_progress.jsonl').read_text().splitlines()];assert len(progress)==12 and progress[-1]['result_sha256']==sha(P/'saved_recovery_results.json')
result={'complete':True,'author_commit':'86549889','source_hash':sha(P/'saved_audit.py'),'result_hash':sha(P/'saved_recovery_results.json'),'manifest_payloads':119,'explicit_missing_records':missing,'archives':20,'models':10,'pruning':52,'guard_counts':guard_total,'pooled':r['pooled'],'review_cpu_seconds':time.process_time()-start,'scope':'Secondary independent raw key aggregates, paired archive fields, provenance, all pruning budget/by-step contractions. Prior source review covers full author analysis-field checker; no independent geometry, effect semantics, mass upper bound, truth or total cost certificate.'}
(O/'t198_recovery_saved_review.json').write_text(json.dumps(result,indent=2)+'\n')
gate={'source_commit':'51ee90d8','source_sha256':sha(P/'prune_tables.py'),'dependency_result_sha256':sha(P/'saved_recovery_results.json'),'source_gate':'pass once <=20CPU review lock after pinning this completed secondary review','coverage':'51 nonzero-tau point-reference rows and10 pooled report lines; zero-tau remains in primary52','qualifications':['Historical author sampler allowance500/666.6666667, not canonicalT196511/682.','Numerical reference points and reported bound values only; source cap already has counterexample.','Not fullcost, actual error upper bounds or uncertainty certification.']}
(O/'t198_prune_tables_source_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({'complete':True,'models':10,'archives':20,'pruning':52,'guard_counts':guard_total}))
