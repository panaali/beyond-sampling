import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import json,hashlib,time,math
P=Path('/workspace/work/recover-small_h2_910/review/fibre_native_completion');start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ledger(l):
 c=Counter()
 for v in l['sites'].values():c.update(v)
 assert c==Counter(l['totals']);num=sum(v for k,v in c.items() if not k.startswith('integer_control'));ints=sum(c.values())-num
 assert (num,ints,num+ints)==(l['named_numeric_total'],l['integer_control_total'],l['inclusive_explicit_total']);return c
inv=read(P/'inventory.json');prior=Path(inv['prior']);native=Path(inv['native'])
for name,h in inv['files'].items():assert sha(Path(name))==h
folder=P/'cell_rescue/12';prep=read(folder/'prepare.json');resc=read(folder/'summary.json');assert prep['cell_count']==144
prev=0.0;prevhash=None;sites={};totals=Counter();st=Counter();leg=prep['legacy_ops']
for item in [prep]:
 totals.update(ledger(item['ledger']))
 for site,c in item['ledger']['sites'].items():sites.setdefault(site,Counter()).update(c)
for k in range(144):
 p=folder/f'cell_{k:05d}.json';r=read(p)
 assert r['complete'] and r['cell']==k and r['poly']==prep['cells'][k]['poly'] and r['source_sha256']==prep['source_sha256'] and r['model_sha256']==inv['model_sha256'];assert r['initial_total'].hex()==prev.hex() and r['previous_sha256']==prevhash and r['prepare_sha256']==sha(folder/'prepare.json');assert r['total_hex']==r['total'].hex() and r['total_type'] in ('float','numpy.float64')
 prev=r['total'];prevhash=sha(p);totals.update(ledger(r['ledger']));st.update(r['stats']);leg+=r['legacy_ops']
 for site,c in r['ledger']['sites'].items():sites.setdefault(site,Counter()).update(c)
assert totals==ledger(resc['ledger']) and {k:dict(v) for k,v in sites.items()}==resc['ledger']['sites'];assert st==Counter(resc['stats']);repeat=143*16*prep['clamps'];assert repeat==resc['repeated_legacy_setup'] and leg==resc['actual_adapter_legacy_ops']==resc['legacy_ops']+repeat and prev.hex()==resc['raw_error_component'].hex()
s=read(P/'summary.json');assert s['complete'] and s['indices']==list(range(48)) and s['cell_rescued_indices']==[12] and not s['pending_indices'];aggregate=Counter();seq=0.0;exact=F(0);cpus=0;opaques=[]
for i in range(48):
 r=read(native/'component.json') if i==0 else resc if i==12 else read(P/'components'/f'{i:02d}.json');e=read(prior/'noise_preflight.json' if i==0 else prior/'noise_components'/f'{i:02d}.json');assert r['complete'] and r['model_sha256']==inv['model_sha256'];assert r['raw_error_component'].hex()==e['raw_error_component'].hex() and r['stats']==e['stats'] and r['legacy_ops']==e['legacy_ops']
 if i!=0:assert (r['index'],r['family'],r['side'])==(i,i//2,i%2)
 c=ledger(r['ledger']);numeric=sum(v for k,v in c.items() if not k.startswith('integer_control'));integer=sum(c.values())-numeric;diag=sum(sum(v.values()) for site,v in r['ledger']['sites'].items() if "stats['traces'] += 1" in site or "stats['pieces'] += 1" in site);v=dict(numeric=numeric,integer=integer,diagnostic=diag,visible_after_diagnostic=numeric+integer-diag,legacy=r['legacy_ops']);aggregate.update(v)
 for k,x in v.items():assert s['rows'][i][k]==x
 assert s['rows'][i]['opaque_calls']==r['ledger']['opaque_calls'];opaques.extend(r['ledger']['opaque_calls']);cpus+=r['cpu_seconds'];seq+=r['raw_error_component'];exact+=F(r['raw_error_component'])
for k,v in aggregate.items():assert s[k]==v
assert cpus==s['cpu'] and float(exact)==s['raw_exact_saved_float_sum'];estimate=1-seq/(2*math.pi**2);assert estimate==s['original_order_estimate']==read(prior/'noise_summary.json')['estimate'];failure=read(P/'components/12.json');assert not failure['complete'] and failure['cpu_seconds']==s['failure_recorded_cpu'] and len(s['resource_failures'])==1
cold=read(P/'cold.json');cv=sum(sum(ledger(r['ledger']).values()) for r in cold['modules'].values());outer=read(P/'assembly.json');ov=sum(ledger(outer['ledger']).values());assert (cv,ov)==(6930,50) and outer['estimate'].hex()==estimate.hex()
comp=read(P/'comparison.json');pin=read(P/'reference_pin.json');assert sha(Path(pin['path']))==pin['sha256'];ref=read(Path(pin['path']));cost=aggregate['visible_after_diagnostic']+cv+ov;budget=cost//512;assert cost==comp['total_visible'] and budget==comp['integer_sample_budget'] and cost==1838399658 and budget==3590624
assert comp['model_sha256']==inv['model_sha256'] and comp['model_dependent_opaque']==opaques and ref['estimate']==estimate and ref['reference_n']==100000000
qs=list(map(F,ref['reference_CP95']));a=F(estimate);p=F(ref['reference']);ratio=lambda q:budget*(a-q)**2/(q*(1-q));bounds=[0.0 if qs[0]<=a<=qs[1] else float(min(map(ratio,qs))),float(max(map(ratio,qs)))];assert float(ratio(p))==comp['point_ratio'] and bounds==comp['CP95_ratio_bounds']
out=dict(components=48,rescued_cell_chain=144,repeated_legacy_setup=repeat,estimate=estimate,numeric=aggregate['numeric'],integer=aggregate['integer'],diagnostic=aggregate['diagnostic'],total_visible=cost,sampling_budget=budget,point_ratio=float(ratio(p)),CP95_ratio_bounds=bounds,failure_cpu_retained=failure['cpu_seconds'],review_cpu=time.process_time()-start,scope='All48 saved component ledgers,144 exact-order cell chain, source/model pins, original output order, cold-once/outer assembly and rational uncertainty comparison checked. Failed partial and control repeats excluded from estimator count but failure CPU retained; repeated adapter setup included. One fixed model; incomplete native/control/memory/setup cost and MC uncertainty remain, no full-cost or ladder claim.')
Path(__file__).with_name('fibre_full_native_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
