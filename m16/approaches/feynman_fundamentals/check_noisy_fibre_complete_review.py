import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,json,math,time,zlib
from collections import defaultdict
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/fibre_t4_recovery')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b,tol=1e-12):assert abs(a-b)<=tol*max(1,abs(a),abs(b)),(a,b)
start=time.process_time(); identity=read(P/'noise_identity.json'); summary=read(P/'noise_summary.json'); comparison=read(P/'noise_comparison.json'); ref=read(P/'reference_summary.json')
rows=[read(P/'noise_preflight.json')]+[read(P/'noise_components'/f'{i:02d}.json') for i in range(1,48)]
raw=0.; obs=defaultdict(dict)
for i,r in enumerate(rows):
 assert r['complete'] and r['model_sha256']==identity['model_sha256'] and r['observer_sha256']==sha(P/'observer.py')
 assert (r['family'],r['side'])==divmod(i,2)
 if i:assert r['index']==i and r['runner_sha256']==sha(P/'noise_components.py')
 assert r['stats']['traces']==r['observer']['trace']['calls']
 assert math.isfinite(r['raw_error_component']);raw+=r['raw_error_component']
 for name,counters in r['observer'].items():
  for key,value in counters.items():obs[name][key]=max(obs[name].get(key,0),value) if key.startswith('max_') else obs[name].get(key,0)+value
 for name in ['trace','_walk_down']:assert r['observer'].get(name,{}).get('exhaustions',0)==0
 assert r['observer'].get('subdivide',{}).get('unsatisfied_depth_caps',0)==0
near(raw,summary['raw_error_sum']);e=1-raw/(2*math.pi**2);assert e==summary['estimate']==comparison['estimate'];assert dict(obs)==summary['observer']
cost=sum(r['legacy_ops'] for r in rows);assert cost==summary['legacy_ops']==1166612480
for key in ['traces','pieces']:assert sum(r['stats'][key] for r in rows)==summary['stats'][key]
for key in ['calls','exact_zero']:assert sum(r['guard'][key] for r in rows)==summary['guard'][key]
near(sum(r['cpu_seconds'] for r in rows),summary['component_cpu_seconds'])
records=[read(P/'noise_reference'/f'{i:03d}.json') for i in range(100)];pins=read(P/'reference_pins.json')
for path,digest in pins.items():assert sha(Path(path))==digest
seed=zlib.crc32(identity['name'].encode());state=np.random.default_rng(seed).bit_generator.state
for i,r in enumerate(records):
 assert r['index']==i and r['complete'] and r['n']==r['target_n']==1000000 and r['planned_total_n']==100000000
 assert r['seed_crc32']==seed and r['source_pins']==pins and r['model_sha256']==identity['model_sha256']
 assert r['rng_before']==state;state=r['rng_after'];assert len(r['batches'])==2
 assert all(b['n']==500000 and isinstance(b['hits'],int) and 0<=b['hits']<=b['n'] for b in r['batches'])
 assert sum(b['hits'] for b in r['batches'])==r['hits']
n=sum(r['n'] for r in records);k=sum(r['hits'] for r in records);assert (k,n)==(60561725,100000000)==(ref['hits'],ref['n']);p=k/n;assert p==ref['accuracy']
se=math.sqrt(p*(1-p)/n);near(se,ref['standard_error']);assert sum(r.get('extra_parity_evaluations',0) for r in records)==312==ref['extra_parity_evaluations'];near(sum(r['cpu_seconds'] for r in records),ref['reference_cpu_seconds'])
# Invert binomial tails independently of the author's beta-quantile formula.
lo=brentq(lambda q:binom.sf(k-1,n,q)-.025,p-.001,p,xtol=5e-15)
hi=brentq(lambda q:binom.cdf(k,n,q)-.025,p,p+.001,xtol=5e-15)
for a,b in zip([lo,hi],ref['binomial_CP95']):near(a,b,3e-14)
assert lo<e<hi
err=e-p;near(err,comparison['error']);near(cost*err**2/(320*p*(1-p)),comparison['historical_ratio_320'])
near(max(err**2-se**2,0)*cost/(320*p*(1-p)),comparison['historical_variance_subtracted_floor_ratio_320'])
near(cost-6*summary['guard']['calls'],comparison['original_legacy_without_guard'])
budget=min(512*q*(1-q)/(e-q)**2 for q in ref['binomial_CP95']);near(budget,comparison['conditional_sampling_break_even_actual_cost_512']['minimum_over_CP95']);assert comparison['conditional_sampling_break_even_actual_cost_512']['unbounded_maximum']
out=dict(components=48,reference_blocks=100,hits=k,n=n,estimate=e,reference=p,binomial_tail_inversion=[lo,hi],legacy_cost=cost,conditional_CP95_minimum_budget_512=budget,no_caps=True,reference_rng_adjacency_and_initial_state=True,extra_parity_forwards=312,author_component_cpu=summary['component_cpu_seconds'],author_reference_cpu=ref['reference_cpu_seconds'],review_cpu=time.process_time()-start,scope='Saved records and source pins only; no fibre, RNG stream or network rerun. One fixed noisy model, statistical binomial uncertainty and incomplete estimator cost. State adjacency is not independent replay of generated samples.')
Path(__file__).with_name('noisy_fibre_complete_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
