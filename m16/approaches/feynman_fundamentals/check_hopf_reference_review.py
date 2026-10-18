import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_native_cost');A=Path('/workspace/work/recover-small_h2_34/m16/approaches');start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(s,q):assert abs(F(s)-q)<=F(1,10**55)*max(abs(q),F(1))
d=read(P/'reference_comparison.json');ref=read(A/'radial_t4_reference/combined.json');identity=read(P/'model_identity.json');assert d['reference_sha256']==sha(A/'radial_t4_reference/combined.json') and d['runner_sha256']==sha(A/'class_region_t4/runner.py');assert identity['checkpoint_sha256']==d['checkpoint_sha256'] and identity['float64_c_order_sha256']==d['model_sha256'] and d['checkpoint_sha256'] in (A/'class_region_t4/runner.py').read_text();assert ref==d['reference'] and ref['cones']==6347 and not ref['full_float_accuracy_certified']
p,b=F(ref['estimate']),F(ref['integration_only_bound']);lo,hi=p-b,p+b;outrows=[]
for r in d['rows']:
 h64=r['level']=='h64';src=read(P/('model_h64_original.json' if h64 else 'model_probe_original.json'));assert r['estimate']==src['value'];cost=src['conditional_source_with_constant_setup'];assert r['candidate_conditional_source_budget']==cost
 per,extra=(193,8) if r['sampler']=='dense_original' else (161,7);assert (per,extra)==(r['per_sample_with_hit_aggregation'],r['per_batch_index_control']);n=r['sampling_draws'];price=lambda k:per*k+extra*((k+999999)//1000000)+2;assert price(n)<=cost<price(n+1) and price(n)==r['sampling_explicit_cost'];e=F(r['estimate']);f=lambda q:n*(e-q)**2/(q*(1-q));close(r['error'],e-p);close(r['conditional_error_interval'][0],e-hi);close(r['conditional_error_interval'][1],e-lo);close(r['matched_sampling_variance'],p*(1-p)/n);close(r['conditional_point_mse_ratio'],f(p));close(r['conditional_integration_only_ratio_range'][0],0 if lo<=e<=hi else min(f(lo),f(hi)));close(r['conditional_integration_only_ratio_range'][1],max(f(lo),f(hi)));outrows.append(dict(level=r['level'],sampler=r['sampler'],draws=n,point_ratio=float(f(p)),conditional_ratio_range=list(map(float,r['conditional_integration_only_ratio_range']))))
out=dict(rows=outrows,checkpoint=d['checkpoint_sha256'],review_cpu=time.process_time()-start,scope='Same-checkpoint provenance, maximal integer budgets and60-digit displayed ratios independently recomputed as exact rationals. Source-count conventions and integration-only reference interval are conditional; no full native/hardware or floating geometry certificate, no other model/noise inference.')
Path(__file__).with_name('hopf_reference_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
