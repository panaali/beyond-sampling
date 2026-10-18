"""Independent saved h3 mean/reference/baseline aggregate, no facet replay."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time
import mpmath as mp
import numpy as np
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/third_step_means/trained')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(x):
 f=F(x);return mp.mpf(f.numerator)/f.denominator
def close(x,y,tol='1e-95'):
 assert abs(x-y)<=mp.mpf(tol)*(1+abs(y)),(x,y)
start=time.process_time();p=read(P/'protocol.json');r=read(P/'result.json');g=read(P/'geometry.json')
for path,h in p['pins'].items():assert sha(path)==h
assert r['protocol_sha256']==sha(P/'protocol.json') and r['complete'] and r['all_checks_pass'] and g['complete'] and r['model']=='H16T4seed3'
assert p['control_saved_review'] and p['runner_source_review']
c=g['certificate'];S=len(c['sectors']);h=len(c['wi']);assert S==34 and h==16 and len(r['rows'])==544
weights=read(p['weights_json']);assert weights['complete'] and len(weights['rows'])==1;w=weights['rows'][0];assert w['name']==r['model'] and w['passed']
f=lambda pair:F(int(pair[0]),int(pair[1]))
assert list(map(f,c['wi']))==list(map(F,w['wi'])) and [[f(x) for x in row] for row in c['wh']]==[[F(x) for x in row] for row in w['wh']]
mp.mp.dps=220;means=[mp.mpf(0) for _ in range(h)];refs=means.copy();counts=Counter();bits=[0,0];keys=set();refs_evaluated=0;maxerr=mp.mpf(0)
for row in r['rows']:
 key=row['sector'],row['unit'];assert key not in keys;keys.add(key);assert all(row[k] for k in ('reference_pass','precision_pass','positivity_pass'))
 a=mp.mpf(row['value200']['value']);b=mp.mpf(row['value160']['value']);ref=mp.mpf(row['reference']['value']);err=abs(a-ref)
 assert abs(a-b)<=mp.mpf('1e-100')*(1+abs(a)) and err<=mp.mpf('1e-60')+abs(ref)*mp.mpf('1e-50') and a>=-mp.mpf('1e-100')
 close(mp.mpf(row['absolute_error']),err);means[key[1]]+=a;refs[key[1]]+=ref;maxerr=max(maxerr,err);counts.update(row['facet']['counts']);bits=[max(a,b) for a,b in zip(bits,row['facet']['bits'])];refs_evaluated+=row['reference']['evaluations']
assert keys=={(i,j) for i in range(S) for j in range(h)}
for j in range(h):close(means[j],mp.mpf(r['summary']['mean'][j]),'1e-190');close(refs[j],mp.mpf(r['summary']['reference_mean'][j]))
for x,y in zip(means,r['summary']['diagnostic']['exact_facet_h3']['mean']):close(x,mp.mpf(y),'1e-190')
close(maxerr,mp.mpf(r['summary']['max_cell_absolute_error']))
assert dict(counts)==r['counts']['facet_exact_named'] and bits==r['counts']['facet_retained_result_bits'] and refs_evaluated==r['reference_evaluations']==222042
assert r['counts']['network']==c['exact_named_counts']=={'mul':2*h*h+S*(2*h+2*h*h),'add':2*h*(h-1)+S*(2+h+2*h*(h-1))}
assert r['counts']['network_extra']==c['extra_named_counts'] and r['counts']['validation']==g['map_controls']['exact_named_counts']=={'mul':5*S*(4*h*h+16*h),'add':5*S*(4*h*h+7*h)}
assert len(g['map_controls']['rows'])==g['map_controls']['direct_analytic_forwards']==510
for key in ('candidate_cpu','reference_cpu'):
 total=0.
 for row in r['rows']:total+=row[key]
 assert r[key]==total
for key in ('build_cpu','validation_cpu'):assert r[key]==g[key]
assert r['candidate_cpu']+r['reference_cpu']+r['build_cpu']+r['validation_cpu']<=r['cpu']
with np.load(p['sample_reference'],allow_pickle=False) as z:
 sample=z['mean'];blocks=z['block_means'];se=z['mean_block_se'];assert sample.shape==(4,16) and blocks.shape==(8,4,16) and se.shape==(4,16)
 sample3=list(map(q,sample[2]));block3=[list(map(q,blocks[i,2])) for i in range(8)];assert se[2].tolist()==r['summary']['saved_mean_block_se']
closure=read(p['closure_result']);mixture=read(p['mixture_result']);assert closure['complete'] and mixture['complete'] and closure['reference_sha256']==sha(p['sample_reference']) and closure['weight_sha256']==p['original_weight_sha256']
cr=next(x for x in closure['rows'] if x['step']==3);mr=next(x for x in mixture['rows'] if x['step']==3)
methods={name:list(map(q,x['mean'])) for name,x in cr['methods'].items()};methods['retained_sector_gaussian']=list(map(q,mr['mean']));methods['exact_facet_h3']=means
assert set(methods)==set(r['summary']['diagnostic']) and all(len(v)==16 for v in methods.values())
def relative(a,b):return mp.sqrt(sum((x-y)**2 for x,y in zip(a,b))/sum(y*y for y in b))
diagnostic={}
for name,values in methods.items():
 saved=r['summary']['diagnostic'][name]
 for x,y in zip(values,saved['mean']):close(x,mp.mpf(y),'1e-190')
 e=relative(values,means);s=relative(values,sample3);close(e,mp.mpf(saved['relative_rms_to_exact_mean']));close(s,mp.mpf(saved['relative_rms_to_saved_sample']))
 for block,error in zip(block3,saved['block_relative_errors']):close(relative(values,block),mp.mpf(error))
 diagnostic[name]=dict(relative_to_exact=float(e),relative_to_saved_sample=float(s))
progress=[json.loads(x) for x in (P/'progress.jsonl').read_text().splitlines()]
assert len(progress)==2+2*544+1 and progress[0]['stage']=='model_attempt' and progress[-1]['stage']=='model_complete' and sum(x['stage']=='cell_attempt' for x in progress)==544 and sum(x['stage']=='cell_result' for x in progress)==544
out=dict(complete=True,cells=544,sectors=34,diagnostic=diagnostic,counts=r['counts'],reference_evaluations=refs_evaluated,research_cpu=r['cpu'],review_cpu=time.process_time()-start,result_sha256=sha(P/'result.json'),geometry_sha256=sha(P/'geometry.json'),new_author_facet_angular_or_forward_calls=0,scope='Exact stored weights, all saved cell/precision/reference checks, 16 mean sums, prior step3 baseline/sample diagnostics and counts independently contracted. Detailed sector/facet algebra is a separate reciprocal78 audit. Finite MP mean evidence only, no outward certificate/covariance/downstream accuracy/fullcost claim.')
Path(__file__).with_name('third_step_trained_aggregate_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='counts'}))
