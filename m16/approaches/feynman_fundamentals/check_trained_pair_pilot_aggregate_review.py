"""Independent saved five-pair aggregates; geometry reviewed separately by 78."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
from collections import Counter
import json,hashlib,resource,time
import numpy as np
import mpmath as mp
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();mp.mp.dps=180
O=Path(__file__).resolve().parent;P=Path('/workspace/work/recover-small_h2_34/m16/approaches/third_step_pairs/trained_pilot')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();dec=lambda x:F(Decimal(x));q=lambda x:mp.mpf(x.numerator)/x.denominator
outpath=O/'trained_pair_pilot_aggregate_review_results.json';assert not outpath.exists()
g=read(P/'protocol.json');d=read(P/'results.json');m=read(g['means']);assert d['complete'] and d['all_pass'] and d['protocol_sha256']==sha(P/'protocol.json')
for p,h in g['pins'].items():assert sha(p)==h,p
assert [(r['i'],r['j']) for r in d['rows']]==list(map(tuple,g['pairs']))==[(0,0),(1,1),(15,15),(0,1),(0,15)]
mu=list(map(dec,m['summary']['mean']));assert len(mu)==16
with np.load(g['sample_reference'],allow_pickle=False) as z:bm=z['block_means'];br=z['block_raw_second'];sc=z['covariance'];assert bm.shape==(8,4,16) and br.shape==(8,4,16,16) and sc.shape==(4,16,16)
closure=read(g['closure']);mixture=read(g['mixture']);assert closure['reference_sha256']==sha(g['sample_reference'])
cr=[r for r in closure['rows'] if r['step']==3];mr=[r for r in mixture['rows'] if r['step']==3];assert len(cr)==len(mr)==1
base={k:v['covariance'] for k,v in cr[0]['methods'].items()};base['retained_sector_gaussian']=mr[0]['covariance'];assert len(base)==3
progress=[json.loads(x) for x in (P/'progress.jsonl').read_text().splitlines()];assert len(progress)==351
assert [x['cell'] for x in progress if x['stage']=='cell_result']==[c for r in d['rows'] for c in r['cells']]
counts=Counter();statuses=Counter();evals=0;ccpu=rcpu=0.;maximum=F(0);rows=[];covs={}
for r in d['rows']:
 i,j=r['i'],r['j'];assert r['complete'] and [c['sector'] for c in r['cells']]==list(range(34))
 raw=ref=F(0)
 for c in r['cells']:
  v=dec(c['value120']['value']);rr=dec(c['reference']['value']);lo=dec(c['value80']['value']);e=abs(v-rr);maximum=max(maximum,e)
  assert c['passed'] and e<=F(1,10**50)+F(1,10**40)*abs(rr) and abs(v-lo)<=F(1,10**60)*(1+abs(v)) and v>=-F(1,10**60)
  assert abs(e-dec(c['absolute_error']))<F(1,10**150)
  raw+=v;ref+=rr;counts.update(c['certificate']['counts']);statuses[c['certificate']['status']]+=1;evals+=c['reference']['evaluations'];ccpu+=c['candidate_cpu'];rcpu+=c['reference_cpu']
 cov=raw-mu[i]*mu[j];covs[i,j]=cov
 assert abs(raw-dec(r['raw_second']))<F(1,10**135) and abs(ref-dec(r['reference_raw_second']))<F(1,10**95) and abs(cov-dec(r['covariance']))<F(1,10**135)
 dg=r['diagnostic'];sample=F(float(sc[2,i,j]));assert dg['saved_sample_covariance_hex']==float(sample).hex() and abs(dec(dg['new_minus_saved_sample_covariance'])-(cov-sample))<F(1,10**95)
 blocks=[F(float(br[b,2,i,j]))-F(float(bm[b,2,i]))*F(float(bm[b,2,j])) for b in range(8)]
 assert all(abs(v-dec(w))<F(1,10**115) for v,w in zip(blocks,dg['block_covariances'],strict=True))
 av=sum(blocks)/8;se=mp.sqrt(q(sum((v-av)**2 for v in blocks)/(7*8)));assert abs(se-mp.mpf(dg['descriptive_block_standard_error']))<mp.mpf('1e-95')
 for name,a in base.items():
  v=F(float(a[i][j]));b=dg['baseline_covariances'][name];assert b['value_hex']==float(v).hex() and abs(dec(b['baseline_minus_new'])-(v-cov))<F(1,10**95)
 rows.append({'pair':[i,j],'raw_second':float(raw),'covariance':float(cov),'sample_difference':float(cov-sample),'descriptive_block_se':float(se),'baseline_differences':{k:float(F(float(a[i][j]))-cov) for k,a in base.items()}})
for r in d['selected_covariance_checks']:
 j=r['j'];v=covs[0,0]*covs[j,j]-covs[0,j]**2;assert r['i']==0 and r['passed'] and v>=0 and abs(v-dec(r['minor']))<F(1,10**115)
assert statuses==Counter(pointed_full=145,measure_zero=25) and evals==76612
assert d['process_cpu']>=d['cpu']>=ccpu+rcpu and d['preflight_cpu']>=0
out={'pass':True,'pairs':5,'cells':170,'statuses':dict(statuses),'named_exact_counts':dict(counts),'reference_evaluations':evals,'max_reference_difference':float(maximum),'candidate_cpu':ccpu,'reference_cpu':rcpu,'author_process_cpu':d['process_cpu'],'rows':rows,'review_cpu':time.process_time()-start,'result_sha256':sha(P/'results.json'),'protocol_sha256':sha(P/'protocol.json'),'progress_sha256':sha(P/'progress.jsonl'),'scope':'Saved finite-MP five-pair raw-second/covariance/sample/baseline/count aggregation only. Geometry independently reviewed by 78. No author or reference replay, no full covariance or confidence/FP certificate.'}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
