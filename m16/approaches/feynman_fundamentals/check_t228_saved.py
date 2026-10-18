"""Independent T228 unsigned saved evidence audit; no author code or model work."""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict
import hashlib,itertools,json,math,time,traceback
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h2_34/review/width16_structure_review')
DATA=BASE/'snapshot/m16/approaches/width16_probe';TOL=F(1,2000000000)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def q(x):assert type(x) in (float,int) and math.isfinite(x);return F(x)
def main():
 start=time.process_time();dest=HERE/'t228_saved_review.json';assert not dest.exists()
 r=read(BASE/'results.json');p=read(BASE/'protocol.json');gate=read(BASE/'gate.json')
 assert r['complete'] and r['pass_all'] and r['protocol_sha256']==sha(BASE/'protocol.json') and r['gate_sha256']==sha(BASE/'gate.json')
 for path,digest in p['pins'].items():assert sha(BASE/path)==digest,path
 for path,digest in gate['pins'].items():assert sha(path)==digest,path
 for role,v in gate['reviews'].items():a=read(v['path']);assert a['complete'] and a['pass_all']
 expected=[(n,k) for n in ['target','zoo16_4','zoo16_5'] for k in [2,3,4]]
 assert [(x['model'],x['k']) for x in r['arrays']]==expected
 totalpieces=0;negative=0;zeros=0;comparisons=0;maxdiff=0.;maxentropy=0.;summary=[]
 for row in r['arrays']:
  name,k=row['model'],row['k'];old=read(DATA/f'eventword/{name}_k{k}.json')
  with np.load(DATA/f'eventword/{name}_k{k}.npz',allow_pickle=False) as z:
   assert set(z.files)=={'event','gate','mass'};ee=z['event'];gg=z['gate'];ww=z['mass']
  assert ee.ndim==gg.ndim==ww.ndim==1 and len(ee)==len(gg)==len(ww)==row['pieces']==old['pieces']
  assert ee.dtype.kind=='U' and gg.dtype.kind in 'iu' and ww.dtype==np.float64 and np.isfinite(ww).all()
  ratios=[float(x).as_integer_ratio() for x in ww];scale=max(b for a,b in ratios)
  joint={};neg=[];zero=0;ids=set()
  for i,(e,g,(a,b)) in enumerate(zip(ee.tolist(),gg.tolist(),ratios,strict=True)):
   assert g>=0 and len(e)==k-1 and e[0] in 'RS' and set(e)<={'R','S','L'}
   joint[e,g]=joint.get((e,g),0)+a*(scale//b);ids.add(g)
   if a<0:neg.append(dict(index=i,mass=str(F(a,b).numerator)+'/'+str(F(a,b).denominator)))
   zero+=a==0
  event=defaultdict(int);gate_mass=defaultdict(int)
  for (e,g),v in joint.items():event[e]+=v;gate_mass[g]+=v
  total=sum(event.values());assert total>0 and all(v>0 for v in event.values())
  assert ids==set(range(len(ids))) and len(ids)==row['regions']==old['regions']
  assert len(joint)==row['joint_keys']==old['eg_words'] and len(event)==old['event_words']
  assert neg==row['negative_piece_masses'] and zero==row['zero_piece_masses']
  assert row['positive_probability_interpretation_supported_by_saved_signs']==(not neg)
  negative+=len(neg);zeros+=zero;totalpieces+=len(ww)
  checks={v['label']:v for v in row['checks']};assert len(checks)==len(row['checks'])
  def compare(label,value,want,entropy=False):
   nonlocal comparisons,maxdiff,maxentropy
   c=checks.pop(label);assert c['saved']==float(want) and c['pass'] is True
   if entropy:
    assert abs(float(value)-c['got'])<=float(TOL);maxentropy=max(maxentropy,abs(float(value)-c['got']))
    d=abs(F(c['got'])-F(float(want)))
   else:
    value=F(value);assert c['got']==float(value);d=abs(value-F(float(want)))
   assert c['absolute_difference']==float(d) and d<=TOL
   comparisons+=1;maxdiff=max(maxdiff,float(d))
  bestsum=0;weighted=0;negativejoint=0
  for e in sorted(event):
   values=sorted((v for (label,g),v in joint.items() if label==e),reverse=True);mass=event[e]
   count=next(j for j,v in enumerate(itertools.accumulate(values),1) if 100*v>=99*mass)
   bestsum+=values[0];weighted+=mass*count;negativejoint+=sum(x<0 for x in values)
   # Independent insertion-rank law: at time t each rank 0..t has
   # probability 1/(t+1), independently across t for iid continuous inputs.
   prob=F(1)
   for t,letter in enumerate(e,1):prob*=F(t-1 if letter=='L' else 1,t+1)
   assert prob>0
   saved=row['per_event'][e];source=old['per_event'][e]
   assert F(saved['mass'])==F(mass,scale) and F(saved['analytic_event_mass'])==prob and F(saved['modal'])==F(values[0],mass)
   assert saved['n99']==source['n99']==count and saved['gate_words']==source['gate_words']==len(values)
   compare(e+'.mass',F(mass,scale),source['mass']);compare(e+'.modal',F(values[0],mass),source['modal']);compare(e+'.analytic_event_mass',F(mass,scale),prob)
  assert set(row['per_event'])==set(event) and row['negative_joint_masses']==negativejoint
  assert F(row['total_mass'])==F(total,scale) and F(row['modal_share'])==F(bestsum,total) and F(row['n99_mass_weighted'])==F(weighted,total)
  cond=math.exp(math.fsum(float(F(v,total))*math.log(float(F(event[e],v))) for (e,g),v in joint.items() if v>0))
  uncond=math.exp(math.fsum(float(F(v,total))*math.log(float(F(total,v))) for v in gate_mass.values() if v>0))
  for label,value in [('mass_sum_minus_1',F(total,scale)-1),('modal_share',F(bestsum,total)),('n99_mass_weighted',F(weighted,total))]:compare(label,value,old[label])
  for label,value in [('branching_given_event',cond),('branching_unconditional',uncond)]:
   assert abs(value-row[label])<=float(TOL);compare(label,value,old[label],True)
  assert not checks and row['arithmetic_pass'] and row['sign0_unreconstructable'] and row['geometry_mass_accuracy_certified'] is False
  assert row['historical_cpu_and_ops_unverified']=={key:old[key] for key in ['cpu','cpu_total','ops_geometry']}
  summary.append(dict(model=name,k=k,pieces=len(ww),regions=len(ids),modal_share=float(F(bestsum,total)),branching_given_event=cond,total_mass=float(F(total,scale))))
 tables=r['tables'];sym=read(DATA/'symm.json');mc=read(DATA/'symm_acc.json');variants=read(DATA/'symm_variants.json')
 records={}
 for a in sym['near_pairs_09']+sum(sym['top5'].values(),[])+sym['top5_both_perm']:
  key=(a['i'],a['j']);assert 0<=key[0]<key[1]<16
  if key in records:assert records[key]==a
  records[key]=a
 for field in ['cos_in','cos_in_perm','cos_out','cos_out_perm']:
  values=[abs(q(v[field])) for v in sym['top5'][field]];assert values==sorted(values,reverse=True)
  assert max(abs(q(v[field])) for v in records.values())==q(sym['max_abs'][field])
 assert all(any(abs(q(v[key]))>F(9,10) for key in ['cos_in','cos_in_perm','cos_out','cos_out_perm']) for v in sym['near_pairs_09'])
 assert tables['summary_unique_pair_records']==len(records)==17 and tables['declared_pairs']==sym['n_pairs']==120 and tables['all_pairs_reconstructed'] is False
 assert tables['variants_preserved']==variants and tables['maxima_match_retained_summaries']
 for record,(name,a,n) in zip(tables['paired_summary_consistency'],[('full',mc,10000000)]+[(name,variants[name],2000000) for name in ['pair47','pair68','pair17']],strict=True):
  mean=q(a['diff_pp'])/100;se=q(a['diff_se_pp'] if name=='full' else a['se_pp'])/100;second=n*se*se+mean*mean
  assert record['name']==name and record['n']==n and F(record['implied_difference_second_moment'])==second
  assert F(record['lower'])==abs(mean) and F(record['upper'])==min(F(1),q(a['disagree']))
  assert abs(mean)-TOL<=second<=min(F(1),q(a['disagree']))+TOL and record['consistency_pass']
  assert record['n_in_record']==(name=='full') and record['variant_n_only_readme_assumption']==(name!='full') and record['not_raw_outcome_or_CI_reconstruction']
 mean=q(mc['diff_pp'])/100;want=float(q(mc['acc_sym'])-q(mc['acc_target']));diff=abs(mean-F(want));c=tables['checks'][0]
 assert len(tables['checks'])==1
 assert c['label']=='fullgroup_mean_difference' and c['got']==float(mean) and c['saved']==want and c['absolute_difference']==float(diff) and c['pass'] and diff<=TOL;comparisons+=1
 assert tables['arithmetic_pass']
 prefix=[]
 for name in ['target','sym']:
  for k in [2,3,4]:prefix.append(dict(model=name,**read(DATA/f'prefix/{name}_k{k}.json')))
 assert tables['prefix_summaries']==prefix
 assert [v['regions'] for v in prefix if v['model']=='target']==[34,792,21032]
 events=list(map(json.loads,(BASE/'progress.jsonl').read_text().splitlines()));expected_events=[dict(phase='start')]
 for n,k in expected:expected_events += [dict(phase='attempt',model=n,k=k),dict(phase='result',model=n,k=k,arithmetic_pass=True)]
 expected_events += [dict(phase='tables_attempt'),dict(phase='tables_result',pass_=True)]
 expected_events[-1]={'phase':'tables_result','pass':True};expected_events.append(dict(phase='complete',cpu_seconds=r['cpu_seconds']))
 assert events==expected_events and comparisons==280 and totalpieces==118586
 for field in ['author_calls','model_calls','mass_calls','geometry_calls']:assert r[field]==0
 result=dict(complete=True,pass_all=True,task='T228 independent saved unsigned evidence audit',raw_commit='a29e4386',pins={str(BASE/n):sha(BASE/n) for n in ['results.json','progress.jsonl','gate.json','protocol.json','manifest.json','audit.py']},checker_sha256=sha(__file__),arrays=9,pieces=totalpieces,comparisons=comparisons,progress_events=len(events),negative_piece_masses=negative,zero_piece_masses=zeros,retained_symmetry_pair_records=len(records),max_saved_comparison_difference=maxdiff,max_independent_entropy_difference=maxentropy,arrays_summary=summary,review_cpu_seconds=time.process_time()-start,prior_checker_cpu_seconds=r['cpu_seconds'],new_author_model_geometry_mass_rng_calls=0,scope='Integer-scaled exact binary64 mass grouping, unsigned event/modal/99percent cover statistics, independent insertion-rank event probabilities and conditional-log entropy, retained symmetry/MC-summary constraints and prefix-count summaries. All saved numerical masses are positive; this is not their numerical integration certification.',limitations=['Missing sign0/history/raw rays/checkpoint/inline-variant inputs remain unreconstructed.','Low whole-history modal share does not lower-bound output accuracy or forbid history aggregation.','Pair scans and destructive SVD interventions do not exclude every symmetry or quotient.','No new confidence coverage, historical execution identity or complete compute claim.'])
 dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,sha256=sha(dest),cpu=result['review_cpu_seconds'],pieces=totalpieces,maxentropy=maxentropy)))
if __name__=='__main__':
 try:main()
 except BaseException:
  (HERE/'t228_saved_reviewer_failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2)+'\n');raise
