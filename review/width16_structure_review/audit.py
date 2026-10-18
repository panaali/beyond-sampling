"""T228 fixed saved-only arithmetic. Never import author modules or evaluate models."""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict, Counter
import hashlib, itertools, json, math, os, resource, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parent
PINS=ROOT/'protocol.json';GATE=ROOT/'gate.json'
OUT=ROOT/'results.json';PROGRESS=ROOT/'progress.jsonl'
BASE=ROOT/'snapshot/m16/approaches/width16_probe'
ABS_TOL=F(1,2_000_000_000)  # 5e-10, frozen before any array decode

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(obj):
 p=OUT.with_suffix('.tmp');p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(p,OUT)
def event(obj):
 with PROGRESS.open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def enc(x):return str(x.numerator)+'/'+str(x.denominator)
def checkclose(got,want,label,checks):
 a=got if isinstance(got,F) else F(float(got));b=F(float(want));d=abs(a-b)
 checks.append({'label':label,'got':float(a),'saved':float(b),'absolute_difference':float(d),'pass':d<=ABS_TOL})
def entropy(groups,total):
 # Independent decomposition H(G,E)-H(E), summing normalized terms with fsum.
 by_e=defaultdict(F)
 for (e,g),w in groups.items():by_e[e]+=w
 joint=math.fsum(-float(w/total)*math.log(float(w/total)) for w in groups.values() if w>0)
 marginal=math.fsum(-float(w/total)*math.log(float(w/total)) for w in by_e.values() if w>0)
 return math.exp(joint-marginal)
def run_array(name,k,np):
 fn=BASE/'eventword'/f'{name}_k{k}';saved=json.loads(fn.with_suffix('.json').read_text());checks=[]
 with np.load(fn.with_suffix('.npz'),allow_pickle=False) as z:
  assert set(z.files)=={'event','gate','mass'},z.files
  ea,ga,ma=[z[q] for q in ('event','gate','mass')]
 assert ea.ndim==ga.ndim==ma.ndim==1 and len(ea)==len(ga)==len(ma)
 assert ea.dtype.kind=='U' and ga.dtype.kind in 'iu' and ma.dtype==np.dtype('float64')
 joint=defaultdict(F);by_e=defaultdict(F);by_g=defaultdict(F);neg=[];zeros=0;labels=set();ids=set()
 for i,(e,g,w) in enumerate(zip(ea.tolist(),ga.tolist(),ma.tolist())):
  assert math.isfinite(w) and isinstance(g,int) and g>=0
  assert len(e)==k-1 and e[0] in 'RS' and set(e)<={'R','S','L'}
  q=F(w);joint[e,g]+=q;by_e[e]+=q;by_g[g]+=q;labels.add(e);ids.add(g)
  if q<0:neg.append({'index':i,'mass':enc(q)})
  if q==0:zeros+=1
 total=sum(by_e.values(),F());assert total>0 and all(w>0 for w in by_e.values())
 assert ids==set(range(len(ids)))
 assert len(ma)==saved['pieces'] and len(ids)==saved['regions'] and len(joint)==saved['eg_words'] and len(labels)==saved['event_words']
 # Each strict order permutation has exact probability 1/k! under iid continuous inputs.
 event_counts=Counter()
 for order in itertools.permutations(range(k)):
  rank={coord:j for j,coord in enumerate(order)};s=''
  for t in range(1,k):
   n_above=sum(rank[j]>rank[t] for j in range(t))
   s+='R' if n_above==0 else 'S' if n_above==1 else 'L'
  event_counts[s]+=1
 assert set(event_counts)==set(by_e)==set(saved['per_event'])
 per={};modal=F();weighted_cover=F();negative_joint=0
 for e in sorted(by_e):
  ws=sorted((w for (ev,g),w in joint.items() if ev==e),reverse=True)
  target=F(99,100)*by_e[e];cum=F();n99=0
  for w in ws:
   cum+=w;n99+=1
   if cum>=target:break
  best=ws[0];modal+=best;weighted_cover+=by_e[e]*n99
  rec=saved['per_event'][e];assert len(ws)==rec['gate_words'];assert n99==rec['n99'],(e,n99,rec['n99'])
  checkclose(by_e[e],rec['mass'],e+'.mass',checks);checkclose(best/by_e[e],rec['modal'],e+'.modal',checks)
  exact_event=F(event_counts[e],math.factorial(k))
  checkclose(by_e[e],exact_event,e+'.analytic_event_mass',checks)
  negative_joint+=sum(w<0 for w in ws)
  per[e]={'mass':enc(by_e[e]),'analytic_event_mass':enc(exact_event),'modal':enc(best/by_e[e]),'n99':n99,'gate_words':len(ws)}
 hg=math.exp(math.fsum(-float(w/total)*math.log(float(w/total)) for w in by_g.values() if w>0))
 hge=entropy(joint,total)
 got={'mass_sum_minus_1':total-1,'modal_share':modal/total,'n99_mass_weighted':weighted_cover/total,'branching_unconditional':hg,'branching_given_event':hge}
 for key,value in got.items():checkclose(value,saved[key],key,checks)
 return {'model':name,'k':k,'pieces':len(ma),'regions':len(ids),'joint_keys':len(joint),'negative_piece_masses':neg,'zero_piece_masses':zeros,'negative_joint_masses':negative_joint,'positive_probability_interpretation_supported_by_saved_signs':not neg,
         'total_mass':enc(total),'modal_share':enc(modal/total),'n99_mass_weighted':enc(weighted_cover/total),'branching_given_event':hge,'branching_unconditional':hg,'per_event':per,'checks':checks,'arithmetic_pass':all(c['pass'] for c in checks),
         'sign0_unreconstructable':True,'historical_cpu_and_ops_unverified':{q:saved[q] for q in ('cpu','cpu_total','ops_geometry')},'geometry_mass_accuracy_certified':False}
def source_tables():
 s=json.loads((BASE/'symm.json').read_text());a=json.loads((BASE/'symm_acc.json').read_text());v=json.loads((BASE/'symm_variants.json').read_text())
 assert s['H']==16 and s['n_pairs']==120
 keys=('cos_in','cos_in_perm','cos_out','cos_out_perm');unique={}
 for r in s['near_pairs_09']+sum(s['top5'].values(),[])+s['top5_both_perm']:
  ij=(r['i'],r['j']);assert 0<=ij[0]<ij[1]<16
  if ij in unique:assert unique[ij]==r
  unique[ij]=r
 for key in keys:
  top=s['top5'][key];assert len(top)==5 and all(abs(top[i][key])>=abs(top[i+1][key]) for i in range(4))
  assert max(abs(r[key]) for r in unique.values())==s['max_abs'][key]
 assert all(any(abs(r[key])>.9 for key in keys) for r in s['near_pairs_09'])
 checks=[];checkclose(F(a['diff_pp'])/100,F(a['acc_sym'])-F(a['acc_target']),'fullgroup_mean_difference',checks)
 assert a['n']==10_000_000 and set(v)=={'pair47','pair68','pair17'}
 mc=[]
 for name,row,n,has_n in [('full',a,a['n'],True)]+[(name,row,2_000_000,False) for name,row in v.items()]:
  mean=F(row['diff_pp'])/100;se=F(row.get('diff_se_pp',row.get('se_pp')))/100;dis=F(row['disagree']);implied_second=n*se*se+mean*mean
  lo=abs(mean);hi=min(F(1),dis)
  ok=lo-ABS_TOL<=implied_second<=hi+ABS_TOL
  mc.append({'name':name,'n':n,'n_in_record':has_n,'variant_n_only_readme_assumption':not has_n,'implied_difference_second_moment':enc(implied_second),'lower':enc(lo),'upper':enc(hi),'consistency_pass':ok,'not_raw_outcome_or_CI_reconstruction':True})
  assert ok
 prefix=[]
 for model in ('target','sym'):
  for k in (2,3,4):
   r=json.loads((BASE/'prefix'/f'{model}_k{k}.json').read_text());assert r['k']==k and 0<r['regions']<=r['cells'];prefix.append({'model':model,**r})
 assert [r['regions'] for r in prefix if r['model']=='target']==[34,792,21032]
 return {'summary_unique_pair_records':len(unique),'declared_pairs':120,'all_pairs_reconstructed':len(unique)==120,'maxima_match_retained_summaries':True,'checks':checks,'paired_summary_consistency':mc,'prefix_summaries':prefix,'variants_preserved':v,'arithmetic_pass':all(c['pass'] for c in checks)}
def main():
 assert not OUT.exists() and not PROGRESS.exists() and not OUT.with_suffix('.tmp').exists()
 protocol=json.loads(PINS.read_text());gate=json.loads(GATE.read_text());assert gate['complete'] and gate['execution_allowed'] and gate['protocol_sha256']==sha(PINS)
 required={'audit.py','protocol.json','manifest.json','PROTOCOL.md'}
 assert required<={str(Path(p).relative_to(ROOT)) for p in gate['pins'] if Path(p).is_relative_to(ROOT)}
 assert {'audit.py','manifest.json','PROTOCOL.md','SOURCE_NOTES.md'}<=set(protocol['pins'])
 manifest=json.loads((ROOT/'manifest.json').read_text())
 assert {r['snapshot'] for r in manifest['files']}<=set(protocol['pins'])
 assert set(gate['reviews'])=={'source_provenance','checker_source_resource'}
 for role,review in gate['reviews'].items():
  path=Path(review['path']);assert gate['pins'].get(str(path))==review['sha256']==sha(path)
  artifact=json.loads(path.read_text());assert artifact['complete'] is True and artifact['pass_all'] is True
  assert review['commit'] in gate['published_commits']
 for rel,h in protocol['pins'].items():assert sha(ROOT/rel)==h,(rel,'pin')
 for path,h in gate['pins'].items():assert sha(Path(path))==h,(path,'gate pin')
 for commit in gate['published_commits']:subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/ccr-6a61f25c-wn80ii'],check=True)
 for limit,cap in ((resource.RLIMIT_CPU,20),(resource.RLIMIT_AS,2097152*1024)):
  soft,hard=resource.getrlimit(limit);assert 0<soft<=cap and 0<hard<=cap
 assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
 start=time.process_time();out={'task':'T228','complete':False,'pass_all':False,'protocol_sha256':sha(PINS),'gate_sha256':sha(GATE),'arrays':[],'author_calls':0,'model_calls':0,'mass_calls':0,'geometry_calls':0}
 save(out);event({'phase':'start'})
 try:
  import numpy as np
  out['runtime_numpy']=np.__version__
  for name in ('target','zoo16_4','zoo16_5'):
   for k in (2,3,4):
    event({'phase':'attempt','model':name,'k':k});save(out)
    row=run_array(name,k,np);out['arrays'].append(row);save(out);event({'phase':'result','model':name,'k':k,'arithmetic_pass':row['arithmetic_pass']})
    assert row['arithmetic_pass'],(name,k,'saved arithmetic or marginal mismatch')
    assert time.process_time()-start<17,'soft CPU cap'
  event({'phase':'tables_attempt'});out['tables']=source_tables();save(out);event({'phase':'tables_result','pass':out['tables']['arithmetic_pass']});assert out['tables']['arithmetic_pass']
  assert len(out['arrays'])==9
  assert len(PROGRESS.read_text().splitlines())==21
  out['complete']=out['pass_all']=True;out['cpu_seconds']=time.process_time()-start;out['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  save(out);event({'phase':'complete','cpu_seconds':out['cpu_seconds']})
 except BaseException as ex:
  out['failure']={'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc()};out['cpu_seconds']=time.process_time()-start;save(out);event({'phase':'failure','type':type(ex).__name__,'message':str(ex)});raise
if __name__=='__main__':main()
