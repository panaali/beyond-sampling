"""Independent T227 saved contraction using integer-scaled dyadic tail masses.
No author imports, RNG, walks, LPs, models or mass integration.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time,traceback
HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h2_78/review/tie_tail_evidence')
MODELS=['z6s0','z6s1','z6s2','z6s3','z6s4','orig','s005','s010']
SCHEMES=['mech','pos','pos_flips','nopos','relpos','coarse'];LS=[10,30,100]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def q(x):assert type(x) in (float,int) and math.isfinite(x);return F(x)
def normalized(x):return json.loads(json.dumps(x))
def key(p,T,s):
 a,b,c=p['a'],p['b'],p['p1'];kind=p['kind'];ft=tuple(map(tuple,p['flips']));head=c if kind=='23' else -1
 position=(kind,a,b,c);shape=(p['ub'],p['wt'],ft)
 last='max' if head==T-1 else ('pair' if T-1 in (a,b) else ('third' if kind=='12' and c==T-1 else 'below'))
 age=T-1-(head if kind=='23' else b)
 return {'mech':position+shape,'pos':position,'pos_flips':position+(ft,),
         'nopos':(kind,)+shape,'relpos':(kind,b-a,age,last)+shape,
         'coarse':(kind,p['ub'],p['wt'],min(p['nflips'],3))}[s]
def group(points,weights,T,s):
 d={}
 for p,w in zip(points,weights,strict=True):d.setdefault(key(p,T,s),[]).append(w)
 return d
def ranked(g):
 mass={k:sum(v) for k,v in g.items()};return sorted(mass,key=mass.get,reverse=True),mass
def share(v):return F(sum(sorted(v,reverse=True)[:(len(v)+19)//20]),sum(v))
def main():
 start=time.process_time();dest=HERE/'t227_saved_review.json';assert not dest.exists()
 r=read(BASE/'audit/results.json');p=read(BASE/'protocol.json');gate=read(BASE/'gate.json')
 assert r['complete'] and r['current'] is None and 'failure' not in r
 assert r['protocol_sha256']==sha(BASE/'protocol.json') and r['gate_sha256']==sha(BASE/'gate.json')
 for path,h in {**p['pins'],**gate['pins']}.items():assert sha(path)==h,path
 for role,entry in gate['reviews'].items():
  review=read(entry['path']);assert review['complete'] and review['pass_all'] and sha(entry['path'])==entry['sha256']
 rawsummary=read(BASE/'frozen/summary.json');blind=read(BASE/'frozen/results/blind_split.json')
 comparisons={v['field']:v for v in r['comparisons']};assert len(comparisons)==len(r['comparisons'])==1015
 maxdiff=F();checked=0
 def compare(field,value,saved):
  nonlocal maxdiff,checked
  c=comparisons.pop(field);assert F(c['exact'])==value and c['saved']==saved
  diff=abs(value-q(saved));assert F(c['absolute_difference'])==diff and diff<=F(1,200000000000)
  maxdiff=max(maxdiff,diff);checked+=1
 assert [m['name'] for m in r['models']]==MODELS
 totals=dict(points=0,missing=0,duplicate_flip_lists=0,ordered_not_sorted_lists=0);primary={};coarse={};cache={}
 for name,model in zip(MODELS,r['models'],strict=True):
  d=read(BASE/f'frozen/results/{name}.json');s=rawsummary[name];T=6 if name.startswith('z6s') else 10
  assert (d['model'],d['T'],d['n'])==(name,T,16384) and model['T']==T
  assert set(model['schemes'])==set(s['schemes'])==set(SCHEMES) and model['primary_decision'] is None
  weights={rule:[q(p['g2']) for p in d['rules'][rule]['tail']] for rule in ['700','701']}
  scale=max(x.denominator for ws in weights.values() for x in ws);assert scale&(scale-1)==0
  points={};ints={}
  for j,rule in enumerate(['700','701']):
   raw=d['rules'][rule];allp=raw['tail'];ws=weights[rule];assert len(allp)==1639 and ws==sorted(ws,reverse=True) and all(x>=0 for x in ws)
   ii=[x.numerator*(scale//x.denominator) for x in ws];tail=F(sum(ii),scale);globalmass=q(raw['sum_g2']);assert globalmass>0
   compare(f'{name}/{rule}/tail_g2',tail,raw['tail_g2']);compare(f'{name}/{rule}/top5',tail/globalmass,raw['top5pct_share']);compare(f'{name}/{rule}/top1',F(sum(ii[:327]),scale)/globalmass,raw['top1pct_share'])
   compare(f'{name}/{rule}/summary_error',2*q(raw['err']),s['err'][j]);assert s['top5'][j]==raw['top5pct_share'] and s['top1'][j]==raw['top1pct_share']
   assert type(raw['nzero']) is int and 0<=raw['nzero']<=32768
   missing=[];keep=[];ww=[];dup=0;unsorted=0
   for pp,w in zip(allp,ii,strict=True):
    assert q(pp['Fcheck'])>=0
    if not pp.get('kind'):missing.append(w);continue
    a,b,c=pp['a'],pp['b'],pp['p1'];assert all(type(v) is int and 0<=v<T for v in [a,b,c]) and a<b and c not in (a,b)
    assert pp['kind'] in ['12','23'] and pp['side'] in ['+','-']
    assert pp['wt'] in (['tie','third','other'] if pp['kind']=='12' else ['tie','max','other'])
    assert type(pp['nflips']) is int and pp['nflips']>=0 and len(pp['flips'])==min(3,pp['nflips'])
    ft=list(map(tuple,pp['flips']));assert all(len(x)==2 and all(type(v) is int for v in x) and 0<=x[0]<T and 0<=x[1]<16 for x in ft)
    dup+=len(ft)!=len(set(ft));unsorted+=ft!=sorted(ft)
    u=q(pp['u']);bucket=sum(u>x for x in [F(0),F(1,4),F(1,2)]);assert pp['ub']==bucket
    assert type(pp['npos']) is int and 0<=pp['npos']<=T and (pp['npos']<=1)==(u<=0)
    assert type(pp['ncomp']) is int and pp['ncomp']>=1 and q(pp['width'])>=0
    keep.append(pp);ww.append(w)
   row=dict(rule=rule,total_points=1639,retained=len(keep),missing_kind=len(missing),missing_g2=str(F(sum(missing),scale)),max_saved_Fcheck=str(max(q(x['Fcheck']) for x in allp)),retained_duplicate_flip_lists=dup,retained_non_sorted_flip_lists=unsorted,total_tail_g2=str(tail))
   assert row==model['rules'][j];points[rule]=keep;ints[rule]=ww
   totals['points']+=len(allp);totals['missing']+=len(missing);totals['duplicate_flip_lists']+=dup;totals['ordered_not_sorted_lists']+=unsorted
  assert s['ntail']==len(points['700'])
  compare(f'{name}/blind-error-summary',q(blind[name]['err']),s['err'][0]);compare(f'{name}/blind-top5-summary',q(blind[name]['top5_all']),s['top5'][0])
  for scheme in SCHEMES:
   ga=group(points['700'],ints['700'],T,scheme);gb=group(points['701'],ints['701'],T,scheme);gab=group(points['700']+points['701'],ints['700']+ints['701'],T,scheme)
   ka,ma=ranked(ga);kb,mb=ranked(gb);kab,mab=ranked(gab);cache[name,scheme]=(ka,ma,mb)
   old=s['schemes'][scheme];saved=model['schemes'][scheme]
   assert saved['nlabels_A']==old['nlabels_A']==len(ga) and saved['nlabels_AB']==old['nlabels_AB']==len(gab)
   for field,keys,mass in [('selected_A100',ka,ma),('selected_B100',kb,mb)]:assert saved[field]==normalized([dict(label=k,mass=str(F(mass[k],scale))) for k in keys[:100]])
   metrics={};gaps={}
   for L in LS:
    aa,bb,ab=set(ka[:L]),set(kb[:L]),set(kab[:L])
    vals={f'in_A@{L}':F(sum(ma[k] for k in aa),sum(ma.values())),f'oos_AB@{L}':F(sum(v for k,v in mb.items() if k in aa),sum(mb.values())),f'oos_BA@{L}':F(sum(v for k,v in ma.items() if k in bb),sum(ma.values())),f'in_AB@{L}':F(sum(mab[k] for k in ab),sum(mab.values()))}
    for field,value in vals.items():metrics[field]=str(value);compare(f'{name}/{scheme}/{field}',value,old[field])
    gaps[str(L)]={k:None if len(keys)<=L else str(F(mass[keys[L-1]]-mass[keys[L]],scale)) for k,keys,mass in [('A',ka,ma),('B',kb,mb)]}
   assert metrics==saved['metrics'] and gaps==saved['selection_gaps']
   big=sorted([v for v in ga.values() if len(v)>=20],key=sum,reverse=True)[:10]
   within=dict(pooled=str(share(ints['700'])),labels=[str(share(v)) for v in big],npts=list(map(len,big)))
   assert within==saved['within'] and within['npts']==old['within']['npts']
   compare(f'{name}/{scheme}/within/pooled',F(within['pooled']),old['within']['pooled'])
   for i,value in enumerate(within['labels']):compare(f'{name}/{scheme}/within/{i}',F(value),old['within']['labels'][i])
  for L in LS:compare(f'{name}/singleton/{L}',F(sum(sorted(ints['700'],reverse=True)[:L]),sum(ints['700'])),s['null_singleton'][str(L)])
  pv=F(model['schemes']['mech']['metrics']['oos_AB@100']);assert model['primary_below_70pct']==(pv<F(7,10));primary[name]=str(pv);coarse[name]=model['schemes']['coarse']['metrics']['oos_AB@30']
 cross=[]
 for scheme in ['mech','pos','relpos','nopos']:
  ka,_,_=cache['orig',scheme]
  for name in ['orig','s005','s010']:
   _,_,mb=cache[name,scheme]
   for L in LS:
    chosen=set(ka[:L]);value=F(sum(v for k,v in mb.items() if k in chosen),sum(mb.values()));kk=f'{name}@{L}'
    cross.append(dict(scheme=scheme,key=kk,exact=str(value)));compare(f'cross/{scheme}/{kk}',value,rawsummary['cross_copy_from_orig'][scheme][kk])
 assert cross==r['cross_copy'] and len(cross)==36
 for pr in r['prefix']:
  data=read(BASE/'frozen/results'/pr['file']);assert data['steps']==pr['steps']
  assert all(v['complete'] is True and v['regions']<=v['cells'] for v in data['steps'])
  for i,entry in enumerate(data['steps'][1:],1):
   for kind in ['cells','regions']:compare(f'{pr["file"]}/{entry["k"]}/growth_{kind}',F(entry[kind],data['steps'][i-1][kind]),entry['growth_'+kind])
 assert not comparisons and checked==1015
 assert r['primary_fixed_rule_kill']==any(F(primary[n])<F(7,10) for n in ['orig','s005','s010'])
 for field in ['author_imports','model_calls','random_calls','LP_calls','mass_calls']:assert r[field]==0
 events=list(map(json.loads,(BASE/'audit/progress.jsonl').read_text().splitlines()));expected=[dict(stage='started')]
 for name in MODELS:expected += [dict(stage='scheme_complete',model=name,scheme=s) for s in SCHEMES]+[dict(stage='model_complete',model=name)]
 expected.append(dict(stage='completed',results_sha256=sha(BASE/'audit/results.json')));assert events==expected
 result=dict(complete=True,pass_all=True,task='T227 independent deterministic saved audit',raw_commit='abe4faab',pins={str(BASE/n):sha(BASE/n) for n in ['audit/results.json','audit/progress.jsonl','run1.log','gate.json','protocol.json','source_manifest.json','check_saved.py']},checker_sha256=sha(__file__),models=8,rules=16,schemes_per_model=6,comparisons=checked,cross_copy_scores=36,progress_events=len(events),totals=totals,max_historical_display_difference=str(maxdiff),primary_mech_L100={k:float(F(v)) for k,v in primary.items()},coarse_oos_L30={k:float(F(v)) for k,v in coarse.items()},review_cpu_seconds=time.process_time()-start,prior_checker_cpu_seconds=r['cpu_seconds'],new_author_model_rng_lp_mass_calls=0,scope='Exact saved contractions independently use integer-scaled dyadic masses, stable encounter-order selections, all deterministic metrics/within-tail/cross-copy/schema and progress. Frozen primary finite-rule test fails; high coarse-label coverage remains retained.',limitations=['No reconstruction of random half/null assignments, full-face arrays or original label/walk correctness.','Source face-sign versus actual-input-event distinction remains.','No population coverage CI, universal tail-family exclusion, model identity, exact prefix feasibility or integration/cost certificate.','Within-label statistics condition on global tail; coarse coverage alone does not prove efficient dedicated-family integration.'])
 dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,sha256=sha(dest),cpu=result['review_cpu_seconds'],totals=totals,maxdiff=float(maxdiff))))
if __name__=='__main__':
 try:main()
 except BaseException:
  (HERE/'t227_saved_reviewer_failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2)+'\n');raise
