"""Frozen T122 selected rank-region mass; exact topology, floating positive mass."""
import time
START=time.process_time()
import gzip, hashlib, importlib.util, json, math, resource
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/clean-neurons/m16/approaches')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out
EXACT_PATH=ROOT/'small_ladder_review/rational_geometry/exact.py'
MASS_PATH=ROOT/'small_h4_4/stable_volume.py'
ART=ROOT/'small_ladder_review/rational_geometry/geometry_h4_t4.json.gz'
ATTR=HERE.parent/'bulk_error_mechanism/attribution.json'
PANEL=HERE.parent/'bulk_joint_feasibility/panel_phase2.json'
e=module('exact',EXACT_PATH);v=module('volume',MASS_PATH)
def planes(top,second):
 return [tuple(int(q==i)-int(q==j) for q in range(4)) for i,j in [(top,second)]+[(second,j) for j in range(4) if j not in (top,second)]]
def initial(ex,rays):
 normals={}
 for i in range(4):
  rows=[r for j,r in enumerate(rays) if j!=i]
  n=tuple((-1)**j*ex.det([r[:j]+r[j+1:] for r in rows]) for j in range(4))
  if ex.dot(n,rays[i])<0:n=tuple(-x for x in n)
  normals[i]=ex.primitive(n,'facet')
 cone=e.Cone(rays,[15^(1<<i) for i in range(4)],normals);ex.certify(cone);return cone

def intersect(ex,cone,top,second):
 for i,n in enumerate(planes(top,second)):
  if ex.counts['clips']>=5000:raise TimeoutError('5000 clip cap')
  cone,_=ex.clip(cone,n,4+i,(True,False))
  if cone is None:return []
 ex.certify(cone)
 return ex.triangulate(cone)
def evaluate(rays):
 for q in range(4):assert min(r[q] for r in rays)>=0 or max(r[q] for r in rays)<=0
 normalized=np.array([[float(F(x,sum(abs(t) for t in r))) for x in r] for r in rays]).T
 a,b=v.mass(normalized,20);c,d=v.mass(normalized,16)
 return a,b,c,d

def controls(ex):
 results=[]
 for signs in [(1,1,1,1),(1,-1,1,-1),(-1,-1,-1,-1)]:
  rays=tuple(tuple(s*int(i==j) for j in range(4)) for i,s in enumerate(signs));cone=initial(ex,rays);det=F(0);mass=0.;nonempty=0
  for top in range(4):
   for second in range(4):
    if top==second:continue
    pieces=intersect(ex,cone,top,second);nonempty+=bool(pieces)
    for r,d in pieces:
     det+=d;mass+=evaluate(r)[0]
  assert det==1 and abs(mass-1/16)<1e-12
  results.append(dict(signs=signs,determinant=str(det),mass=mass,nonempty_pairs=nonempty))
 return results

def main():
 ex=e.Exact(seconds=max(.1,30-(time.process_time()-START)))
 out={'complete':False,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [EXACT_PATH,MASS_PATH,ART,ATTR,PANEL,Path(__file__)]}}
 out['controls']=controls(ex)
 data=json.load(gzip.open(ART,'rt'));attribution=json.loads(ATTR.read_text());panel=json.loads(PANEL.read_text());selected={h['history']:h for h in panel['histories']}
 pair_rows={(h['history'],p['top'],p['second']):dict(history=h['history'],top=p['top'],second=p['second'],classification=p['classification'],mass20=0.,bound20=0.,mass16=0.,bound16=0.,simplices=0,correct_mass20=0.,correct_bound20=0.,correct_simplices=0) for h in panel['histories'] for p in h['pairs']}
 out['parents']=[];out['pairs']=list(pair_rows.values())
 def save():
  out['cpu_seconds_including_imports']=time.process_time()-START;out['counts']=dict(ex.counts);out['max_bits']=dict(ex.bits);out['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
 try:
  for index,(simplex,key) in enumerate(zip(data['gate_simplices'],attribution['sets']['gate']['history_keys'])):
   if key not in selected:continue
   rays=tuple(tuple(int(x) for x in r) for r in simplex['integer_rays']);parent_det=F(*map(int,simplex['normalized_absolute_determinant']));cone=initial(ex,rays);total=F(0);children=0;local=[]
   for top in range(4):
    for second in range(4):
     if top==second:continue
     row=pair_rows[key,top,second];pieces=intersect(ex,cone,top,second)
     if row['classification']=='empty_rank':assert not pieces
     vals=[0.]*4
     for r,d in pieces:
      assert all(sum(a*b for a,b in zip(n,ray))>=0 for n in planes(top,second) for ray in r)
      total+=d;children+=1
      for j,value in enumerate(evaluate(r)):vals[j]+=value
     local.append((row,vals,len(pieces)))
   assert total==parent_det,'exact parent partition'
   # Only complete, determinant-verified parents enter the checkpoint.
   for row,vals,n in local:
    for k,value in zip(['mass20','bound20','mass16','bound16'],vals):row[k]+=value
    row['simplices']+=n
   out['parents'].append(dict(index=index,history=key,children=children,partition_exact=True,normalized_det=[str(parent_det.numerator),str(parent_det.denominator)]));save()
  # Existing correct-event simplices already fix top and second; verify all rays.
  for index,(simplex,key) in enumerate(zip(data['correct_simplices'],attribution['sets']['correct']['history_keys'])):
   if key not in selected:continue
   ex.check();rays=tuple(tuple(int(x) for x in r) for r in simplex['integer_rays']);interior=[sum(r[j] for r in rays) for j in range(4)];order=sorted(range(4),key=lambda j:(-interior[j],j));top,second=order[:2]
   assert all(sum(a*b for a,b in zip(n,r))>=0 for n in planes(top,second) for r in rays),'correct simplex needs rank refinement'
   row=pair_rows[key,top,second];assert row['classification'] not in ('empty_rank','uniformly_wrong')
   mass,bound,_,_=evaluate(rays);row['correct_mass20']+=mass;row['correct_bound20']+=bound;row['correct_simplices']+=1
  for row in pair_rows.values():
   if row['classification']=='uniformly_correct':assert abs(row['mass20']-row['correct_mass20'])<=row['bound20']+row['correct_bound20']+1e-14
  totals=defaultdict(lambda:defaultdict(float))
  for row in pair_rows.values():
   for k in ['mass20','bound20','mass16','bound16','simplices','correct_mass20','correct_bound20','correct_simplices']:totals[row['classification']][k]+=row[k]
  out['totals']={k:dict(v) for k,v in totals.items()};out['selected_mass20']=sum(r['mass20'] for r in pair_rows.values());out['selected_reference_mass']=panel['selected_gate_mass'];out['covered_uniform_mass20']=sum(r['mass20'] for r in pair_rows.values() if r['classification'] in ('uniformly_correct','uniformly_wrong'));out['unresolved_mass20']=1-out['covered_uniform_mass20'];out['complete']=True
 except Exception as err:
  out['failure']=repr(err);save();raise
 save();print(json.dumps({k:out[k] for k in ['complete','totals','selected_mass20','covered_uniform_mass20','unresolved_mass20','cpu_seconds_including_imports','counts']}))
if __name__=='__main__':main()
