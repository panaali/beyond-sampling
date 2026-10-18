"""Independent exact coordinates, projections, canonical records and fixed controls."""
from check_domain_sufficient_saved import unpack,matrix,dot,evaluate,sha,H
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import itertools,json,time
P=Path('/workspace/work/recover-small_h2_910/review/cone_domain_equivalence/implementation')
def determinant(a):
 n=len(a);return sum((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))*product(a[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
def product(a):
 v=F(1)
 for x in a:v*=x
 return v
def rank(a):
 if not a:return 0
 for n in range(min(len(a),len(a[0])),0,-1):
  if any(determinant([[a[i][j] for j in cs] for i in rs]) for rs in itertools.combinations(range(len(a)),n) for cs in itertools.combinations(range(len(a[0])),n)):return n
 return 0
def elimination_counts(a,limit):
 # Independent scalar contraction of the frozen named-operation convention.
 a=[r[:] for r in a];n=len(a);w=len(a[0]);at=0;c=Counter()
 for col in range(limit):
  choices=[i for i in range(at,n) if a[i][col]]
  if not choices:continue
  k=choices[0];a[k],a[at]=a[at],a[k];v=a[at][col];a[at]=[x/v for x in a[at]];c['div']+=w
  for i in range(n):
   if i==at or not a[i][col]:continue
   v=a[i][col];a[i]=[x-v*y for x,y in zip(a[i],a[at],strict=True)];c['mul']+=w;c['sub']+=w
  at+=1
  if at==n:break
 return c

def main():
 start=time.process_time();raw=json.loads((P/'controls.json').read_text());d=unpack(raw);p=json.loads((P/'protocol.json').read_text());events=[json.loads(s) for s in (P/'progress.jsonl').read_text().splitlines()]
 assert d['status']=='complete' and d['all_pass'] and d['protocol_sha256']==sha(P/'protocol.json')
 for path,h in p['pins'].items():assert sha(path)==h
 assert sha(p['source_gate']['artifact'])==p['source_gate']['sha256']
 assert len(d['rows'])==24 and len(d['guards'])==9
 assert [r['row'] for r in events if r['stage']=='case_result']==raw['rows'] and [r['row'] for r in events if r['stage']=='guard_result']==raw['guards']
 assert Counter(r['stage'] for r in events)==dict(attempt=1,case_attempt=24,case_result=24,guard_attempt=9,guard_result=9,complete=1)
 calls=0;equal=0;dims=Counter()
 inherited=unpack(json.loads(Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence/domain_sufficient/controls.json').read_text()))['rows']
 for ri,row in enumerate(d['rows']):
  fixture=row['fixture'];point_offset=0;answers=[]
  if ri<16:
   old=inherited[ri]['fixture'];assert fixture['inherited_expected']==old['expected'] and all(fixture[k]==v for k,v in old.items() if k!='expected')
   assert fixture['expected']==(True if old['id']=='lower_dimensional_linear_unknown' else old['expected'])
  for domain,rays in zip(row['result']['domains'],[fixture['rays_a'],fixture['rays_b']],strict=True):
   R=matrix(rays);n=len(R[0]);S=domain['span'];B=S['basis'];coords=S['coordinates'];dim=S['dimension'];dims[dim]+=1
   assert R==domain['rays'] and dim==rank(R)==len(B) and rank(B)==dim and len(coords)==len(R)
   chosen=[]
   for j,r in enumerate(R):
    if rank([R[i] for i in chosen]+[r])>len(chosen):chosen.append(j)
   assert S['indices']==chosen and B==[R[j] for j in chosen]
   assert all(len(z)==dim for z in coords) and [[sum(z[i]*B[i][j] for i in range(dim)) for j in range(n)] for z in coords]==R
   count=elimination_counts([list(z) for z in zip(*R)],len(R));K=len(fixture['a']['V'])
   for key in ['a','b']:
    net=fixture[key];original=dict(B=matrix(net['B']),V=matrix(net['V']),L=matrix(net.get('L',[[0]*n for _ in net['V']])))
    assert domain['original_'+key]==original
    if dim:
     projected=dict(B=[[dot(r,b) for b in B] for r in original['B']],V=original['V'],L=[[dot(r,b) for b in B] for r in original['L']]);assert domain['projected_'+key]==projected
     for op in ['add','mul']:count[op]+=(len(original['B'])+K)*dim*n
    else:assert domain['projected_'+key] is None
   if dim:
    count.update(elimination_counts([[B[i][j] for i in range(dim)]+[r[j] for r in R] for j in range(n)],dim))
    a,b=domain['projected_a'],domain['projected_b'];forms=a['B']+b['B'];V=[x+[-v for v in y] for x,y in zip(a['V'],b['V'],strict=True)];L=[[x-y for x,y in zip(u,v,strict=True)] for u,v in zip(a['L'],b['L'],strict=True)]
    count['mul']+=K*len(b['B']);count['sub']+=K*dim;groups=defaultdict(lambda:[F(0)]*K);cc=Counter()
    for i,(form,record) in enumerate(zip(forms,domain['signs'],strict=True)):
     dots=[dot(form,z) for z in coords];v=[x[i] for x in V];branch='drop' if max(dots)<=0 else 'linear' if min(dots)>=0 else 'keep'
     assert record==dict(form=form,coefficients=v,dots=dots,branch=branch)
     count['add']+=len(R)*dim;count['mul']+=len(R)*dim
     if branch=='linear':
      for k in range(K):L[k]=[x+v[k]*y for x,y in zip(L[k],form,strict=True)]
      count['mul']+=K*dim;count['add']+=K*dim
     elif branch=='keep':
      f=next(x for x in form if x);u=tuple(x/f for x in form);groups[u]=[x+abs(f)*y for x,y in zip(groups[u],v,strict=True)];cc['div']+=dim;cc['mul']+=K;cc['add']+=K
      if f<0:
       for k in range(K):L[k]=[x+v[k]*y for x,y in zip(L[k],form,strict=True)]
       cc['mul']+=K*dim;cc['add']+=K*dim
    hinges=[[list(u),v] for u,v in sorted(groups.items()) if any(v)];key=[L,hinges];assert domain['canonical']['key']==key
    scale=next((abs(x) for x in [v for r in L for v in r]+[v for u,c in hinges for v in c] if x),None)
    projective=key if scale is None else [[[x/scale for x in r] for r in L],[[u,[x/scale for x in v]] for u,v in hinges]]
    if scale is not None:cc['div']+=K*dim+K*len(hinges)
    assert domain['canonical']['projective']==projective and domain['canonical']['counts']==dict(cc)
    assert domain['sign_comparison_allowance']==2*len(forms)*len(R);yes=not hinges and not any(x for r in L for x in r)
   else:
    assert domain['canonical'] is None and not domain['signs'] and domain['sign_comparison_allowance']==0;yes=True
   assert domain['arithmetic']=={k:v for k,v in count.items() if v} and domain['certified']==yes and domain['status']==('equal' if yes else 'unequal_on_represented_hull');answers.append(yes)
   weights=[[0]*len(R)]+[[int(i==j) for i in range(len(R))] for j in range(len(R))]+[[1]*len(R),list(range(1,len(R)+1))];pts={}
   for w in weights:
    x=tuple(sum(w[i]*R[i][j] for i in range(len(R))) for j in range(n));z=tuple(sum(w[i]*coords[i][j] for i in range(len(R))) for j in range(dim));pts[x]=z
   checks=row['point_checks'][point_offset:point_offset+len(pts)];point_offset+=len(pts)
   for check,(x,z) in zip(checks,sorted(pts.items()),strict=True):
    va=evaluate(domain['original_a'],x);vb=evaluate(domain['original_b'],x);diff=[u-v for u,v in zip(va,vb,strict=True)];calls+=2
    if dim:
     qa=evaluate(domain['projected_a'],z);qb=evaluate(domain['projected_b'],z);calc=[dot(r,z)+sum(v[k]*max(F(0),dot(u,z)) for u,v in hinges) for k,r in enumerate(L)];calls+=2
    else:qa=qb=[F(0)]*K;calc=[F(0)]*K
    assert check==dict(x=list(x),z=list(z),a=va,b=vb,projected_a=qa,projected_b=qb,difference=diff,canonical=calc,passed=True) and va==qa and vb==qb and diff==calc and (not yes or not any(diff))
  assert point_offset==len(row['point_checks']) and all(answers)==fixture['expected']==row['result']['certified'] and row['passed'];equal+=all(answers)
 assert equal==19 and calls==880==d['synthetic_direct_calls'] and all(r['passed'] and r['error'] in ['ValueError','OverflowError'] for r in d['guards'])
 out=dict(complete=True,source_result_commit='12dec3ed',pins={n:sha(P/n) for n in ['controls.json','progress.jsonl','protocol.json']},checker_sha256=sha(Path(__file__)),pairs=24,guards=9,equal=19,unequal=5,domain_dimensions=dict(dims),saved_direct_calls=880,review_cpu=time.process_time()-start,author_cpu=d['cpu_seconds'],new_author_model_geometry_calls=0,limitations=['Exact represented-cone equality only; no trained domain/coverage or prediction claim.','Named rational counts verified; peak-bit ledger internals not independently replayed.','Guard outcomes retained; author invalid paths not rerun.','No timing comparison or complete-cost claim.'])
 (H/'t208_implementation_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,review_cpu=out['review_cpu'])))
if __name__=='__main__':main()
