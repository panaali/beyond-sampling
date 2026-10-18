"""Independent exact record contraction; imports no author primitive."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence/domain_sufficient');H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(x):
 if isinstance(x,list):
  if len(x)==2 and all(isinstance(a,str) for a in x):return F(int(x[0]),int(x[1]))
  return [unpack(a) for a in x]
 if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
 return x
def matrix(m):return [[F(x) for x in r] for r in m]
def dot(a,b):return sum(x*y for x,y in zip(a,b,strict=True))
def evaluate(net,x):
 vals=[max(F(0),dot(matrix(net['B'])[i],x)) for i in range(len(net['B']))]
 return [dot(row,vals)+dot(l,x) for row,l in zip(matrix(net['V']),matrix(net.get('L',[[0]*len(x) for _ in net['V']])),strict=True)]
def main():
 start=time.process_time();raw=json.loads((P/'controls.json').read_text());d=unpack(raw);p=json.loads((P/'protocol.json').read_text());events=[json.loads(x) for x in (P/'progress.jsonl').read_text().splitlines()]
 assert d['complete'] and d['all_pass'] and d['protocol_sha256']==sha(P/'protocol.json')
 for f,h in p['pins'].items():assert sha(f)==h
 assert len(d['rows'])==16 and len(d['guards'])==4 and d['cases']==20
 assert [x['row'] for x in events if x['stage']=='case_result']==raw['rows'] and [x['row'] for x in events if x['stage']=='guard_result']==raw['guards']
 assert Counter(x['stage'] for x in events)==dict(phase_attempt=1,case_attempt=16,case_result=16,guard_attempt=4,guard_result=4,complete=1)
 evaluations=0;certified=0
 for row in d['rows']:
  c=row['fixture'];a,b=c['a'],c['b'];allcert=[];point_offset=0
  for domain,rays in zip(row['result']['domains'],[c['rays_a'],c['rays_b']],strict=True):
   R=matrix(rays);dim=len(R[0]);K=len(a['V']);forms=matrix(a['B'])+matrix(b['B']);coeff=[list(x)+[-v for v in y] for x,y in zip(matrix(a['V']),matrix(b['V']),strict=True)]
   linear=[[x-y for x,y in zip(u,v,strict=True)] for u,v in zip(matrix(a.get('L',[[0]*dim for _ in range(K)])),matrix(b.get('L',[[0]*dim for _ in range(K)])),strict=True)]
   assert domain['rays']==R
   retained=[];vectors=[];nlinear=0
   for i,(form,r) in enumerate(zip(forms,domain['hinges'],strict=True)):
    dots=[dot(form,x) for x in R];v=[coeff[k][i] for k in range(K)];branch='drop' if max(dots)<=0 else 'linear' if min(dots)>=0 else 'keep'
    assert r==dict(form=form,coefficients=v,dots=dots,branch=branch)
    if branch=='linear':
     nlinear+=1
     for k in range(K):linear[k]=[x+v[k]*y for x,y in zip(linear[k],form,strict=True)]
    elif branch=='keep':retained.append(form);vectors.append(v)
   assert domain['simplified']==dict(B=retained,V=[[v[k] for v in vectors] for k in range(K)],L=linear)
   counts=Counter(mul=K*len(b['B'])+len(forms)*len(R)*dim+nlinear*K*dim,sub=K*dim,add=len(forms)*len(R)*dim+nlinear*K*dim)
   assert {k:v for k,v in counts.items() if v}==domain['restriction_counts'] and domain['sign_comparison_allowance']==2*len(forms)*len(R)
   groups=defaultdict(lambda:[F(0)]*K);canon_counts=Counter();L=[x[:] for x in linear]
   for form,v in zip(retained,vectors,strict=True):
    scale=next(x for x in form if x);u=tuple(x/scale for x in form);canon_counts['div']+=dim;canon_counts['mul']+=K;canon_counts['add']+=K
    groups[u]=[x+abs(scale)*y for x,y in zip(groups[u],v,strict=True)]
    if scale<0:
     canon_counts['mul']+=K*dim;canon_counts['add']+=K*dim
     for k in range(K):L[k]=[x+v[k]*y for x,y in zip(L[k],form,strict=True)]
   hinges=[[list(u),v] for u,v in sorted(groups.items()) if any(v)];key=[L,hinges]
   assert domain['canonical']['key']==key
   scalars=[x for r in L for x in r]+[x for u,v in hinges for x in v];scale=next((abs(x) for x in scalars if x),None)
   projected=key if scale is None else [[[x/scale for x in r] for r in L],[[u,[x/scale for x in v]] for u,v in hinges]]
   if scale is not None:canon_counts['div']+=K*dim+K*len(hinges)
   assert domain['canonical']['projective']==projected and dict(canon_counts)==domain['canonical']['counts']
   yes=not hinges and not any(x for r in L for x in r);assert domain['certified']==yes and domain['status']==('certified_equal' if yes else 'unknown');allcert.append(yes)
   pts={tuple(F(0) for _ in range(dim))}|{tuple(r) for r in R}|{tuple(sum(r[j] for r in R) for j in range(dim)),tuple(sum((i+1)*r[j] for i,r in enumerate(R)) for j in range(dim))}
   checks=row['point_checks'][point_offset:point_offset+len(pts)];point_offset+=len(pts)
   assert [x['x'] for x in checks]==[list(x) for x in sorted(pts)]
   for check in checks:
    x=check['x'];va,vb=evaluate(a,x),evaluate(b,x);diff=[u-v for u,v in zip(va,vb,strict=True)];calc=[dot(r,x)+sum(v[k]*max(F(0),dot(u,x)) for u,v in hinges) for k,r in enumerate(L)]
    assert check==dict(x=x,a=va,b=vb,difference=diff,restricted=calc,passed=True) and diff==calc and (not yes or not any(diff));evaluations+=2
  assert point_offset==len(row['point_checks']) and row['result']['certified']==all(allcert)==c['expected'] and row['passed'];certified+=all(allcert)
 assert certified==12 and evaluations==d['synthetic_direct_evaluations']==312
 assert d['guards']==[dict(id=n,passed=True,error=e) for n,e in [('empty_hull','ValueError'),('ragged_hull','ValueError'),('coefficient_width','ValueError'),('nonfinite','OverflowError')]]
 out=dict(complete=True,source_result_commit='f557b774',pins={n:sha(P/n) for n in ['controls.json','progress.jsonl','protocol.json']},checker_sha256=sha(Path(__file__)),cases=16,guards=4,certified=12,unknown=4,recorded_synthetic_direct_evaluations=312,review_cpu=time.process_time()-start,author_cpu=d['cpu_seconds'],new_author_trained_geometry_mass_calls=0,limitations=['Sufficient ambient canonical test only; lower-dimensional true equality can remain unknown.','Exact positive hulls are represented domains, not certified neural-history feasibility or coverage.','Restriction/canonical named operation totals checked; observed peak-bit instrumentation not replayed or upgraded to all temporary integers.','Malformed input rejection is checked from saved outcomes and source paths, without rerunning author functions.','No probability, full-cost, label-equivalence necessity or trained application conclusion.'])
 (H/'domain_sufficient_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,review_cpu=out['review_cpu'])))
if __name__=='__main__':main()
