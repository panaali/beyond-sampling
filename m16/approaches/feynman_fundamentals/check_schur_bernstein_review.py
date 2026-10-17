"""Independent exact inverse/cofactor identities and frozen saved-panel audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import importlib.util,json,hashlib,time,math
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_910/m16/approaches/schur_bernstein_rescue');spec=importlib.util.spec_from_file_location('candidate_schur',P/'kernel.py');K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
def inverse_det(R):
 a=[list(row)+[F(i==j) for j in range(4)] for i,row in enumerate(R)];det=F(1)
 for j in range(4):
  k=next(k for k in range(j,4) if a[k][j])
  if k!=j:a[k],a[j]=a[j],a[k];det=-det
  p=a[j][j];det*=p;a[j]=[v/p for v in a[j]]
  for i in range(4):
   if i!=j:
    p=a[i][j];a[i]=[v-p*w for v,w in zip(a[i],a[j])]
 return [row[4:] for row in a],det
checks=0
for seed in range(8):
 A=[[F(((i+2)*(j+3)+seed*(i-j+1))%11-5,7) for j in range(4)] for i in range(4)];C=[[sum(A[i][k]*A[j][k] for k in range(4))+int(i==j) for j in range(4)] for i in range(4)];coeff=K.exact_coefficients(C)
 for t in map(F,[0,F(1,7),F(3,5),1]):
  R=[[F(i==j)*(1-t)+t*C[i][j] for j in range(4)] for i in range(4)];inv,det=inverse_det(R);v,p=K.polynomials(coeff,t)
  assert v==[det*inv[i][i] for i in range(4)];assert p==[-det*inv[k][l] for _,_,k,l in K.PAIRS];checks+=10
controls=json.loads((P/'controls.json').read_text());assert controls['exact_identities']==checks==320 and controls['gate_pass'];assert all(r['nodes']==120 and not r['failures'] and r['max_exact_pc_error']<=1e-12 for r in controls['near_singular'])
d=json.loads((P/'panel.json').read_text());assert d['complete'] and len(d['rows'])==256;assert d['indices']==[j*35097//255 for j in range(256)]
for p,sha in d['hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
rows=d['rows'];assert [r['index'] for r in rows]==d['indices'];assert not any('failure' in r for r in rows);assert sum(r['diagnostics'].get('excursions',0) for r in rows)==0
diffs=[F(r['candidate'])-F(r['source']) for r in rows];l1=sum(map(abs,diffs),F(0));maxdiff=max(map(abs,diffs));s=d['summary'];assert abs(float(l1)-s['candidate_source_L1'])<1e-25 and float(maxdiff)==s['candidate_source_max'];assert maxdiff<F(1,10**10)
for k,field in [('candidate_positive_L1','candidate'),('source_positive_L1','source')]:
 v=float(sum((abs(F(r[field])-F(r['positive'])) for r in rows),F(0)));assert abs(v-s[k])<1e-20
assert sum(r['candidate']<0 for r in rows)==sum(r['source']<0 for r in rows)==3
out=dict(independent_exact_inverse_identities=checks,frozen_panel=256,failures=0,excursions=0,integrated_candidate_source_max=float(maxdiff),integrated_candidate_source_L1=float(l1),candidate_reference_L1=s['candidate_positive_L1'],source_reference_L1=s['source_positive_L1'],negative_candidate_and_source=3,maximum_node_correlation_difference=max(r.get('diagnostic_max_pc_difference',0) for r in rows),candidate_cpu=s['candidate_cpu'],source_cpu=s['source_cpu'],exact_setup_cpu=s['setup_cpu'],review_cpu=time.process_time()-start,scope='Exact cofactor algebra and frozen saved-panel consistency, no full corpus or pointwise floating guarantee. One ordered timing is not a speedup claim; exact Fraction preparation cost remains explicit.')
(H/'schur_bernstein_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
