"""Independent Fraction network and whole-neighborhood certificate review."""
from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,time,sys
start=time.process_time();p=Path('/workspace/work/recover-small_h2_910/m16/approaches/rank_chart_ambiguity/results.json');d=json.loads(p.read_text());art=Path('/workspace/clean-neurons/m16/approaches/small_ladder_review/rational_geometry/geometry_h4_t4.json.gz');assert hashlib.sha256(art.read_bytes()).hexdigest()==d['source_sha256'];a=json.load(gzip.open(art,'rt'));decode=lambda q:F(int(q[0]),int(q[1]));wi=list(map(decode,a['model'][0]));wh=[list(map(decode,r)) for r in a['model'][1]];wo=[list(map(decode,r)) for r in a['model'][2]];checked=0;mixed=F(0);total=F(0)
artifact_mode=len(sys.argv)>1 and sys.argv[1]=='artifact'
if artifact_mode:
 extra=json.loads(p.with_name('artifact_results.json').read_text());assert extra['prior_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
 for item in extra['unknown_chart_results']:
  if item['status']!='mixed_open':continue
  chart=next(c for c in d['charts'] if all(c[k]==item[k] for k in ('top','second','sign_x0')))
  w=next(a['witness'] for a in item['attempts'] if a['witness'] is not None);chart['witnesses']['correct']=w;chart['status']='mixed_open'
 d['mixed_chart_mass']='63/64';d['whole_chart_uniform_coverage_ceiling']='1/64'

def dot(a,b):return sum((v*w for v,w in zip(a,b)),F(0))
for chart in d['charts']:
 top,second,sign=chart['top'],chart['second'],chart['sign_x0'];positive=F(15 if top==0 else 11 if second==0 else 3,16);mass=(positive if sign==1 else 1-positive)/12;assert mass==F(chart['analytic_mass']);total+=mass
 rows=[[F(int(q==i)-int(q==j)) for q in range(4)] for i,j in [(top,second)]+[(second,k) for k in range(4) if k not in(top,second)]]+[[F(sign*int(q==0)) for q in range(4)]]
 for kind,w in chart['witnesses'].items():
  x=list(map(F,w['x']));radius=F(w['radius_linf']);assert radius>0;assert 'coefficients' not in w or [dot(r,x) for r in rows]==list(map(F,w['coefficients']));constraints=list(rows);A=[[F(0)]*4 for _ in range(4)];history=''
  for t in range(4):
   B=[[sum(wh[i][j]*A[j][q] for j in range(4))+wi[i]*int(q==t) for q in range(4)] for i in range(4)];A=[]
   for b in B:
    active=dot(b,x)>0;history+=str(int(active))
    if any(b):constraints.append(b if active else [-v for v in b])
    A.append(b if active else [F(0)]*4)
  L=[[sum(row[j]*A[j][q] for j in range(4)) for q in range(4)] for row in wo];pred=max(range(4),key=lambda j:dot(L[j],x));assert pred==w['prediction'] and history==w['history'];assert (pred==second)==(kind=='correct')
  if kind=='correct':
   for j in range(4):
    if j==second:continue
    margin=[v-z for v,z in zip(L[second],L[j])]
    if any(margin):constraints.append(margin)
    else:assert j>second
  else:
   j=w['competitor'];margin=[v-z for v,z in zip(L[j],L[second])]
   if any(margin):constraints.append(margin)
   else:assert j<second
  assert len(constraints)==w['constraint_count']
  assert all(dot(r,x)-radius*sum(map(abs,r))>0 for r in constraints);checked+=1
 if len(chart['witnesses'])==2:assert chart['status']=='mixed_open';mixed+=mass
assert total==1 and mixed==F(d['mixed_chart_mass']) and 1-mixed==F(d['whole_chart_uniform_coverage_ceiling'])
out={'independent_open_neighborhoods':checked,'mixed_chart_mass':str(mixed),'whole_chart_uniform_coverage_ceiling':str(1-mixed),'cpu_seconds':time.process_time()-start,'scope':'whole chart constant-correctness representation only; unknown charts not declared uniform'}
Path(__file__).with_name('chart_artifact_review_results.json' if artifact_mode else 'chart_ambiguity_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
