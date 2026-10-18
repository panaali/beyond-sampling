"""Saved-only independent rational normal-equation and routing audit; no producer imports."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,resource,struct,time,traceback
P=Path(__file__).resolve().parent;START=time.process_time()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pins(p):
 assert isinstance(p,dict) and p
 for path,h in p.items():assert sha(path)==h,path
def save(r):
 r.update(cpu_seconds=time.process_time()-START,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (P/'results.tmp').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(P/'results.tmp').replace(P/'results.json')
def cap():assert time.process_time()-START<19,'review internal CPU cap'
def shape(v):
 if not isinstance(v,list):return []
 if not v:return [0]
 s=shape(v[0]);assert all(shape(x)==s for x in v);return [len(v)]+s
def flat(v):
 if isinstance(v,list):
  for x in v:yield from flat(x)
 else:yield v
def arr(z,dtype=None,expected=None):
 assert set(z)=={'dtype','shape','values'};v=z['values'];assert z['shape']==shape(v)
 if dtype is not None:assert z['dtype']==dtype
 for x in flat(v):
  if z['dtype']=='bool':assert type(x)is bool
  elif z['dtype']=='int64':assert type(x)is int and -2**63<=x<2**63
  else:assert type(x)in(int,float) and math.isfinite(x)
 if expected is not None:assert v==expected
 return v
def rational(z):
 if isinstance(z,list):return [rational(x) for x in z]
 if isinstance(z,dict):
  assert set(z)=={'fraction'};n,d=z['fraction'];assert type(n)is str and type(d)is str and str(int(n))==n and str(int(d))==d and int(d)>0
  q=F(int(n),int(d));assert [str(q.numerator),str(q.denominator)]==[n,d];return q
 assert type(z)is int;return F(z)
def packed(v):return [struct.pack('>d',float(x)).hex() for x in flat(v)]
def solve(a,b):
 n=len(a);q=len(b[0]);z=[list(map(F,a[i]))+list(map(F,b[i])) for i in range(n)]
 for k in range(n):
  pivot=next(i for i in range(k,n) if z[i][k]);z[k],z[pivot]=z[pivot],z[k];v=z[k][k];z[k]=[x/v for x in z[k]]
  for i in range(n):
   if i!=k:
    c=z[i][k];z[i]=[x-c*y for x,y in zip(z[i],z[k],strict=True)]
 return [row[n:n+q] for row in z]
def norm(a):return max(sum(abs(x) for x in row) for row in a)
def ferr(a,b):return max(abs(F.from_float(float(x))-y)/max(F(1),abs(y)) for x,y in zip(flat(a),flat(b),strict=True))
def fixture_data():
 return [
 ('scalar',[0,0,0],[[1],[2],[3]],[[2,-1],[4,-2],[6,-3]]),
 ('two_cells',[2,2,2,9,9,9],[[1,0],[0,1],[1,1],[2,0],[0,2],[2,2]],[[1,3],[2,-1],[3,2],[4,1],[2,5],[6,6]]),
 ('collinear',[0,0,0],[[1,2],[2,4],[-1,-2]],[[1,3],[2,6],[-1,-3]]),
 ('three_cells',[-2,-2,0,0,7,7],[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]],[[1,2],[2,-1],[3,1],[3,1],[5,0],[4,3]]),
 ('zero_cell',[0,0,1,1],[[0,0],[0,0],[1,0],[0,1]],[[5,-3],[-5,2],[1,2],[2,4]]),
 ('singular_pool',[0,0],[[0,0],[0,0]],[[1],[-1]])]
def build(ids,x,y):
 labels=sorted(set(ids));d=len(x[0]);q=len(y[0]);grams=[];cross=[];counts=[]
 for label in labels:
  rows=[i for i,c in enumerate(ids) if c==label];counts.append(len(rows))
  grams.append([[sum(F(x[k][i])*x[k][j] for k in rows) for j in range(d)] for i in range(d)])
  cross.append([[sum(F(x[k][i])*y[k][j] for k in rows) for j in range(q)] for i in range(d)])
 G=[[sum(g[i][j] for g in grams) for j in range(d)] for i in range(d)];B=[[sum(b[i][j] for b in cross) for j in range(q)] for i in range(d)]
 systems=[]
 for k,(g,b) in enumerate(zip([G]+grams,[B]+cross,strict=True)):
  tr=sum(g[i][i] for i in range(d));ridge=tr/(d*10**9) if k==0 else (tr/d+F(1,10**12))/10**6
  a=[[g[i][j]+(ridge if i==j else 0) for j in range(d)] for i in range(d)]
  fr=1e-9*float(tr)/d if k==0 else 1e-6*(float(tr)/d+1e-12)
  fa=[[float(g[i][j])+(fr if i==j else 0.) for j in range(d)] for i in range(d)]
  systems.append({'label':'pooled' if k==0 else 'cell:'+str(labels[k-1]),'a':a,'b':b,'float_a':fa})
 return labels,counts,grams,cross,systems

def main(r):
 s=read(P/'scope.json');g=read(P/'gate.json');ci=read(P/'completed_inputs.json');rv=read(g['source_review']);own={str(P/n) for n in ('check.py','scope.json','README.md')}
 assert all(g[k] for k in ('ready','source_review_publication','execution_publication'))
 assert rv['complete'] is True and rv['pass_all'] is True and rv['execution_authorized'] is True
 assert own<=set(rv['pins']) and own|{str(P/'completed_inputs.json'),g['source_review']}<=set(g['pins'])
 for pm in (s['pins'],g['pins'],rv['pins'],ci['pins']):pins(pm)
 a=Path(s['author']);pr=read(a/'protocol.json');ag=read(a/'gate.json');sr=read(pr['source_review']);hr=read(pr['helper_review'])
 mandatory={str(a/n) for n in ('core.py','fixtures.py','reference.py','run.py','launch.sh','PROTOCOL.md','protocol.json','gate.json','output/results.json','output/progress.jsonl','run1.log','PUBLICATION_DEVIATION.md')}|{pr['source_review'],pr['helper_review']}
 assert mandatory<=set(ci['pins']) and ci['raw_commit'] and ci['publication']
 assert ag['ready'] is True and all(sr[k] is True for k in ('complete','pass_all','execution_authorized')) and hr['complete'] is True and hr['pass_all'] is True and hr['total_rows']==138
 for pm in (pr['pins'],ag['pins'],sr['pins'],hr['input_pins']):pins(pm)
 assert {str(a/n) for n in ('core.py','fixtures.py','reference.py','run.py','launch.sh','PROTOCOL.md','protocol.json')}<=set(sr['pins'])
 assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,)*2
 d=read(a/'output/results.json');assert d['complete'] is True and d['pass_all'] is True and d['pins']==ag['pins']
 assert d['limits']=={'cpu':[16,20],'AS':[2**31]*2} and d['runtime']['numpy']==pr['runtime']['numpy_version']
 assert math.isfinite(d['cpu_seconds']) and 0<=d['cpu_seconds']<15 and type(d['maxrss_kib'])is int and d['maxrss_kib']>0
 assert len(d['rows'])==14 and d['events']==30
 assert d['attempts']=={'fit':6,'reference_fit':5,'predict':10,'reference_predict':5,'dumps':5,'loads':5,'guard_fit':5,'guard_predict':1,'guard_loads':2}
 assert d['returns']=={'fit':5,'reference_fit':5,'predict':10,'reference_predict':5,'dumps':5,'loads':5}
 assert d['linear_attempts']==15 and d['linear_returns']==14 and d['reference_counts']=={'systems':14,'determinants':74}
 assert [d[k] for k in ('model_calls','rng_calls','owner_fit_calls')]==[0,0,0]
 systems_checked=0;det_count=0
 for row,(name,ids,x,y) in zip(d['rows'][:6],fixture_data(),strict=True):
  cap();assert row['kind']=='fit' and row['name']==name and row['status']=='passed';assert row['fixture']=={'name':name,'ids':ids,'v':x,'y':y}
  labels,counts,grams,cross,systems=build(ids,x,y);n=len(x[0]);q=len(y[0]);rec=row['solve_records']
  if name=='singular_pool':
   assert len(rec)==1 and rec[0]['event']=='solve_started' and rec[0]['label']=='pooled';arr(rec[0]['a'],'float64',[[0,0],[0,0]]);arr(rec[0]['b'],'float64',[[0],[0]])
   assert row['expected_failure']['type']=='LinAlgError' and row['expected_failure']['message']=='Singular matrix' and 'model' not in row
   r['rows'].append({'name':name,'expected_singularity':True});save(r);continue
  m=row['model'];ref=row['reference'];assert m['schema']==1 and m['d']==n and m['q']==q
  for k,want,dtype in [('ids',labels,'int64'),('counts',counts,'int64'),('grams',grams,'float64'),('cross',cross,'float64')]:
   arr(m[k],dtype,want);assert rational(ref[k])==want
  ws=[arr(m['pooled'],'float64')]+arr(m['coef'],'float64');exact=[];res=[]
  assert len(m['systems'])==len(ref['systems'])==len(systems) and len(rec)==2*len(systems)
  for k,(ss,ms,rs,w) in enumerate(zip(systems,m['systems'],ref['systems'],ws,strict=True)):
   cap();assert ms['label']==rs['label']==ss['label'];arr(ms['matrix'],'float64',ss['float_a']);arr(ms['rhs'],'float64',ss['b'])
   assert rational(rs['matrix'])==ss['a'] and rational(rs['rhs'])==ss['b']
   begin,end=rec[2*k:2*k+2];assert begin['event']=='solve_started' and end['event']=='solve_completed' and begin['label']==end['label']==ss['label']
   assert begin['a']==ms['matrix'] and begin['b']==ms['rhs'] and end['b'] is None;arr(end['a'],'float64',w)
   answer=solve(ss['a'],ss['b']);exact.append(answer);systems_checked+=1;det_count+=1+n*q
   wf=[[F.from_float(float(z)) for z in rr] for rr in w];e=[[sum(ss['a'][i][j]*wf[j][v] for j in range(n))-ss['b'][i][v] for v in range(q)] for i in range(n)]
   res.append(norm(e)/max(F(1),norm(ss['a'])*norm(wf)+norm(ss['b'])))
  assert rational(ref['pooled'])==exact[0] and rational(ref['coef'])==exact[1:]
  errors=[ferr(ws[0],exact[0]),ferr(ws[1:],exact[1:])];assert rational(row['coefficient_normalized_errors'])==errors and max(errors)<=F(1,10**6)
  assert rational(row['equation_residuals'])==res and max(res)<=F(5,10**13)
  qc=[min(ids),max(ids),min(ids)-1,max(ids)+1,min(ids)];qx=[x[0],x[-1],[1]*n,[-1]*n,[0]*n]
  arr(row['query']['cells'],'int64',qc);arr(row['query']['design'],'float64',qx)
  seen=[c in labels for c in qc];ep=[];pred=arr(row['prediction'][0],'float64');arr(row['prediction'][1],'bool',seen)
  assert len(pred)==5 and shape(pred)==[5,q]
  for i,(c,xx) in enumerate(zip(qc,qx,strict=True)):
   k=labels.index(c)+1 if c in labels else 0;ep.append([sum(F(xx[j])*exact[k][j][v] for j in range(n)) for v in range(q)])
   for v in range(q):
    terms=[F(xx[j])*F.from_float(float(ws[k][j][v])) for j in range(n)];ideal=sum(terms);err=abs(F.from_float(float(pred[i][v]))-ideal)
    assert err<=F(8,2**53)*sum(abs(t) for t in terms)
  assert rational(row['reference_prediction'][0])==ep and row['reference_prediction'][1]==seen
  assert rational(row['prediction_normalized_error'])==ferr(pred,ep)<=F(1,10**6)
  payload={'schema':1,'d':n,'q':q,'ids':labels,'counts':counts,'pooled':ws[0],'coef':ws[1:]};text=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False)
  assert row['serialized']==text;decoded=row['decoded'];assert set(decoded)==set(payload)
  for k in ('schema','d','q'):assert decoded[k]==payload[k]
  for k in ('ids','counts'):arr(decoded[k],'int64',payload[k])
  for k in ('pooled','coef'):assert packed(arr(decoded[k],'float64'))==packed(payload[k])
  assert packed(arr(row['decoded_prediction'][0],'float64'))==packed(pred);arr(row['decoded_prediction'][1],'bool',seen)
  r['rows'].append({'name':name,'systems':len(systems),'coefficient_errors':[str(z) for z in errors],'max_equation_residual':str(max(res)),'passed':True});save(r)
 labels=['cell_dtype','design_dtype','target_rows','nonfinite_design','empty_cells','prediction_dimension','codec_schema','codec_duplicate_ids']
 reasons=['nonempty int64 cells','finite float64 design','finite float64 targets','finite float64 design','nonempty int64 cells','prediction dimension','codec schema','codec ids']
 def aa(values,dtype):return {'dtype':dtype,'shape':shape(values),'values':values}
 for row,label,reason in zip(d['rows'][6:],labels,reasons,strict=True):
  cap();assert row['kind']=='guard' and row['name']==label and row['status']=='passed' and row['rejection']==reason and row['solve_records']==[] and 'unexpected' not in row
  c=aa([0,0],'int64');v=aa([[1.],[1.]],'float64');y=aa([[1.,1.],[1.,1.]],'float64')
  if label=='cell_dtype':c=aa([0.,0.],'float64')
  elif label=='design_dtype':v['dtype']='float32'
  elif label=='target_rows':y=aa([[1.,1.]],'float64')
  elif label=='nonfinite_design':v['values'][0][0]={'nonfinite':'inf'}
  elif label=='empty_cells':c=aa([],'int64');v={'dtype':'float64','shape':[0,1],'values':[]};y={'dtype':'float64','shape':[0,2],'values':[]}
  expected={'cells':c,'design':v,'targets':y}
  if label=='prediction_dimension':expected={'model_source':'scalar','cells':c,'design':aa([[1.,1.],[1.,1.]],'float64')}
  if label.startswith('codec_'):
   z={'schema':2 if label=='codec_schema' else 1,'d':1,'q':1,'ids':[0],'counts':[1],'pooled':[[1.]],'coef':[[[1.]]]}
   if label=='codec_duplicate_ids':z.update(ids=[0,0],counts=[1,1],coef=[[[1.]],[[1.]]])
   expected={'text':json.dumps(z)}
  assert json.dumps(row['arguments'],sort_keys=True)==json.dumps(expected,sort_keys=True)
  r['rows'].append({'name':label,'saved_rejection_checked':True});save(r)
 events=[{'event':'started'}]
 for i in range(14):events.extend([{'event':'started_row','index':i},{'event':'completed_row','index':i}])
 events.append({'event':'completed'});assert [json.loads(v) for v in (a/'output/progress.jsonl').read_text().splitlines()]==events
 assert systems_checked==14 and det_count==74;cap()
 r.update(complete=True,pass_all=True,total_rows=14,events=30,independent_exact_systems=systems_checked,producer_determinant_calls=det_count,author_cpu_seconds=d['cpu_seconds'],producer_calls=0,model_fit_rng_calls=0,producer_gate_publication_before_run=False,producer_protocol_complete=False,publication_deviation_path=str(a/'PUBLICATION_DEVIATION.md'),input_pins=ci['pins'],scope_sha256=sha(P/'scope.json'))
if __name__=='__main__':
 assert not (P/'results.json').exists() and not (P/'results.tmp').exists();r={'complete':False,'pass_all':False,'rows':[]};save(r)
 try:main(r)
 except BaseException:r.update(complete=False,pass_all=False,failure=traceback.format_exc());raise
 finally:save(r)
