"""Fixed small regression controls; no trained checkpoint or random inputs."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,resource,sys,time,traceback
P=Path(__file__).resolve().parent;OUT=P/'output'
FILES=['core.py','fixtures.py','reference.py','run.py','launch.sh','PROTOCOL.md','protocol.json']
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def verify(pins):
 assert isinstance(pins,dict) and pins
 for f,h in pins.items():assert sha(f)==h,f

def main():
 assert not OUT.exists() or not any(OUT.iterdir());OUT.mkdir(exist_ok=True);start=time.process_time()
 r={'complete':False,'pass_all':False,'rows':[],'events':0,'attempts':{},'returns':{},'linear_attempts':0,'linear_returns':0,'reference_counts':{'systems':0,'determinants':0},'model_calls':0,'rng_calls':0,'owner_fit_calls':0}
 def raw(x):
  if isinstance(x,F):return {'fraction':[str(x.numerator),str(x.denominator)]}
  if hasattr(x,'dtype') and hasattr(x,'shape'):return {'dtype':str(x.dtype),'shape':list(x.shape),'values':raw(x.tolist())}
  if isinstance(x,dict):return {k:raw(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [raw(v) for v in x]
  if isinstance(x,float) and not math.isfinite(x):return {'nonfinite':str(x)}
  return x
 def save():
  r['cpu_seconds']=time.process_time()-start;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  tmp=OUT/'results.tmp';tmp.write_text(json.dumps(raw(r),indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'results.json')
 def cap():assert time.process_time()-start<15,'internal CPU cap'
 def event(kind,**kw):
  cap()
  with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':kind,**kw})+'\n')
  r['events']+=1;save()
 def call(row,kind,key,fun,*args):
  cap();r['attempts'][kind]=r['attempts'].get(kind,0)+1;save();value=fun(*args);r['returns'][kind]=r['returns'].get(kind,0)+1;row[key]=value;save();return value
 def emit(kind,label,a,b):
  cap();r['linear_attempts' if kind=='solve_started' else 'linear_returns']+=1
  row['solve_records'].append({'event':kind,'label':label,'a':a.copy(),'b':None if b is None else b.copy()});save()
 def flatten(x):
  if isinstance(x,list):
   for z in x:yield from flatten(z)
  else:yield x
 def norm(a):return max(sum(abs(x) for x in row) for row in a)
 try:
  save();p=read(P/'protocol.json');g=read(P/'gate.json');assert g['ready'] is True
  required=[str(P/n) for n in FILES]+[p['source_review'],p['helper_review']]+list(p['pins'])
  assert all(f in g['pins'] for f in required)
  for k in ['source_publication','review_publication','execution_scope_publication']:
   v=g[k];assert isinstance(v,str) and 8<=len(v)<=40 and all(c in '0123456789abcdef' for c in v)
  verify(p['pins']);verify(g['pins']);rv=read(p['source_review']);hr=read(p['helper_review'])
  assert rv['complete'] and rv['pass_all'] and rv['execution_authorized'];assert all(str(P/n) in rv['pins'] for n in FILES);verify(rv['pins'])
  assert hr['complete'] and hr['pass_all'] and hr['total_rows']==138;verify(hr['input_pins'])
  assert resource.getrlimit(resource.RLIMIT_CPU)==(16,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
  assert str(Path(sys.executable).resolve())==p['runtime']['python_resolved']
  import numpy as np
  assert np.__version__==p['runtime']['numpy_version'] and str(Path(np.__file__).resolve())==p['runtime']['numpy_init']
  import core,fixtures,reference
  r.update(pins=g['pins'],limits={'cpu':[16,20],'AS':[2**31,2**31]},runtime={'numpy':np.__version__,'python':sys.version});event('started')
  first_model=None
  for case in fixtures.cases():
   row={'kind':'fit','name':case['name'],'fixture':case,'status':'started','solve_records':[]};r['rows'].append(row);event('started_row',index=len(r['rows'])-1)
   c=np.array(case['ids'],dtype=np.int64);v=np.array(case['v'],dtype=np.float64);y=np.array(case['y'],dtype=np.float64)
   if case['name']=='singular_pool':
    try:call(row,'fit','model',core.fit,c,v,y,emit)
    except np.linalg.LinAlgError as exc:row['expected_failure']={'type':type(exc).__name__,'message':str(exc)};save()
    else:raise AssertionError('singular pooled solve accepted')
    assert len(row['solve_records'])==1 and row['solve_records'][0]['label']=='pooled'
    assert np.array_equal(row['solve_records'][0]['a'],np.zeros((2,2))) and np.array_equal(row['solve_records'][0]['b'],np.zeros((2,1)))
   else:
    mdl=call(row,'fit','model',core.fit,c,v,y,emit);ref=call(row,'reference_fit','reference',reference.fit,case,r['reference_counts'])
    if first_model is None:first_model=mdl
    for k in ['ids','counts','grams','cross']:
     assert mdl[k].tolist()==ref[k]
    normalized=[];residuals=[]
    for sysrow,exact,w in zip(mdl['systems'],ref['systems'],[mdl['pooled']]+list(mdl['coef']),strict=True):
     assert sysrow['label']==exact['label']
     a=exact['matrix'];b=exact['rhs'];ww=[[F.from_float(float(z)) for z in roww] for roww in w]
     err=[[sum(a[i][k]*ww[k][j] for k in range(len(a)))-b[i][j] for j in range(len(b[0]))] for i in range(len(a))]
     den=max(F(1),norm(a)*norm(ww)+norm(b));ratio=norm(err)/den;residuals.append(ratio)
    for actual,exact in [(mdl['pooled'],ref['pooled']),(mdl['coef'],ref['coef'])]:
     normalized.append(max(abs(F.from_float(float(x))-z)/max(F(1),abs(z)) for x,z in zip(actual.flat,flatten(exact),strict=True)))
    row.update(coefficient_normalized_errors=normalized,equation_residuals=residuals);save()
    assert max(normalized)<=F(1,10**6) and max(residuals)<=F(5,10**13)
    qc,qv=fixtures.queries(case);qc=np.array(qc,dtype=np.int64);qv=np.array(qv,dtype=np.float64);row['query']={'cells':qc,'design':qv};save()
    pred=call(row,'predict','prediction',core.predict,mdl,qc,qv);rp=call(row,'reference_predict','reference_prediction',reference.predict,ref,qc.tolist(),qv.tolist())
    prediction_error=max(abs(F.from_float(float(x))-z)/max(F(1),abs(z)) for x,z in zip(pred[0].flat,flatten(rp[0]),strict=True));row['prediction_normalized_error']=prediction_error;save()
    assert pred[1].tolist()==rp[1] and prediction_error<=F(1,10**6)
    text=call(row,'dumps','serialized',core.dumps,mdl);decoded=call(row,'loads','decoded',core.loads,text)
    for k in ['ids','counts']:assert np.array_equal(mdl[k],decoded[k])
    for k in ['pooled','coef']:assert np.array_equal(mdl[k].view(np.uint64),decoded[k].view(np.uint64))
    again=call(row,'predict','decoded_prediction',core.predict,decoded,qc,qv);assert all(np.array_equal(a,b) for a,b in zip(pred,again))
   row['status']='passed';event('completed_row',index=len(r['rows'])-1)
  for label in fixtures.guards():
   row={'kind':'guard','name':label,'status':'started','solve_records':[]};r['rows'].append(row);event('started_row',index=len(r['rows'])-1)
   c=np.array([0,0],dtype=np.int64);v=np.ones((2,1));y=np.ones((2,2));fun=core.fit;kind='guard_fit';args=(c,v,y,emit)
   if label=='cell_dtype':c=c.astype(np.float64)
   elif label=='design_dtype':v=v.astype(np.float32)
   elif label=='target_rows':y=y[:1]
   elif label=='nonfinite_design':v[0,0]=float('inf')
   elif label=='empty_cells':c=c[:0];v=v[:0];y=y[:0]
   args=(c,v,y,emit);row['arguments']={'cells':c,'design':v,'targets':y}
   if label=='prediction_dimension':fun=core.predict;kind='guard_predict';v=np.ones((2,2));args=(first_model,c,v);row['arguments']={'model_source':'scalar','cells':c,'design':v}
   if label in ('codec_schema','codec_duplicate_ids'):
    z={'schema':1,'d':1,'q':1,'ids':[0],'counts':[1],'pooled':[[1.]],'coef':[[[1.]]]}
    if label=='codec_schema':z['schema']=2
    else:z.update(ids=[0,0],counts=[1,1],coef=[[[1.]],[[1.]]])
    text=json.dumps(z);args=(text,);fun=core.loads;kind='guard_loads';row['arguments']={'text':text}
   save()
   try:call(row,kind,'unexpected',fun,*args)
   except ValueError as exc:row['rejection']=str(exc);save()
   else:raise AssertionError('invalid input accepted '+label)
   assert not row['solve_records'];row['status']='passed';event('completed_row',index=len(r['rows'])-1)
  assert r['attempts']=={'fit':6,'reference_fit':5,'predict':10,'reference_predict':5,'dumps':5,'loads':5,'guard_fit':5,'guard_predict':1,'guard_loads':2}
  assert r['returns']=={'fit':5,'reference_fit':5,'predict':10,'reference_predict':5,'dumps':5,'loads':5}
  assert r['linear_attempts']==15 and r['linear_returns']==14 and r['reference_counts']=={'systems':14,'determinants':74}
  assert len(r['rows'])==14 and all(z['status']=='passed' for z in r['rows'])
  event('completed');assert r['events']==30;r['complete']=r['pass_all']=True;save()
 except BaseException:
  r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
