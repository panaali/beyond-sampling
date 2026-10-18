"""Six unattempted guards only; preserve prior capped attempt unchanged."""
from pathlib import Path
import hashlib,json,math,resource,sys,time,traceback
P=Path(__file__).resolve().parent;OLD=P.parent;OUT=P/'output'
FILES=['run.py','launch.sh','PROTOCOL.md','protocol.json']
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def verify(pins):
 assert isinstance(pins,dict) and pins
 for f,h in pins.items():assert sha(f)==h,f

def main():
 assert not OUT.exists() or not any(OUT.iterdir());OUT.mkdir(exist_ok=True);start=time.process_time()
 r={'complete':False,'pass_all':False,'rows':[],'guard_attempts':0,'events':0,'candidate_pairs_repeated':0,'reference_calls':0,'fit_calls':0,'model_calls':0,'rng_calls':0}
 def raw(x):
  if hasattr(x,'dtype') and hasattr(x,'shape'):return {'dtype':str(x.dtype),'shape':list(x.shape),'values':raw(x.tolist())}
  if isinstance(x,dict):return {k:raw(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [raw(v) for v in x]
  if isinstance(x,float) and not math.isfinite(x):return {'nonfinite':str(x)}
  return x
 def save():
  r['cpu_seconds']=time.process_time()-start;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  q=OUT/'results.tmp';q.write_text(json.dumps(raw(r),indent=2,allow_nan=False)+'\n');q.replace(OUT/'results.json')
 def event(kind,**kw):
  assert time.process_time()-start<15,'internal CPU cap'
  with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':kind,**kw})+'\n')
  r['events']+=1;save()
 try:
  save();p=read(P/'protocol.json');g=read(P/'gate.json');assert g['ready'] is True
  required=[str(P/n) for n in FILES]+[p['review']]+list(p['pins'])
  assert all(f in g['pins'] for f in required)
  for k in ['source_publication','review_publication','execution_scope_publication']:
   v=g[k];assert isinstance(v,str) and 8<=len(v)<=40 and all(c in '0123456789abcdef' for c in v)
  verify(p['pins']);verify(g['pins']);rv=read(p['review'])
  assert rv['complete'] and rv['pass_all'] and rv['execution_authorized'];assert all(str(P/n) in rv['pins'] for n in FILES);verify(rv['pins'])
  oldp=read(OLD/'protocol.json');oldg=read(OLD/'gate.json');verify(oldp['pins']);verify(oldg['pins'])
  for key in ['written_review','source_review']:verify(read(oldp[key])['pins'])
  assert resource.getrlimit(resource.RLIMIT_CPU)==(16,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
  assert str(Path(sys.executable).resolve())==oldp['runtime']['python_resolved']
  prior=read(OLD/'output/results.json');events=[json.loads(z) for z in (OLD/'output/progress.jsonl').read_text().splitlines()]
  assert prior['complete'] is False and prior['pass_all'] is False and 'internal CPU cap' in prior['failure']
  assert len(prior['rows'])==132 and len(events)==prior['events']==264 and prior['guard_attempts']==4
  assert all(prior[k]==128 for k in ['candidate_attempts','candidate_completed','reference_attempts','reference_completed'])
  assert all(z['status']=='passed' for z in prior['rows'])
  assert [(z['t'],z['family']) for z in prior['rows'][:120]]==[(t,f) for t in range(1,11) for f in range(12)]
  assert [z['label'] for z in prior['rows'][120:124]]==['all_tie','increasing','column_ties','negative']
  assert [z['label'] for z in prior['rows'][124:128]]==['zero','signed','readout_tie','float32_boundary']
  assert [z['label'] for z in prior['rows'][128:]]==['x_dtype','x_shape','x_infinite','t_zero']
  assert events[0]=={'event':'started'}
  expected=[{'event':'started'}]
  for i in range(132):
   expected.append({'event':'started_row','index':i})
   if i<131:expected.append({'event':'completed_row','index':i})
  assert events==expected
  import numpy as np
  assert np.__version__==oldp['runtime']['numpy_version'] and str(Path(np.__file__).resolve())==oldp['runtime']['numpy_init']
  sys.path.insert(0,str(OLD));import core
  xrec=prior['fixed_inputs']['x'];hrec=prior['fixed_inputs']['hinges']
  assert xrec['dtype']=='float64' and xrec['shape']==[12,10] and hrec['dtype']=='float32' and hrec['shape']==[12,11,4]
  x=np.array(xrec['values'],dtype=np.float64);h=np.array(hrec['values'],dtype=np.float32)
  r.update(pins=g['pins'],limits={'cpu':[16,20],'AS':[2**31,2**31]},prior_cpu_seconds=prior['cpu_seconds'],prior_rows=132,prior_events=264,prior_missing_events=['completed_row index131','original completed'],reuse={'candidate_pairs':128,'guards':4,'source':str(OLD/'output/results.json')})
  event('started')
  labels=['t_boolean','family_range','hinge_shape','hinge_dtype','score_nonfinite','choice_range']
  for label in labels:
   xx=x.copy();hh=h.copy();tt=10;ff=0;fun=core.features
   if label=='t_boolean':tt=True
   elif label=='family_range':ff=12
   elif label=='hinge_shape':hh=hh[:,:10]
   elif label=='hinge_dtype':hh=hh.astype(np.float64)
   args=(xx,tt,ff,hh)
   if label=='score_nonfinite':fun=core.select;scores=np.zeros((12,16));scores[0,0]=float('nan');args=(scores,)
   if label=='choice_range':fun=core.assemble;c=np.zeros(16,dtype=np.int64);c[0]=12;args=(np.zeros((12,1,16)),c,np.zeros((10,16)))
   row={'kind':'guard','label':label,'arguments':args,'status':'started'};r['rows'].append(row);event('started_row',index=len(r['rows'])-1);r['guard_attempts']+=1;save()
   try:fun(*args)
   except ValueError as exc:row['rejection']=str(exc);save()
   else:raise AssertionError('accepted invalid input '+label)
   row['status']='passed';event('completed_row',index=len(r['rows'])-1)
  assert r['guard_attempts']==len(r['rows'])==6 and all(z['status']=='passed' for z in r['rows'])
  assert [z['label'] for z in prior['rows'][128:]+r['rows']]==['x_dtype','x_shape','x_infinite','t_zero']+labels
  r['combined_coverage']={'candidate_attempts':128,'candidate_completed':128,'reference_attempts':128,'reference_completed':128,'guard_attempts':10,'rows':138,'old_events':264,'new_events':14,'old_complete':False,'qualification':'Two distinct progress streams; historical missing events are not reconstructed.'}
  event('completed');assert r['events']==14
  r['complete']=r['pass_all']=True;save()
 except BaseException:
  r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
