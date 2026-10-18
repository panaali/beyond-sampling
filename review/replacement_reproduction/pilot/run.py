"""Fixed actual T258 model-binding and small reproduction resource pilot."""
from pathlib import Path
import hashlib,importlib.util,json,math,os,resource,sys,time,traceback
P=Path(__file__).resolve().parent;OUT=P/'output'
FILES=['run.py','math_core.py','launch.sh','PROTOCOL.md','protocol.json','weight_manifest.json']
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def verify(pins):
 assert isinstance(pins,dict) and pins
 for f,h in pins.items():assert sha(f)==h,f

def main():
 assert not OUT.exists() or not any(OUT.iterdir());OUT.mkdir(exist_ok=True);start=time.process_time()
 r={'complete':False,'pass_all':False,'archives':{},'calls':{},'returns':{},'linear_attempts':0,'linear_returns':0,'models':[],'stages':{},'events':0,'archive_bytes':0,'active_system_writes':0,'active_solution_writes':0,'normal_values':0,'forward_vectors':0,'owner_fit_calls':0,'torch_reference_calls':0}
 def save():
  r['cpu_seconds']=time.process_time()-start;r['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  q=OUT/'results.tmp';q.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');q.replace(OUT/'results.json')
 def cap():assert time.process_time()-start<90,'internal CPU cap'
 def event(kind,**fields):
  cap();line=json.dumps({'event':kind,'cpu_seconds':time.process_time()-start,**fields})+'\n'
  with (OUT/'progress.jsonl').open('a') as f:f.write(line)
  r['events']+=1;save()
 def begin(kind):
  cap();r['calls'][kind]=r['calls'].get(kind,0)+1;save();return time.process_time()
 def end(kind,t):
  r['returns'][kind]=r['returns'].get(kind,0)+1;r['stages'][kind]=r['stages'].get(kind,0.)+time.process_time()-t;save()
 try:
  save();p=read(P/'protocol.json');g=read(P/'gate.json');assert g['ready'] is True
  required=[str(P/n) for n in FILES]+list(p['pins'])+[p['source_review']]
  assert all(f in g['pins'] for f in required)
  for k in ['source_publication','review_publication','execution_scope_publication']:
   z=g[k];assert isinstance(z,str) and 8<=len(z)<=40 and all(c in '0123456789abcdef' for c in z)
  verify(p['pins']);verify(g['pins']);rv=read(p['source_review']);assert rv['complete'] and rv['pass_all'] and rv['execution_authorized'];assert all(str(P/n) in rv['pins'] for n in FILES);verify(rv['pins'])
  for key,n in [('helper_review',138),('regression_review',14)]:
   v=read(p[key]);assert v['complete'] and v['pass_all'] and v['total_rows']==n;verify(v['input_pins'])
  assert read(p['regression_review'])['producer_protocol_complete'] is False
  oldreview=read(p['checkpoint_review']);oldpins=read(p['checkpoint_review_pins']);assert oldreview['complete'] and oldreview['pass_all'] and oldreview['observations']==1024 and oldreview['pins_sha256']==sha(p['checkpoint_review_pins']);verify(oldpins['pins'])
  assert resource.getrlimit(resource.RLIMIT_CPU)==(105,110) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
  assert str(Path(sys.executable).resolve())==p['runtime']['python_resolved']
  import numpy as np
  assert np.__version__==p['runtime']['numpy_version'] and str(Path(np.__file__).resolve())==p['runtime']['numpy_init']
  def module(name,path):
   spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
  features=module('t258_features',p['features']);reg=module('t258_regression',p['regression']);import math_core
  r.update(pins=g['pins'],limits={'cpu':[105,110],'AS':[2**31,2**31]},runtime={'numpy':np.__version__,'python':sys.version},prior_regression_publication_deviation_retained=True);event('started')
  def archive(name,**arrays):
   cap();f=OUT/(name+'.npz');assert not f.exists();assert all(isinstance(a,np.ndarray) and a.dtype!=object for a in arrays.values())
   estimate=sum(a.nbytes for a in arrays.values())+65536;assert r['archive_bytes']+estimate<=256*2**20,'retained binary storage cap'
   t=begin('archive');tmp=f.with_suffix('.tmp')
   with tmp.open('xb') as stream:np.savez(stream,**arrays)
   tmp.replace(f);r['archive_bytes']+=f.stat().st_size
   r['archives'][name]={'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size,'arrays':{k:{'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()} for k,a in arrays.items()}};end('archive',t)
  def load_npz(path,keys):
   t=begin('npz_load')
   with np.load(path,allow_pickle=False) as z:
    assert set(z.files)==set(keys);out={k:z[k].copy() for k in keys}
   end('npz_load',t);return out
  wm=read(P/'weight_manifest.json');b=Path(wm['path']).read_bytes();assert len(b)==wm['bytes'] and hashlib.sha256(b).hexdigest()==wm['sha256'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==wm['blob']
  owner=load_npz(wm['path'],['Wi','Wh','Wo']);canonical=load_npz(p['canonical_weights'],['Wi','Wh','Wo'])
  expected={'Wi':(16,),'Wh':(16,16),'Wo':(10,16)};weights={};r['weight_binding']={}
  for k,shape in expected.items():
   a=owner[k];z=canonical[k];assert a.shape==z.shape==shape and a.dtype in (np.dtype('float32'),np.dtype('float64')) and z.dtype==np.float64 and np.isfinite(a).all()
   converted=a.astype(np.float64);assert np.array_equal(converted.view(np.uint64),z.view(np.uint64));weights[k]=converted;r['weight_binding'][k]={'stored_dtype':str(a.dtype),'canonical_dtype':str(z.dtype),'exact_binary64_match':True}
  archive('weights',**weights);wi,wh,wo=[weights[k] for k in ['Wi','Wh','Wo']]
  xold=load_npz(p['wiring_inputs'],['x'])['x'];lib=load_npz(p['wiring_library'],['hidden','logits','labels','gaps']);assert xold.shape==(1024,10) and xold.dtype==np.float64
  t=begin('forward');r['forward_vectors']+=len(xold);save();wire=math_core.forward(xold,wi,wh,wo);end('forward',t);archive('wiring',**wire)
  check={};r['wiring']=check
  for k,z in [('hidden',wire['hidden'][:,-1]),('logits',wire['logits'])]:
   assert z.shape==lib[k].shape and lib[k].dtype==np.float64;scale=max(1.,float(np.max(np.abs(lib[k]))));error=float(np.max(np.abs(z-lib[k])));check[k]={'max_abs':error,'scale':scale,'tolerance':1e-10*scale};save();assert error<=1e-10*scale
  r['wiring_label_mismatches']=np.flatnonzero(wire['labels']!=lib['labels']).tolist();save();assert not r['wiring_label_mismatches'];r['wiring_labels_equal']=True;event('binding_completed')
  data={};r['streams']={}
  for split,n,seed in [('train',4096,258031),('validation',1024,258032),('evaluation',1024,258033)]:
   t=begin('normal_draw');rng=np.random.Generator(np.random.PCG64(seed));r['streams'][split]={'seed':seed,'N':n,'before':rng.bit_generator.state};r['normal_values']+=10*n;save();x=rng.standard_normal((n,10));r['streams'][split]['after']=rng.bit_generator.state;end('normal_draw',t);archive(split+'_inputs',x=x)
   chunks=[]
   for lo in range(0,n,1024):
    t=begin('forward');r['forward_vectors']+=min(1024,n-lo);save();z=math_core.forward(x[lo:lo+1024],wi,wh,wo);end('forward',t);archive(split+'_forward_'+str(lo),**z);chunks.append(z)
   data[split]={'x':x,**{k:np.concatenate([z[k] for z in chunks],axis=0) for k in chunks[0]},'hinges':np.zeros((n,11,4),dtype=np.float32)};event('stream_completed',split=split)
  active='';solve_labels=[]
  def emit(kind,label,a,b):
   if kind=='solve_started':cap()
   field='linear_attempts' if kind=='solve_started' else 'linear_returns';r[field]+=1
   if kind=='solve_started':
    current={'matrix':a,'rhs':b};r['active_system_writes']+=1;path=OUT/'active_system.npz'
   else:current={'coefficient':a};r['active_solution_writes']+=1;path=OUT/'active_solution.npz'
   tmp=path.with_suffix('.tmp')
   with tmp.open('wb') as f:np.savez(f,**current)
   tmp.replace(path);record={'model':active,'event':kind,'label':label,'cpu_seconds':time.process_time()-start,'array_hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in current.items()}}
   with (OUT/'solve_progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   solve_labels.append((kind,label));save()
   if kind=='solve_completed':cap()
  final_predictions={k:[] for k in data};val_scores=[];val_sse=[];val_sst=[]
  for category,index in [('bank',s) for s in range(1,10)]+[('final',f) for f in range(12)]:
   active=category+'_'+str(index);step=index if category=='bank' else 10;family=5 if category=='bank' else index;entry={'name':active,'category':category,'index':index,'status':'started'};r['models'].append(entry);event('model_started',name=active)
   designs={}
   for split,z in data.items():
    t=begin('features');cells,v=features.features(z['x'],step,family,z['hinges']);end('features',t);target=z['pre'][:,step,[1,6,7,8]] if category=='bank' else z['pre'][:,10,:]
    designs[split]=(cells,v,target);archive(active+'_'+split+'_design',cells=cells,design=v,targets=target)
   cells,v,target=designs['train'];solve_labels=[];t=begin('fit');mdl=reg.fit(cells,v,target,emit);end('fit',t)
   entry.update(cells=len(mdl['ids']),d=mdl['d'],q=mdl['q'],linear_attempts=sum(k=='solve_started' for k,l in solve_labels),linear_returns=sum(k=='solve_completed' for k,l in solve_labels));save()
   assert entry['linear_attempts']==entry['linear_returns']==1+len(mdl['ids'])
   archive(active+'_model',**{k:mdl[k] for k in ['ids','counts','grams','cross','pooled','coef']})
   for split,z in data.items():
    cells,v,target=designs[split];t=begin('predict');pred,seen=reg.predict(mdl,cells,v);end('predict',t);assert np.isfinite(pred).all();archive(active+'_'+split+'_prediction',prediction=pred,seen=seen)
    entry.setdefault('unseen',{})[split]=int(np.count_nonzero(~seen));save()
    if category=='bank':z['hinges'][:,step,:]=np.maximum(pred,0.).astype(np.float32);assert np.isfinite(z['hinges']).all()
    else:
     final_predictions[split].append(pred)
     if split=='validation':
      t=begin('scores');score,sse,sst=math_core.scores(target,pred);end('scores',t);val_scores.append(score);val_sse.append(sse);val_sst.append(sst)
   entry['status']='completed';event('model_completed',name=active)
   del mdl,designs
  for split,z in data.items():archive(split+'_bank_hinges',hinges=z['hinges'])
  scores=np.stack(val_scores);t=begin('select');choices,noh=features.select(scores);end('select',t);archive('validation_selection',scores=scores,sse=np.stack(val_sse),sst=np.stack(val_sst),choices=choices,no_hinge_choices=noh)
  r['metrics']={}
  for split,z in data.items():
   pred=np.stack(final_predictions[split]);r['metrics'][split]={}
   for label,selected in [('all12',choices),('no_hinge8',noh)]:
    t=begin('assemble');out=features.assemble(pred,selected,wo);end('assemble',t);archive(split+'_'+label+'_endpoint',**out)
    t=begin('metrics');r['metrics'][split][label]=math_core.metrics(z['x'],z['labels'],out['prediction']);end('metrics',t)
  assert len(r['models'])==21 and all(m['status']=='completed' for m in r['models'])
  assert r['calls']['fit']==r['returns']['fit']==21 and r['calls']['features']==r['returns']['features']==63 and r['calls']['predict']==r['returns']['predict']==63
  assert r['calls']['forward']==r['returns']['forward']==7 and r['forward_vectors']==7168 and r['calls']['normal_draw']==r['returns']['normal_draw']==3 and r['normal_values']==61440
  assert r['linear_attempts']==r['linear_returns']==sum(1+m['cells'] for m in r['models'])
  assert r['calls']==r['returns'] and r['calls']['archive']==168 and r['calls']['npz_load']==4 and r['calls']['scores']==12 and r['calls']['metrics']==6 and r['calls']['assemble']==6 and r['calls']['select']==1;r['qualification']='Fixed small resource pilot only, new coefficients/validation choices; no historical full reproduction, uncertainty/calibration or ARC cost claim.'
  event('completed');assert r['events']==48;r['complete']=r['pass_all']=True;save()
 except BaseException:
  r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
