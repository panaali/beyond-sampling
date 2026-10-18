"""Saved-only pilot verification. No author imports, solve, forward or RNG."""
from pathlib import Path
import hashlib,json,math,resource,sys,time,traceback
from helpers import design,binary32
P=Path(__file__).resolve().parent;START=time.process_time()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify(pins):
 assert isinstance(pins,dict) and pins
 for p,h in pins.items():assert sha(p)==h,p
def save(r):
 r.update(cpu_seconds=time.process_time()-START,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 f=P/'results.tmp';f.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');f.replace(P/'results.json')
def cap():assert time.process_time()-START<55,'internal reviewer cap'
def main(r):
 s=read(P/'scope.json');g=read(P/'gate.json');ci=read(P/'completed_inputs.json');rv=read(g['source_review'])
 own={str(P/n) for n in ('check.py','helpers.py','scope.json','README.md')}
 assert own<=set(rv['pins']) and own|{str(P/'completed_inputs.json'),g['source_review']}<=set(g['pins'])
 assert all(rv[k] is True for k in ('complete','pass_all','execution_authorized')) and g['ready'] is True and g['source_review_publication'] and g['execution_publication']
 for m in (s['pins'],g['pins'],ci['pins'],rv['pins']):verify(m)
 a=Path(s['author']);out=a/'output';p=read(a/'protocol.json');ag=read(a/'gate.json');ar=read(p['source_review'])
 mandatory={str(a/n) for n in ('run.py','math_core.py','launch.sh','PROTOCOL.md','protocol.json','weight_manifest.json','gate.json','owner_weights.npz','run1.log')}|{str(out/n) for n in ('results.json','progress.jsonl','solve_progress.jsonl','active_system.npz','active_solution.npz')}|{p['source_review']}
 assert mandatory<=set(ci['pins']) and ci['raw_publication']
 assert ag['ready'] is True and all(ar[k] is True for k in ('complete','pass_all','execution_authorized'))
 for m in (p['pins'],ag['pins'],ar['pins']):verify(m)
 for key,n in (('helper_review',138),('regression_review',14)):
  q=read(p[key]);assert q['complete'] is True and q['pass_all'] is True and q['total_rows']==n;verify(q['input_pins'])
 assert read(p['regression_review'])['producer_protocol_complete'] is False
 old=read(p['checkpoint_review']);oldpins=read(p['checkpoint_review_pins']);assert old['complete'] is True and old['pass_all'] is True and old['observations']==1024 and old['pins_sha256']==sha(p['checkpoint_review_pins']);verify(oldpins['pins'])
 assert resource.getrlimit(resource.RLIMIT_CPU)==(59,60) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,)*2
 import numpy as np
 assert np.__version__==p['runtime']['numpy_version'] and str(Path(np.__file__).resolve())==p['runtime']['numpy_init'] and str(Path(sys.executable).resolve())==p['runtime']['python_resolved']
 d=read(out/'results.json');assert d['complete'] is True and d['pass_all'] is True and d['pins']==ag['pins']
 assert d['limits']=={'cpu':[105,110],'AS':[2**31]*2} and d['runtime']['numpy']==np.__version__ and d['prior_regression_publication_deviation_retained'] is True
 assert 0<=d['cpu_seconds']<110 and math.isfinite(d['cpu_seconds']) and type(d['maxrss_kib'])is int and d['maxrss_kib']>0
 assert d['owner_fit_calls']==d['torch_reference_calls']==0
 cats=[('bank',j) for j in range(1,10)]+[('final',j) for j in range(12)];splits={'train':4096,'validation':1024,'evaluation':1024}
 names={'weights','wiring','validation_selection'}
 for split,n in splits.items():names|={split+'_inputs',split+'_bank_hinges',split+'_all12_endpoint',split+'_no_hinge8_endpoint'}|{split+'_forward_'+str(i) for i in range(0,n,1024)}
 for kind,j in cats:
  name=kind+'_'+str(j);names.add(name+'_model')
  names|={name+'_'+split+'_'+suffix for split in splits for suffix in ('design','prediction')}
 assert len(names)==168 and set(d['archives'])==names
 assert {str(out/(n+'.npz')) for n in names}<=set(ci['pins'])
 loaded=set();cache={}
 def load_external(path):
  with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
 def load(name):
  cap();meta=d['archives'][name];f=out/(name+'.npz');assert meta['path']==str(f) and meta['sha256']==sha(f) and meta['bytes']==f.stat().st_size
  z=load_external(f);assert set(z)==set(meta['arrays'])
  for k,v in z.items():
   q=meta['arrays'][k];assert v.dtype!=object and list(v.shape)==q['shape'] and str(v.dtype)==q['dtype'] and hashlib.sha256(v.tobytes(order='C')).hexdigest()==q['sha256'] and np.isfinite(v).all()
  loaded.add(name);return z
 def same(a,b):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 u=2**-53
 def bounded(a,b,bound):
  assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all();assert np.all(np.abs(a-b)<=bound+1e-300)
 def dotcheck(saved,x,c):
  expected=np.einsum('nd,dq->nq',x,c,optimize=False);scale=np.einsum('nd,dq->nq',abs(x),abs(c),optimize=False)
  bounded(saved,expected,8*u*x.shape[1]*scale);return expected
 wm=read(a/'weight_manifest.json');blob=Path(wm['path']).read_bytes();assert len(blob)==wm['bytes'] and hashlib.sha256(blob).hexdigest()==wm['sha256'] and hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==wm['blob']
 weights=load('weights');owner=load_external(wm['path']);canonical=load_external(p['canonical_weights'])
 assert set(weights)==set(owner)==set(canonical)=={'Wi','Wh','Wo'}
 for k,shape in [('Wi',(16,)),('Wh',(16,16)),('Wo',(10,16))]:
  assert owner[k].shape==shape and owner[k].dtype in (np.float32,np.float64);same(weights[k],owner[k].astype(np.float64));same(weights[k],canonical[k])
  assert d['weight_binding'][k]==dict(stored_dtype=str(owner[k].dtype),canonical_dtype='float64',exact_binary64_match=True)
 lib=load_external(p['wiring_library']);xold=load_external(p['wiring_inputs'])['x'];wire=load('wiring');assert xold.shape==(1024,10)
 def forward_metadata(z,n):
  assert set(z)=={'pre','hidden','logits','labels','gates'} and z['pre'].shape==z['hidden'].shape==(n,11,16) and z['logits'].shape==(n,10) and z['labels'].shape==(n,) and z['gates'].shape==(n,20)
  assert z['pre'].dtype==z['hidden'].dtype==z['logits'].dtype==np.float64 and z['labels'].dtype==np.int64 and z['gates'].dtype==np.uint8
  assert np.all(z['pre'][:,0]==0) and np.all(z['hidden'][:,0]==0)
  same(z['hidden'],np.maximum(z['pre'],0));same(z['labels'],np.argmax(z['logits'],axis=1))
  same(z['gates'],np.packbits((z['pre'][:,1:]>0).reshape(n,160),axis=1,bitorder='little'))
 forward_metadata(wire,1024)
 for k,z in [('hidden',wire['hidden'][:,-1]),('logits',wire['logits'])]:
  scale=max(1.,float(abs(lib[k]).max()));error=float(abs(z-lib[k]).max());assert d['wiring'][k]==dict(max_abs=error,scale=scale,tolerance=1e-10*scale) and error<=1e-10*scale
 assert d['wiring_label_mismatches']==[] and d['wiring_labels_equal'] is True;same(wire['labels'],lib['labels'])
 data={};ties=0;stateincs=[]
 for split,n in splits.items():
  x=load(split+'_inputs')['x'];assert x.dtype==np.float64 and x.shape==(n,10)
  st=d['streams'][split];assert st['seed']=={'train':258031,'validation':258032,'evaluation':258033}[split] and st['N']==n
  for state in (st['before'],st['after']):
   assert state['bit_generator']=='PCG64' and set(state['state'])=={'state','inc'} and all(type(v)is int and 0<=v<2**128 for v in state['state'].values())
  assert st['before']['state']['inc']==st['after']['state']['inc'] and st['before']!=st['after'];stateincs.append(st['before']['state']['inc'])
  chunks=[]
  for i in range(0,n,1024):z=load(split+'_forward_'+str(i));forward_metadata(z,1024);chunks.append(z)
  data[split]={'x':x,**{k:np.concatenate([z[k] for z in chunks]) for k in chunks[0]},'hinges':np.zeros((n,11,4),np.float32)}
  ties+=sum(len(set(row))!=10 for row in x.tolist())
 assert len(set(stateincs))==3
 # No draws or terminal PCG replay. Seed/state advancement remains source-qualified.
 assert ties==0,'preserve input ties: unique-order saved audit domain'
 score_sse=[];score_sst=[];predictions={k:[] for k in splits};maxres=0.;solvecount=0
 solvefile=(out/'solve_progress.jsonl').open();lastsolve=-1.
 def solveevent(model,event,label,arrays):
  nonlocal lastsolve
  row=json.loads(next(solvefile));assert row['model']==model and row['event']==event and row['label']==label
  assert lastsolve<=row['cpu_seconds']<=d['cpu_seconds'];lastsolve=row['cpu_seconds']
  assert row['array_hashes']=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
 assert len(d['models'])==21
 for entry,(category,index) in zip(d['models'],cats,strict=True):
  cap();name=category+'_'+str(index);step=index if category=='bank' else 10;family=5 if category=='bank' else index
  assert entry['name']==name and entry['category']==category and entry['index']==index and entry['status']=='completed'
  designs={}
  for split,n in splits.items():
   z=data[split];ds=load(name+'_'+split+'_design');assert set(ds)=={'cells','design','targets'}
   xx=z['x'].tolist();hh=z['hinges'].tolist();recs=[design(row,hrow,sorted(range(step),key=lambda k:-row[k]),step,family) for row,hrow in zip(xx,hh,strict=True)]
   cell=np.array([r0[0] for r0 in recs],np.int64);v=np.array([r0[1] for r0 in recs],np.float64)
   same(ds['cells'],cell);bounded(ds['design'],v,2e-14*np.maximum(1.,abs(v)))
   target=z['pre'][:,step,[1,6,7,8]] if category=='bank' else z['pre'][:,10,:];same(ds['targets'],target);designs[split]=ds
  ds=designs['train'];c,v,y=[ds[k] for k in ('cells','design','targets')];mdl=load(name+'_model');ids=np.array(sorted(set(c.tolist())),np.int64);same(mdl['ids'],ids)
  assert set(mdl)=={'ids','counts','grams','cross','pooled','coef'} and entry['cells']==len(ids) and entry['d']==v.shape[1] and entry['q']==y.shape[1]
  assert mdl['counts'].shape==(len(ids),) and mdl['counts'].dtype==np.int64
  assert mdl['grams'].shape==(len(ids),v.shape[1],v.shape[1]) and mdl['cross'].shape==mdl['coef'].shape==(len(ids),v.shape[1],y.shape[1]) and mdl['pooled'].shape==(v.shape[1],y.shape[1])
  for k,label in enumerate(ids):
   ix=np.flatnonzero(c==label);vv=v[ix];yy=y[ix];assert mdl['counts'][k]==len(ix)
   gg=np.einsum('ni,nj->ij',vv,vv,optimize=False);bb=np.einsum('ni,nj->ij',vv,yy,optimize=False)
   bounded(mdl['grams'][k],gg,8*u*len(ix)*np.einsum('ni,nj->ij',abs(vv),abs(vv),optimize=False));bounded(mdl['cross'][k],bb,8*u*len(ix)*np.einsum('ni,nj->ij',abs(vv),abs(yy),optimize=False))
  G=mdl['grams'].sum(axis=0);B=mdl['cross'].sum(axis=0);dim=v.shape[1]
  systems=[('pooled',G+(1e-9*np.trace(G)/dim)*np.eye(dim),B,mdl['pooled'])]
  systems += [('cell:'+str(int(label)),mdl['grams'][k]+(1e-6*(np.trace(mdl['grams'][k])/dim+1e-12))*np.eye(dim),mdl['cross'][k],mdl['coef'][k]) for k,label in enumerate(ids)]
  for label,A,b,coef in systems:
   solveevent(name,'solve_started',label,dict(matrix=A,rhs=b));solveevent(name,'solve_completed',label,dict(coefficient=coef))
   residual=np.einsum('ij,jq->iq',A,coef,optimize=False)-b;scale=np.einsum('ij,jq->iq',abs(A),abs(coef),optimize=False)+abs(b)
   rel=float(np.max(abs(residual)/np.maximum(scale,1e-300)));assert rel<=1e-8;maxres=max(maxres,rel)
  solvecount+=len(systems);assert entry['linear_attempts']==entry['linear_returns']==len(systems)
  for split,n in splits.items():
   ds=designs[split];c,v,target=[ds[k] for k in ('cells','design','targets')];pr=load(name+'_'+split+'_prediction');assert set(pr)=={'prediction','seen'} and pr['prediction'].dtype==np.float64 and pr['seen'].dtype==np.bool_
   lookup={int(label):k for k,label in enumerate(ids)};seen=np.array([int(label) in lookup for label in c]);same(pr['seen'],seen);assert entry['unseen'][split]==int((~seen).sum())
   if (~seen).any():dotcheck(pr['prediction'][~seen],v[~seen],mdl['pooled'])
   for label in set(c[seen].tolist()):
    keep=c==label;dotcheck(pr['prediction'][keep],v[keep],mdl['coef'][lookup[label]])
   if category=='bank':data[split]['hinges'][:,step,:]=np.maximum(pr['prediction'],0).astype(np.float32)
   else:
    predictions[split].append(pr['prediction'])
    if split=='validation':
     # Scalar fsum checks primitive SSE/SST; derived near-zero R2 is not compared relatively.
     means=[math.fsum(target[:,j])/n for j in range(16)]
     score_sse.append([math.fsum(float(x)**2 for x in pr['prediction'][:,j]-target[:,j]) for j in range(16)])
     score_sst.append([math.fsum((float(x)-means[j])**2 for x in target[:,j]) for j in range(16)])
  r['models_checked'].append(name);save(r)
 assert next(solvefile,None)is None;solvefile.close();assert d['solve_progress_rows']==2*solvecount and d['solve_progress_sha256']==sha(out/'solve_progress.jsonl')
 assert d['linear_attempts']==d['linear_returns']==d['active_system_writes']==d['active_solution_writes']==solvecount
 for name,h in d['active_final_hashes'].items():assert sha(out/name)==h
 lastA=load_external(out/'active_system.npz');lastC=load_external(out/'active_solution.npz');same(lastA['matrix'],systems[-1][1]);same(lastA['rhs'],systems[-1][2]);same(lastC['coefficient'],systems[-1][3])
 for split in splits:same(load(split+'_bank_hinges')['hinges'],data[split]['hinges'])
 sel=load('validation_selection');assert set(sel)=={'scores','sse','sst','choices','no_hinge_choices'}
 for key,expected in [('sse',score_sse),('sst',score_sst)]:
  ex=np.array(expected);bounded(sel[key],ex,1e-10*np.maximum(abs(ex),1e-12))
 same(sel['scores'],1-sel['sse']/np.maximum(sel['sst'],1e-300))
 for key,stop in [('choices',12),('no_hinge_choices',8)]:same(sel[key],np.array([max(range(stop),key=lambda f:sel['scores'][f,j]) for j in range(16)],np.int64))
 allmetrics={}
 for split,n in splits.items():
  z=data[split];allmetrics[split]={}
  for label,key in [('all12','choices'),('no_hinge8','no_hinge_choices')]:
   ep=load(split+'_'+label+'_endpoint');assert set(ep)=={'chosen','hidden','logits','prediction'}
   chosen=np.array([[binary32(float(predictions[split][int(sel[key][j])][i,j])) for j in range(16)] for i in range(n)],np.float32)
   same(ep['chosen'],chosen);same(ep['hidden'],np.maximum(chosen.astype(np.float64),0));independent=dotcheck(ep['logits'],ep['hidden'],weights['Wo'].T)
   same(ep['prediction'],np.argmax(ep['logits'],axis=1));same(ep['prediction'],np.argmax(independent,axis=1))
   table=[0]*5;correct=replace=agree=nearcount=neardiff=0
   for row,orig,pred in zip(z['x'].tolist(),z['labels'].tolist(),ep['prediction'].tolist(),strict=True):
    order=sorted(range(10),key=lambda k:row[k]);truth=order[-2];aa=orig==truth;bb=pred==truth;ss=orig==pred
    bucket=0 if aa and bb else 1 if aa else 2 if bb else 3 if ss else 4;table[bucket]+=1
    correct+=aa;replace+=bb;agree+=ss;rr=sorted(row);near=rr[-1]-rr[-2]<.05 or rr[-2]-rr[-3]<.05 or sum(x>0 for x in row)<=1;nearcount+=near;neardiff+=near and not ss
   metrics=dict(N=n,original_correct=correct,replacement_correct=replace,agreement=agree,disjoint_outcomes=table,near_union_count=nearcount,near_disagreement=neardiff,input_tie_rows=0)
   assert metrics==d['metrics'][split][label];allmetrics[split][label]=metrics
 assert loaded==names and sum(x['bytes'] for x in d['archives'].values())==d['archive_bytes']<=256*2**20
 calls=dict(archive=168,npz_load=4,forward=7,normal_draw=3,features=63,fit=21,predict=63,scores=12,select=1,assemble=6,metrics=6)
 assert d['calls']==d['returns']==calls and d['forward_vectors']==7168 and d['normal_values']==61440
 assert set(d['stages'])==set(calls) and all(math.isfinite(x) and 0<=x<=d['cpu_seconds'] for x in d['stages'].values())
 events=[dict(event='started'),dict(event='binding_completed')]+[dict(event='stream_completed',split=k) for k in splits]+[dict(event=e,name=kind+'_'+str(j)) for kind,j in cats for e in ('model_started','model_completed')]+[dict(event='completed')]
 last=0.;raw=[json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines()];assert len(raw)==d['events']==48
 for row,ex in zip(raw,events,strict=True):
  stamp=row.pop('cpu_seconds');assert last<=stamp<=d['cpu_seconds'];last=stamp;assert row==ex
 cap();r.update(complete=True,pass_all=True,archives=168,models=21,linear_records=solvecount,events=48,max_scaled_equation_residual=maxres,metrics=allmetrics,author_cpu=d['cpu_seconds'],retained_bytes=d['archive_bytes'],input_pins=ci['pins'],full_forward_replay=False,rng_or_seed_replay=False,refit_calls=0,producer_calls=0,prior_regression_publication_violation_retained=True)
if __name__=='__main__':
 assert not (P/'results.json').exists() and not (P/'results.tmp').exists();r=dict(complete=False,pass_all=False,models_checked=[]);save(r)
 try:main(r)
 except BaseException:r.update(complete=False,pass_all=False,failure=traceback.format_exc());raise
 finally:save(r)
