"""Fifteen fixed current-canonical/original-error controls; never infer old hashes."""
import time
START=time.process_time()
import hashlib,json,sys,traceback,resource,signal
from pathlib import Path
import numpy as np
import torch
from inventory import HERE,DATA,TRACER,sha
from reconstruct import mh
sys.path.insert(0,str(TRACER))
import native as N
def cap(sig,frame):raise RuntimeError('Preserved105CPU soft cap')
signal.signal(signal.SIGXCPU,cap);resource.setrlimit(resource.RLIMIT_CPU,(105,110))
def load(seed):
 meta=json.loads((HERE/'checkpoint_recovery_complete.json').read_text());assert meta['complete'];r=meta['records'][seed];assert r['validated'] and r['seed']==seed;p=Path(r['path']);assert sha(p)==r['sha256']
 state=torch.load(p,map_location='cpu',weights_only=True);keys=('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')
 assert set(state)==set(keys) and [tuple(state[k].shape) for k in keys]==[(4,1),(4,4),(3,4)]
 assert all(isinstance(w,torch.Tensor) and w.dtype==torch.float32 for w in state.values())
 model=(state[keys[0]].double().numpy()[:,0].copy(),state[keys[1]].double().numpy().copy(),state[keys[2]].double().numpy().copy());assert all(np.isfinite(w).all() for w in model)
 return model
def evaluate(model,name,instrument):
 mod,ledger,startup=N.N.load(instrument);mod.STATS={'cut':0,'resolved':0,'cells':0};ops=mod.Ops();info={};recipe=mod.RECIPE85B if name=='r85b' else mod.RECIPE112
 value=mod.est75(model,ops,info=info,**recipe)
 return dict(value=float(value),hex=float(value).hex(),legacy=ops.n,stats=mod.STATS,info=[dict(key=list(k),value=float(v)) for k,v in info['err_by'].items()],ledger=ledger.report(),startup=startup,recipe=recipe)
def main():
 target=HERE/'controls.json';assert not target.exists();gate=json.loads((HERE/'controls_gate.json').read_text());assert gate['complete']
 for p,d in gate['pins'].items():assert sha(Path(p))==d
 inventory=json.loads((HERE/'inventory.json').read_text())
 for p,d in inventory['pins'].items():assert sha(Path(p))==d
 recon=json.loads((HERE/'reconstruction.json').read_text());assert recon['complete'] and recon['models']==5120
 margin=json.loads((DATA/'frozen/margin_k28.json').read_text());rung=json.loads((DATA/'frozen/rung_k28.json').read_text())
 out=dict(started=True,complete=False,claim='06bfbc2a',driver_sha256=sha(Path(__file__)),gate_sha256=sha(HERE/'controls_gate.json'),reconstruction_sha256=sha(HERE/'reconstruction.json'),rows=[],sampler_numeric=111,sampler_legacy=72,historical_identity_verified=False,scope='15 fixed source/error parity controls, both original recipes, current canonical weights. No new truth/MC. No per-copy historical noisy cost is available.')
 def save():
  out.update(cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(target)
 save()
 try:
  for seed in range(5):
   base=load(seed);identity=recon['records'][seed];assert identity['seed']==seed and mh(base)==identity['base_sha256']
   for noise,j in [(None,None),(20,0),(23,255)]:
    if noise is None:model=base;truth=rung['seeds'][str(seed)]['clean']['truth'];index=-1;expected_hash=identity['base_sha256']
    else:
     rng=np.random.default_rng(noise)
     for _ in range(j+1):model=tuple(w+.05*rng.standard_normal(w.shape) for w in base)
     index=(noise-20)*256+j;truth=rung['seeds'][str(seed)]['panel']['truths'][index];expected_hash=identity['noises'][noise-20]['hashes'][j]
    assert mh(model)==expected_hash
    row=dict(seed=seed,noise=noise,copy=j,index=index,truth=truth,model_sha256=mh(model),recipes={});out['rows'].append(row);save()
    for name in ('r85b','r112'):
     record=dict(parity=False,old_error_match=False);row['recipes'][name]=record;save()
     baseline=evaluate(model,name,False);record['baseline']=baseline;save()
     traced=evaluate(model,name,True);record['traced']=traced;save()
     for key in ('hex','legacy','stats','info','recipe'):assert baseline[key]==traced[key],(seed,index,name,key)
     old=margin['seeds'][str(seed)];saved_error=old['clean'][name]['err'] if index<0 else old['p20'][name]['errs'][index]
     actual_error=baseline['value']-truth;record.update(saved_error=saved_error,actual_error=actual_error)
     assert actual_error==saved_error,(seed,index,name,'saved subtraction mismatch',actual_error,saved_error)
     if index<0:assert baseline['legacy']==old['clean'][name]['ops'] and baseline['stats']['cells']==old['clean'][name]['cells']
     record.update(parity=True,old_error_match=True);save()
     if time.process_time()-START>90:raise RuntimeError('Frozen90CPU controls stop')
  assert len(out['rows'])==15;out['complete']=True;save()
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
 print(json.dumps(dict(complete=True,models=len(out['rows']),estimator_calls=60,cpu=out['cpu'],rss_kib=out['rss_kib'])))
if __name__=='__main__':main()
