"""Latest-source frozen full same-ray comparison; reuse266 existing actual cases."""
import time
START=time.process_time()
import argparse,json,math,resource
import numpy as np
import audit as A

def clean(x):
 if isinstance(x,float) and not math.isfinite(x):return {'nonfinite':repr(x)}
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x

def main(batch):
 assert json.loads((A.HERE/'panel.json').read_text())['finite_gate_pass']
 pins=json.loads((A.HERE.parent/'protocol.json').read_text())['pins']
 for f,h in pins.items():assert A.hashlib.sha256(A.Path(f).read_bytes()).hexdigest()==h
 base=next(A.Path(f).parent for f in pins if f.endswith('error_captured.npz'));rays=np.load(base/'error_captured.npz')['rays'];z=np.load(base/f'error_batch_{batch}.npz');ids=z['indices'].tolist();sentinels={ids[0],ids[len(ids)//2],ids[-1]}
 saved={r['index']:r for r in json.loads((A.HERE/'panel.json').read_text())['rows']}
 for r in json.loads((A.HERE/'controls.json').read_text())['rows']:
  if r['kind']=='actual_anomaly':
   idx=int(r['label'].split('_')[-1]);assert idx not in saved;saved[idx]=dict(r,index=idx)
 assert len(saved)==266
 runner=A.Run();rows=[];arrays={};out=dict(batch=batch,rows=rows,complete=False,reused=0)
 def save():
  out['cpu']=time.process_time()-START;out['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  (A.HERE/f'full_{batch}.json').write_text(json.dumps(clean(out),indent=1,allow_nan=False)+'\n')
 for idx,p,b,old in zip(ids,z['positive'],z['bounds'],z['source']):
  try:
   if idx in saved:
    r=dict(saved[idx]);a={};r['reused']=True;out['reused']+=1;assert r['reference']==p and r['reference_bound']==b
   else:r,a=runner.case(rays[idx],str(idx),p,b);r.update(index=idx,reused=False)
   r['original_source']=float(old);rows.append(r);d=r['diagnostics']
   anomaly=(not r['parity'] or not math.isfinite(r['value']) or r['value']<0 or abs(r['difference'])>1e-10+r['reference_bound'] or d.get('max_unit_diagonal_drift',0)>1e-12 or any(d.get(k,0) for k in ['raw_negative_detRt','raw_zero_detRt','angle_nonfinite','denominator_floor_hits']))
   if idx in sentinels or anomaly:
    for key,v in a.items():arrays[str(idx)+'_'+key]=v
   if not r['parity'] or not math.isfinite(r['value']):out['stop_reason']='parity/nonfinite';break
  except Exception as exc:out['stop_reason']=repr(exc);rows.append(dict(index=idx,failure=repr(exc)));break
  if len(rows)%256==0:save()
  if time.process_time()-START>50:out['stop_reason']='CPU cap';break
 out['complete']=len(rows)==len(ids) and 'stop_reason' not in out
 np.savez_compressed(A.HERE/f'full_nodes_{batch}.npz',**arrays);save();print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--batch',type=int,required=True);main(ap.parse_args().batch)
