"""Frozen full same-ray audit; reuse completed256, amortize universal AST/grid setup."""
import time
START=time.process_time()
import argparse,json,math,resource
import numpy as np
import audit as A

class Run:
 def __init__(self):
  self.obs=A.Observer();self.plain=A.engine();self.observed=A.engine(self.obs)
 def case(self,R,label,p,b):
  obs=self.obs;obs.dets.clear();obs.clips.clear();obs.floors.clear();obs.inv_shapes.clear();a=A.Ops();o=A.Ops()
  t=time.process_time();v=self.plain(R,a);cpu=time.process_time()-t
  t=time.process_time();w=self.observed(R,o);dcpu=time.process_time()-t
  diag,arrays=obs.diagnostics()
  return dict(label=label,value=float(v),reference=float(p),reference_bound=float(b),difference=float(v-p),parity=np.float64(v).tobytes()==np.float64(w).tobytes(),legacy_ops=a.n,observed_ops=o.n,source_cpu=cpu,diagnostic_cpu=dcpu,diagnostics=diag),arrays

def controls():
 A.verify_pins();runner=Run();old=json.loads((A.HERE/'controls.json').read_text());cases=[]
 import itertools
 cs=[np.eye(4)]+[(.5*np.eye(4)+.5*np.ones((4,4)))*np.outer(s,s) for s in itertools.product([-1,1],repeat=4)]
 rays=[np.linalg.inv(np.linalg.cholesky(c)).T for c in cs]
 fix=json.loads(next(A.Path(f) for f in A.PROTOCOL['pins'] if f.endswith('negative_mass_fixture.json')).read_text());rays.append(np.array(fix['rays']))
 for R,r in zip(rays,old['rows']):
  new,a=runner.case(R,r['label'],r['reference'],r['reference_bound'])
  cases.append(dict(label=r['label'],value_equal=new['value']==r['value'],diagnostics_equal=new['diagnostics']==r['diagnostics'],counter_equal=new['legacy_ops']==r['legacy_ops'],observer_parity=new['parity']))
 out=dict(rows=cases,gate_pass=all(all(v for k,v in r.items() if k!='label') for r in cases),cpu=time.process_time()-START)
 (A.HERE/'full_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

def main(batch):
 assert json.loads((A.HERE/'full_controls.json').read_text())['gate_pass'];A.verify_pins()
 base=next(A.Path(f).parent for f in A.PROTOCOL['pins'] if f.endswith('error_captured.npz'))
 z=np.load(base/f'error_batch_{batch}.npz');rays=np.load(base/'error_captured.npz')['rays'];ids=z['indices'].tolist();sentinel={ids[0],ids[len(ids)//2],ids[-1]}
 saved={r['index']:r for r in json.loads((A.HERE/'panel.json').read_text())['rows']};runner=Run();rows=[];arrays={};out=dict(batch=batch,rows=rows,complete=False,reused=0)
 def save():
  out['cpu']=time.process_time()-START;out['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  (A.HERE/f'full_{batch}.json').write_text(json.dumps(out,indent=1,allow_nan=False)+'\n')
 for idx,p,b,old in zip(ids,z['positive'],z['bounds'],z['source']):
  try:
   if idx in saved:
    row=dict(saved[idx]);row['reused_from_panel']=True;arrays_here={};out['reused']+=1
    assert row['reference']==p and row['reference_bound']==b and row['old_source']==old
   else:
    row,arrays_here=runner.case(rays[idx],str(idx),p,b);row.update(index=idx,old_source=float(old),old_difference=float(old-p),reused_from_panel=False)
   rows.append(row);d=row['diagnostics']
   anomaly=(not row['parity'] or not math.isfinite(row['value']) or row['value']<0 or abs(row['difference'])>1e-10+row['reference_bound'] or d['cutoff'] or any(d.get(k,0) for k in ['negative_principal_minors','zero_principal_minors','product_floor_hits','denominator_floor_hits','pc_clip_hits','pc_nonfinite']))
   if idx in sentinel or anomaly:
    for key,v in arrays_here.items():arrays[str(idx)+'_'+key]=v
   if not row['parity'] or not math.isfinite(row['value']):out['stop_reason']='parity/nonfinite';break
  except Exception as e:rows.append(dict(index=idx,failure=repr(e)));out['stop_reason']='exception';break
  if len(rows)%256==0:save()
  if time.process_time()-START>50:out['stop_reason']='bounded CPU cap';break
 out['complete']=len(rows)==len(ids) and 'stop_reason' not in out
 np.savez_compressed(A.HERE/f'full_nodes_{batch}.npz',**arrays);save()
 print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--controls',action='store_true');ap.add_argument('--batch',type=int);args=ap.parse_args()
 if args.controls:controls()
 else:main(args.batch)
