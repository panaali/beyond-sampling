"""Frozen256 same-ray comparison; run only after independent analytic/source gate."""
import time
START=time.process_time()
import json,math,resource
import numpy as np
import audit as A

def main():
 assert json.loads((A.HERE/'controls.json').read_text())['gate_pass'];A.verify_pins()
 base=next(A.Path(f).parent for f in A.PROTOCOL['pins'] if f.endswith('error_captured.npz'))
 rays=np.load(base/'error_captured.npz')['rays'];assert rays.shape==(35098,4,4)
 refs={}
 for i in range(8):
  z=np.load(base/f'error_batch_{i}.npz')
  for idx,p,b,s in zip(z['indices'],z['positive'],z['bounds'],z['source']):refs[int(idx)]=(float(p),float(b),float(s))
 rows=[];arrays={};out=dict(rows=rows,complete=False)
 def save():
  out['cpu']=time.process_time()-START;out['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  (A.HERE/'panel.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
 for idx in A.PROTOCOL['panel_indices']:
  p,b,old=refs[idx]
  try:
   row,a=A.check_case(rays[idx],str(idx),p,b);row.update(index=idx,old_source=old,old_difference=old-p);rows.append(row)
   for key,v in a.items():arrays[str(idx)+'_'+key]=v
  except Exception as exc:rows.append(dict(index=idx,failure=repr(exc)))
  if len(rows)%32==0:save()
  if time.process_time()-START>18:out['budget_stop']=True;break
 out['complete']=len(rows)==256;good=[r for r in rows if 'failure' not in r];d=[r['difference'] for r in good]
 out['summary']=dict(failures=len(rows)-len(good),max_abs_error=max(map(abs,d)),signed_error=math.fsum(d),L1_error=math.fsum(map(abs,d)),positive_mass=math.fsum(r['reference'] for r in good),new_mass=math.fsum(r['value'] for r in good),reference_bound_sum=math.fsum(r['reference_bound'] for r in good),old_max_abs_error=max(abs(r['old_difference']) for r in good),old_L1_error=math.fsum(abs(r['old_difference']) for r in good),negative_values=sum(r['value']<0 for r in good),cutoffs=sum(r['diagnostics']['cutoff'] for r in good),parity_failures=sum(not r['parity'] for r in good),source_cpu=math.fsum(r['source_cpu'] for r in good),diagnostic_cpu=math.fsum(r['diagnostic_cpu'] for r in good),legacy_ops=sum(r['legacy_ops'] for r in good))
 for key in ['negative_principal_minors','zero_principal_minors','product_floor_hits','denominator_floor_hits','pc_clip_hits','pc_nonfinite','scalar_3det_calls','scalar_4det_calls','inverse_calls']:
  out['summary'][key]=sum(r['diagnostics'].get(key,0) for r in good)
 out['finite_gate_pass']=out['complete'] and len(good)==256 and all(r['parity'] and np.isfinite(r['value']) and abs(r['difference'])<=1e-10+r['reference_bound'] for r in good)
 np.savez_compressed(A.HERE/'panel_nodes.npz',**arrays);save();print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
