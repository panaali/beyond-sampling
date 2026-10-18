"""Same frozen256 actual cones, latest source; no reference integration."""
import time
START=time.process_time()
import json,math,resource
import numpy as np
import audit as A

def main():
 assert json.loads((A.HERE/'controls.json').read_text())['ordinary_gate_pass']
 pins=json.loads((A.HERE.parent/'protocol.json').read_text())['pins']
 for f,h in pins.items():assert A.hashlib.sha256(A.Path(f).read_bytes()).hexdigest()==h
 base=next(A.Path(f).parent for f in pins if f.endswith('error_captured.npz'));rays=np.load(base/'error_captured.npz')['rays'];old={r['index']:r for r in json.loads((A.HERE.parent/'panel.json').read_text())['rows']}
 runner=A.Run();rows=[];arrays={}
 for idx in A.P['panel_indices']:
  r=old[idx];row,a=runner.case(rays[idx],str(idx),r['reference'],r['reference_bound']);row.update(index=idx,previous_log_value=r['value'],previous_log_difference=r['difference']);rows.append(row)
  for key,v in a.items():arrays[str(idx)+'_'+key]=v
 out=dict(rows=rows,complete=len(rows)==256,cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 s=dict(max_abs_error=max(abs(r['difference']) for r in rows),L1_error=math.fsum(abs(r['difference']) for r in rows),signed_error=math.fsum(r['difference'] for r in rows),negative_values=sum(r['value']<0 for r in rows),parity_failures=sum(not r['parity'] for r in rows),thin=sum(r['diagnostics']['branch']=='duffy10' for r in rows),wide=sum(r['diagnostics']['branch']=='log128_atan2' for r in rows),source_cpu=math.fsum(r['source_cpu'] for r in rows),observer_cpu=math.fsum(r['diagnostic_cpu'] for r in rows),legacy_ops=sum(r['legacy_ops'] for r in rows),reference_bound_sum=math.fsum(r['reference_bound'] for r in rows),reference_only_bound_exceedances=sum(abs(r['difference'])>r['reference_bound'] for r in rows))
 for k in ['raw_negative_detRt','raw_zero_detRt','angle_nonfinite','denominator_floor_hits','inverse_calls','scalar3det_calls','scalar4det_calls']:s[k]=sum(r['diagnostics'].get(k,0) for r in rows)
 s['max_unit_diagonal_drift']=max(r['diagnostics'].get('max_unit_diagonal_drift',0) for r in rows)
 out['summary']=s;out['finite_gate_pass']=all(r['parity'] and math.isfinite(r['value']) and abs(r['difference'])<=1e-10+r['reference_bound'] for r in rows)
 np.savez_compressed(A.HERE/'panel_nodes.npz',**arrays);(A.HERE/'panel.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
