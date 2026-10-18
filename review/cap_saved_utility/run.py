"""Fixed64 same-ray cap diagnostic; no geometry, KKT solve or mass integration."""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import hashlib,importlib.util,json,math,time,resource,statistics
import numpy as np
H=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def main():
 start=time.process_time();out=H/'results.json';progress=H/'progress.jsonl'
 assert not out.exists() and not progress.exists()
 proto=json.loads((H/'protocol.json').read_text());gate=json.loads((H/'gate.json').read_text())
 result=dict(complete=False,rows=[],protocol_sha256=sha(H/'protocol.json'),gate_sha256=sha(H/'gate.json'),new_cap_calls=0,new_KKT_solves=0,new_geometry_mass_model_reference_calls=0)
 def log(x):
  with progress.open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
 def save():
  result.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);write(out,result)
 log(dict(stage='started',gate_sha256=result['gate_sha256']))
 try:
  assert gate['complete'] and gate['protocol_sha256']==result['protocol_sha256']
  for p,h in {**proto['pins'],**gate['pins']}.items():assert sha(p)==h,p
  assert np.__version__==proto['numpy_version']
  source=Path(proto['data_root']);oldproto=json.loads((source/'pilot_protocol.json').read_text());capture=json.loads((source/'capture.json').read_text())
  certs=json.loads((source/'pilot_certificates.json').read_text());old=json.loads((source/'pilot.json').read_text())
  review=json.loads(Path(proto['certificate_review']).read_text())
  assert capture['complete'] and certs['complete'] and old['complete'] and review['complete']
  assert review['raw_sha256']==sha(source/'pilot.json') and review['certificates_sha256']==sha(source/'pilot_certificates.json') and review['protocol_sha256']==sha(source/'pilot_protocol.json')
  assert old['certificates_sha256']==sha(source/'pilot_certificates.json')
  assert certs['protocol_sha256']==old['protocol_sha256']==sha(source/'pilot_protocol.json')
  assert oldproto['indices']==proto['indices']==old['indices']==[x['index'] for x in certs['rows']]==[x['index'] for x in old['rows']]
  assert len(proto['indices'])==64 and proto['indices']==[i*(37739-1)//63 for i in range(64)]
  arrays=[]
  assert len(capture['parents'])==96 and [x['index'] for x in capture['parents']]==list(range(96))
  for row in capture['parents']:
   p=source/'parents'/row['archive'];assert sha(p)==row['sha256']
   with np.load(p,allow_pickle=False) as z:
    assert z.files==['rays'];a=z['rays']
   assert a.shape==(row['N'],4,4) and a.dtype==np.dtype('float64') and np.isfinite(a).all();arrays.append(a)
  rays=np.concatenate(arrays);assert rays.shape==(37739,4,4)
  assert hashlib.sha256(rays.tobytes(order='C')).hexdigest()==proto['ordered_ray_sha256']==oldproto['ordered_ray_sha256']
  spec=importlib.util.spec_from_file_location('t209_cap',proto['cap_source']);cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap)
  result['load_hash_cpu_seconds']=time.process_time()-start;result['loaded_ray_bytes']=rays.nbytes
  for index,certrow,oldrow in zip(proto['indices'],certs['rows'],old['rows'],strict=True):
   if time.process_time()-start>16:raise RuntimeError('16CPU soft phase cap; no automatic resume')
   log(dict(stage='row_started',index=index));R=rays[index];cert=certrow['certificate']
   assert certrow['independent_validation']['passed'] and certrow['index']==oldrow['index']==index
   row=dict(index=index,rays_hex=[[float(v).hex() for v in r] for r in R],ray_sha256=hashlib.sha256(R.tobytes(order='C')).hexdigest(),certificate=cert,original_reference=oldrow['reference'],original_reference_bound=oldrow['reference_bound'])
   result['rows'].append(row);save()
   tick=time.process_time();row['cap']=cap.bound(R.tolist());result['new_cap_calls']+=1;row['cap_cpu_seconds']=time.process_time()-tick;save()
   tick=time.process_time();a=F(cert['a']);det=F(cert['det']);assert a>0 and det!=0
   q=min(F(1),abs(det)/(108*a*a));value,conversion=cap.outward_float(q);assert 0<=q<=F(value)<=1
   row['determinant_upper']=dict(bound=str(q),outward=value,outward_hex=value.hex(),conversion=conversion)
   row['determinant_contraction_cpu_seconds']=time.process_time()-tick;save()
   reference=oldrow['reference']['value'];error=oldrow['reference_bound']['total_bound'];assert math.isfinite(reference) and reference>0 and math.isfinite(error) and error>=0
   for label,arm in [('cap',row['cap']),('determinant',row['determinant_upper'])]:
    bound=F(arm['bound']);upper=F(arm['outward']);assert 0<=bound<=upper<=1
    row[label+'_diagnostic']=dict(rational_to_stored_reference=str(bound/F(reference)),outward_to_stored_reference=str(upper/F(reference)),conditional_reference_disagreement=F(reference)-F(error)>bound)
   save();log(dict(stage='row_completed',index=index,cap_fallback=row['cap']['fallback']))
  summaries={}
  for label,key in [('cap','cap'),('determinant','determinant_upper')]:
   ratios=[F(r[label+'_diagnostic']['outward_to_stored_reference']) for r in result['rows']]
   thresholds=[]
   for text in proto['thresholds']:
    tau=F(text);chosen=[r for r in result['rows'] if F(r[key]['outward'])<=tau]
    thresholds.append(dict(threshold=text,indices=[r['index'] for r in chosen],count=len(chosen),outward_bound_sum=str(sum((F(r[key]['outward']) for r in chosen),F(0))),stored_reference_sum=str(sum((F(r['original_reference']['value']) for r in chosen),F(0)))))
   summaries[label]=dict(minimum_ratio=str(min(ratios)),median_ratio=str(statistics.median(ratios)),maximum_ratio=str(max(ratios)),conditional_reference_disagreement_indices=[r['index'] for r in result['rows'] if r[label+'_diagnostic']['conditional_reference_disagreement']],thresholds=thresholds)
  counts=Counter()
  for r in result['rows']:counts.update(r['cap']['ledger']['counts'])
  result.update(complete=True,summary=summaries,cap_macro_counts=dict(counts),cap_fallbacks=dict(Counter(r['cap']['fallback'] or 'none' for r in result['rows'])),certificate_construction_inherited=dict(review_commit='b5bb78d0',operations=review['recorded_construction_operations'],validation_operations=review['recorded_validation_operations'],peak_bits=review['recorded_peak_rational_bits']),new_determinant_contractions=64,scope=proto['scope'])
  save();log(dict(stage='completed',result_sha256=sha(out)))
  print(json.dumps(dict(complete=True,rows=len(result['rows']),cpu=result['cpu_seconds'],fallbacks=result['cap_fallbacks'],threshold_counts={k:[x['count'] for x in v['thresholds']] for k,v in summaries.items()})))
 except BaseException as e:
  result['failure']=repr(e);save();log(dict(stage='failure',error=repr(e)));raise
if __name__=='__main__':main()
