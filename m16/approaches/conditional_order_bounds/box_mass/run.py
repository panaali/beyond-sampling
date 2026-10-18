"""Frozen controls and 72 saved pair checks; no network or mass-engine calls."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time,sys
import mpmath as mp
from box import gaussian_cdf,per_label_mass,sufficient,decode_model_bounds,se_threshold,I,serial,accounting
HERE=Path(__file__).resolve().parent

def main():
 start=time.process_time();out=HERE/'results.json';progress=HERE/'progress.jsonl'
 assert not out.exists() and not progress.exists()
 P=json.loads((HERE/'protocol.json').read_text())
 for path,h in P['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 data=json.loads(Path(P['trained_data']).read_text());assert data['complete']
 assert [m['name'] for m in data['models']]==['H4T4seed0','H8T4seed0']
 # Trusted exact fractions may be large; this affects serialization only.
 sys.set_int_max_str_digits(0)
 result={'complete':False,'control_rows':[],'models':[],
  'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),
  'prior_and_import_cpu_seconds':start,'network_forwards':0,'checkpoint_loads':0,
  'reference_quadrature_calls':0,'random_draws':0}
 def record(row):
  with progress.open('a') as f:f.write(json.dumps(serial(row),allow_nan=False)+'\n')
 def save():
  result.update(work_cpu_seconds=time.process_time()-start,process_cpu_seconds=time.process_time(),
   accounting=accounting())
  out.write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
 try:
  for name,raw,i,j,expected in P['predicate_controls']:
   record({'stage':'predicate_attempt','name':name})
   r=sufficient(list(map(F,raw)),i,j);assert r['certified_vacuous']==expected
   row={'kind':'predicate','name':name,'result':r,'expected':expected,'passed':True}
   result['control_rows'].append(row);record({'stage':'control_result','row':row});save()
  with mp.workdps(120):
   for x in P['cdf_controls']:
    record({'stage':'cdf_attempt','x':x});q=gaussian_cdf(F(x))
    ref=mp.erfc(-mp.mpf(x)/mp.sqrt(2))/2
    lo=mp.mpf(q.lo.numerator)/q.lo.denominator;hi=mp.mpf(q.hi.numerator)/q.hi.denominator
    assert lo<=ref<=hi
    assert q.hi-q.lo<F(1,10**25)
    row={'kind':'cdf','x':x,'enclosure':q,'reference120':mp.nstr(ref,110),'passed':True}
    result['control_rows'].append(row);record({'stage':'control_result','row':row});save()
  assert len(result['control_rows'])==9
  result['control_accounting']=accounting();mass=per_label_mass();result['per_ordered_label_box_mass']=mass
  result['mass_domain']='Xi in[1,2], Xj in[0,1], all other Xk<=Xj; target j; disjoint over ordered i,j.'
  for model in data['models']:
   record({'stage':'model_attempt','name':model['name']});bounds=decode_model_bounds(model)
   row={'name':model['name'],'checkpoint_sha256':model['checkpoint_sha256'],
    'reused_global_envelope_setup_counts':model['envelope']['counts'],
    'reused_global_envelope_bits':model['envelope']['bits'],'boxes':[]}
   result['models'].append(row)
   for i in range(4):
    for j in range(4):
     if i==j:continue
     record({'stage':'box_attempt','name':model['name'],'largest':i,'second':j})
     rivals=[]
     for rival in range(4):
      if rival==j:continue
      p=sufficient(bounds[(max(j,rival),min(j,rival))],i,j)
      rivals.append(dict(rival=rival,**p))
     certified=all(r['certified_vacuous'] for r in rivals)
     z={'largest':i,'second':j,'rivals':rivals,'certified_whole_box_vacuous':certified,
      'certified_undecided_mass':mass if certified else I.point(0),
      'uncertified_status':None if certified else 'sufficient whole-box test unresolved; no nonvacuity inference'}
     row['boxes'].append(z);record({'stage':'box_result','name':model['name'],'box':z});save()
   assert len(row['boxes'])==12 and sum(len(b['rivals']) for b in row['boxes'])==36
   n=sum(b['certified_whole_box_vacuous'] for b in row['boxes']);union=n*mass
   row.update(certified_boxes=n,certified_union_mass=union,
    integrated_accuracy_interval_width_lower=union.lo,sampling_SE_threshold=se_threshold(union))
   record({'stage':'model_result','row':row});save()
  assert len(result['models'])==2
  result['saved_pair_checks']=72;result['complete']=True;save()
  print(json.dumps(serial({k:v for k,v in result.items() if k not in ('models','control_rows')})))
 except BaseException as e:
  result['failure']=repr(e);record({'stage':'failure','error':repr(e)});save();raise
if __name__=='__main__':main()
