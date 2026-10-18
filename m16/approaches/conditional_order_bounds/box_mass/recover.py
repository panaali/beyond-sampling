"""Decode-only recovery; reuse nine controls/mass, preserve failed attempt."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time,sys
from box import sufficient,se_threshold,I,serial,accounting
HERE=Path(__file__).resolve().parent

def decode_model_bounds(model):
 assert model['complete'] and model['model']['T']==4
 rows=model['envelope']['rows'];assert [r['age'] for r in rows]==list(range(4))
 result={}
 for row in rows:
  pairs=row['pairs'];assert len(pairs)==6
  assert {(p['a'],p['b']) for p in pairs}=={(a,b) for a in range(4) for b in range(a)}
  for p in pairs:
   raw=p['bound'];assert len(raw)==2 and all(isinstance(v,str) for v in raw)
   ints=list(map(int,raw));assert list(map(str,ints))==raw and ints[1]>0
   key=(p['a'],p['b']);value=F(*ints);assert value>=0
   result.setdefault(key,[None]*4)[3-row['age']]=value
 assert all(all(x is not None for x in L) for L in result.values())
 return result

def main():
 start=time.process_time();sys.set_int_max_str_digits(0)
 out=HERE/'recovery_results.json';progress=HERE/'recovery_progress.jsonl'
 assert not out.exists() and not progress.exists()
 R=json.loads((HERE/'recovery_protocol.json').read_text())
 for path,h in R['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 P=json.loads((HERE/'protocol.json').read_text())
 for path,h in P['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 prior=json.loads((HERE/'results.json').read_text())
 assert prior['complete'] is False and prior['models']==[]
 assert prior['failure']=="TypeError('both arguments should be Rational instances')"
 assert len(prior['control_rows'])==9 and all(r['passed'] for r in prior['control_rows'])
 data=json.loads(Path(P['trained_data']).read_text());assert data['complete']
 assert [m['name'] for m in data['models']]==['H4T4seed0','H8T4seed0']
 mass=I(F(prior['per_ordered_label_box_mass']['lower']),F(prior['per_ordered_label_box_mass']['upper']))
 result={'complete':False,'models':[],'recovery_protocol_sha256':hashlib.sha256((HERE/'recovery_protocol.json').read_bytes()).hexdigest(),
  'reused_control_rows':prior['control_rows'],'reused_mass':prior['per_ordered_label_box_mass'],
  'preserved_first_attempt_cpu_seconds':prior['process_cpu_seconds'],
  'preserved_first_attempt_accounting':prior['accounting'],
  'network_forwards':0,'checkpoint_loads':0,'reference_quadrature_calls':0,'random_draws':0,
  'repeated_controls':0,'repeated_model_checks':0,'prior_and_import_cpu_seconds':start}
 def record(row):
  with progress.open('a') as f:f.write(json.dumps(serial(row),allow_nan=False)+'\n')
 def save():
  result.update(work_cpu_seconds=time.process_time()-start,process_cpu_seconds=time.process_time(),
   total_attempt_cpu_seconds=prior['process_cpu_seconds']+time.process_time(),accounting=accounting())
  out.write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
 try:
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
  print(json.dumps([{'name':m['name'],'certified_boxes':m['certified_boxes'],
   'mass':float(m['certified_union_mass'].lo),'sampling':m['sampling_SE_threshold']} for m in result['models']]))
 except BaseException as e:
  result['failure']=repr(e);record({'stage':'failure','error':repr(e)});save();raise
if __name__=='__main__':main()
