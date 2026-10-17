import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,math,hashlib,time
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/log_cofactor_cost');start=time.process_time()
def read(p):return json.loads(p.read_text())
b,i,k=[read(P/(n+'.json')) for n in ['baseline','instrumented','kernels']];env=read(P/'frozen/det_environment.json');dispatch=read(P/'frozen/runtime_dispatch.json');counts=Counter();shapes=Counter();exact=0
for x in [b,i,k]:assert x['complete'] and len(x['rows'])==18 and x['binary_hashes']==env['binary_hashes']
for name,h in env['binary_hashes'].items():assert hashlib.sha256(Path(env['binary_paths'][name]).read_bytes()).hexdigest()==h
assert k['runtime_dispatch']['functions']==[dict(offset=int(e['table_offset'],16),target=int(e['target_relative_elf_offset'],16)) for e in dispatch['functions']];assert k['runtime_dispatch']['block_parameters']==dispatch['block_parameter_offsets']
for a,c,d in zip(b['rows'],i['rows'],k['rows']):
 assert a=={key:v for key,v in c.items() if key!='spy'}=={key:v for key,v in d.items() if key!='events'};counts.update(c['spy'])
 vals=list(map(float.fromhex,a['input_hex']))
 if all(map(math.isfinite,vals)):
  dets=[]
  for off in range(0,len(vals),9):
   x=list(map(F,vals[off:off+9]));det=x[0]*(x[4]*x[8]-x[5]*x[7])-x[1]*(x[3]*x[8]-x[5]*x[6])+x[2]*(x[3]*x[7]-x[4]*x[6]);dets.append(str(det));exact+=1
  assert dets==a['exact_determinants'] and all(v['passed'] for v in a['reference_checks'])
 for e in d['events']:
  name=e['kernel'];args=tuple(e['args']);shapes[(name,args)]+=1
  if name=='iamax':assert args in [(3,1,0,0),(2,1,0,0),(1,1,0,0)]
  elif name=='dot':assert args==(1,3,1,0)
  elif name=='gemv':assert args in [(3,0,1,1),(2,1,1,1),(1,2,1,1)]
  elif name=='scal':assert args in [(2,1,0,0),(1,1,0,0)]
  else:assert name=='swap'
assert counts['getrf_calls']==counts['m3']==counts['n3']==counts['lda3']==19 and counts['success']==17 and counts['singular']==2 and counts['invalid']==0;assert counts['log_calls']==51 and counts['exp_calls']==19
assert 4+14+6==24 and 18+29+12+4==63 and 4+6+3+4==17 and 24+63+17==104
assert 150+38001+290+176==38617 and 150+16953+784*104+129*176+290==121633
summary=read(P/'summary.json');census=read(Path('/workspace/work/recover-small_h2_56/m16/approaches/log_cofactor_review/latest/full_summary.json'))
assert summary['source_sha256']==hashlib.sha256((Path('/workspace/work/recover-small_h2_56/m16/approaches/log_cofactor_review/latest/estT_snapshot.py')).read_bytes()).hexdigest()
thin,wide=census['thin'],census['wide'];visible=thin*38151+wide*17103;native=dict(determinant3=census['scalar3det_calls']*104,determinant4=census['scalar4det_calls']*176,inverse4=census['inverse_calls']*290)
assert visible==summary['visible_expression_total'] and native==summary['native_small_subset_envelopes'];assert visible+sum(native.values())==summary['hybrid_subtotal']==2778522754;assert thin*2+wide*9==summary['model_free_runtime_constants']==190197;assert summary['old_legacy_mass']==census['legacy_ops_once'];assert summary['complete_native_upper_or_lower'] is False and summary['sampling_ratio_not_computed'] is True
out=dict(fixtures=18,matrices=19,independent_exact_determinants=exact,spy_totals=dict(counts),kernel_shapes=[dict(kernel=n,args=list(a),count=v) for (n,a),v in sorted(shapes.items())],hybrid_corpus_subtotal=2778522754,runtime_model_free_constants=190197,det3_numeric_subset=104,thin_hybrid_subtotal=38617,wide_hybrid_subtotal=121633,review_cpu=time.process_time()-start,scope='Saved transparent native parity, exact stored-input determinants and runtime dispatch/shape checks. Numeric-lane envelope depends on inherited installed-source derivations. Visible expression plus small-library subtotal is not complete compiled cost; no mass replay or sampling inference.')
Path(__file__).with_name('det3_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({key:v for key,v in out.items() if key!='kernel_shapes'}))
