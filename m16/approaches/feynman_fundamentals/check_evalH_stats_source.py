"""Independent read-only source gate; never imports or executes author loops."""
import ast, hashlib, json, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_910')
D=ROOT/'review/evalH_stats_review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(x):return ast.dump(x,include_attributes=False)
def main():
 start=time.process_time();p=json.loads((D/'protocol.json').read_text());checked={}
 for path,h in p['pins'].items():
  path=Path(path);assert sha(path)==h,str(path);checked[str(path)]=h
 inv=json.loads((D/'inventory.json').read_text())
 # Freeze original code and native statistics semantics; mutable report prose is outside this gate.
 for name in ('evalH.py','est75.py','reportH.py'):
  rel='m16/approaches/understand_m43/'+name;assert sha(ROOT/rel)==inv['pins'][rel];checked[rel]=sha(ROOT/rel)
 old=(D/'frozen/evalH.py').read_text();fixed=(D/'evalH_fixed.py').read_text()
 line='            base_stats = dict(E75.STATS)\n';assert fixed.count(line)==1
 restored=fixed.replace(line,'')
 for field in ('cells','cut','resolved'):
  token='base_stats["'+field+'"]';assert restored.count(token)==1;restored=restored.replace(token,'E75.STATS["'+field+'"]')
 assert restored==old
 oldtree=ast.parse(old);newtree=ast.parse(fixed)
 def loop(tree):
  main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
  matches=[x for x in ast.walk(main) if isinstance(x,ast.For) and ast.unparse(x.target)=='(i, m)' and dump(x.iter)==dump(ast.parse('enumerate(models)',mode='eval').body)]
  assert len(matches)==1;return matches[0]
 ol,nl=loop(oldtree),loop(newtree)
 assert len(nl.body)==len(ol.body)+1
 snapshot=next(i for i,x in enumerate(nl.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='base_stats' for t in x.targets))
 assert isinstance(nl.body[snapshot+1],ast.If)
 assert dump(nl.body[snapshot].value)==dump(ast.parse('dict(E75.STATS)',mode='eval').body)
 expected=[('clean_only',1,False,False),('no_noisy_retest',8,False,False),('one_noisy_retest',9,False,False),('two_noisy_retests',17,False,False),('zero_counts',1,True,False),('nonzero_invariance_difference',17,False,True)]
 assert [(c['name'],c['N'],c['zero'],c['changed_output']) for c in p['cases']]==expected
 ctl=ast.parse((D/'controls.py').read_text());imports=[]
 for n in ast.walk(ctl):
  if isinstance(n,ast.Import):imports += [a.name for a in n.names]
  elif isinstance(n,ast.ImportFrom):imports.append(n.module)
 assert set(imports)=={'ast','hashlib','json','resource','signal','time','types','pathlib'}
 native=ast.parse((ROOT/'m16/approaches/understand_m43/est75.py').read_text())
 changes=[n for n in ast.walk(native) if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Subscript) and isinstance(n.target.value,ast.Name) and n.target.value.id=='STATS']
 assert len(changes)==9 and all(isinstance(n.op,ast.Add) for n in changes)
 native_calls=[n for n in ast.walk(native) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='STATS'];assert not native_calls
 output={'complete':True,'source_gate':'pass_fixed_six_controls_only','owner_source_commit':'52eea74f','reviewer_source_sha256':sha(Path(__file__)),'author_protocol_sha256_before_gate':sha(D/'protocol.json'),'pins_checked':checked,'whole_file_restoration_exact':True,'snapshot_immediately_before_rescale_branch':True,'native_stats_sites':len(changes),'cases':p['cases'],'predicted_total_fake_model_rows_per_arm':53,'predicted_total_fake_estimator_calls_both_arms':128,'predicted_total_fake_truth_calls_both_arms':106,'predicted_rescale_calls_both_arms':22,'source_calls_executed':0,'real_model_truth_mass_random_calls':0,'resource_gate':'one 19/20 CPU review-lock run; external 2 GB and one thread','claims':['Only cells/cut/resolved reporting changes; original call order, outputs and Ops objects are preserved.','Native STATS are cumulative until caller reset; shallow dict copy suffices for integer values.','Unequal rescaled counts invalidate universal halving and noisy 9/8 correction.'],'limitations':['Controls are synthetic extracted-loop checks, not historical statistics reconstruction.','No actual estimator/model replay is authorized by this source gate.','Proposed shared source patch remains for DRI integration; this review changes no shared code.','Saved results require separate independent review.'],'cpu_seconds':time.process_time()-start}
 (HERE/'evalH_stats_source_gate.json').write_text(json.dumps(output,indent=2)+'\n')
 print(json.dumps({k:output[k] for k in ('complete','source_gate','native_stats_sites','cpu_seconds')}))
if __name__=='__main__':main()
