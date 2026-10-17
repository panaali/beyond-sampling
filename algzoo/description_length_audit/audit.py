"""Read-only scalar-plus-executable-syntax audit, not minimal complexity."""
import time
START=time.process_time()
import ast,hashlib,json,math,struct,sys,resource
from collections import Counter
from pathlib import Path
import numpy as np
import codec as C
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/clean-neurons/algzoo')
T23=Path('/workspace/work/recover-small_h2_56/algzoo/constant_budget_multifamily')
def norm(tree):return C.Normalize().visit(tree)
def dump(tree):return ast.dump(tree,include_attributes=False)
def program(path,functions_only=False):
 tree=norm(ast.parse(path.read_text()))
 if functions_only:tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='features']
 else:tree.body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LABELS' for t in n.targets))]
 return tree

def assess(name,tree,fitted,extra=None,frame=False):
 payload=(tree,extra) if frame else tree;bits=C.encode(payload);decoded=C.decode(bits);restored=decoded[0] if frame else decoded;assert dump(tree)==dump(restored)
 if extra is not None:assert C.encode(extra)==C.encode(decoded[1])
 values=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,float)]
 out=dict(name=name,encoded_bits=len(bits),fitted_float_count=fitted,fitted_payload_bits=64*fitted,other_code_and_structure_bits=len(bits)-64*fitted,ast_nodes=sum(1 for _ in ast.walk(tree)),float_literals=len(values),syntax_sha256=hashlib.sha256(dump(tree).encode()).hexdigest(),encoded_sha256=hashlib.sha256(bits.encode()).hexdigest(),roundtrip=True)
 assert out['other_code_and_structure_bits']>=0
 return out,restored

def environment(tree):
 scope={'np':np};exec(compile(ast.fix_missing_locations(tree),'<decoded predictor>','exec'),scope);return scope

def main():
 out=dict(complete=False,codec_controls=C.controls(),python=sys.version,ast_schema=C.SCHEMA,codebook=C.NAMES,source_hashes={},original=[],universal=[],model_forward_calls=0,fits=0)
 # Fixed deterministic arrays; no original recurrent network is loaded.
 x=np.array([[((i+3)*(j+5)%31-15)/7 for j in range(10)] for i in range(31)],dtype=float)
 x=np.concatenate([x,np.zeros((1,10)),np.ones((1,10)),np.tile(np.arange(10),(1,1)),np.tile(-np.arange(10),(1,1))])
 for filename,H in [('labels.py',16),('labels_32_10.py',32)]:
  path=ROOT/filename;out['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();tree=program(path)
  if H==16:
   fitted_nodes=[n for f in tree.body if isinstance(f,ast.FunctionDef) and f.name.startswith('_n') and f.name[2:].isdigit() for n in ast.walk(f) if isinstance(n,ast.Constant) and isinstance(n.value,float)]
  else:
   table=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='P' for t in n.targets));fitted_nodes=[n for n in ast.walk(table) if isinstance(n,ast.Constant) and isinstance(n.value,float)]
  row,decoded=assess(filename,tree,len(fitted_nodes),frame=True);original=environment(ast.parse(path.read_text()));restored=environment(decoded);checks=0
  for i in range(H):
   a=original['f'](i,x);b=restored['f'](i,x);assert np.array_equal(a,b);checks+=len(x)
  subsets=[]
  for n in ast.walk(tree):
   is_set=isinstance(n,ast.Set);is_selected_list=H==16 and isinstance(n,ast.List)
   if (is_set or is_selected_list) and n.elts and all(isinstance(t,ast.Constant) and type(t.value) is int and 0<=t.value<9 for t in n.elts):subsets.append(sorted(t.value for t in n.elts))
  position_uses=[]
  selected_functions=[f for f in tree.body if isinstance(f,ast.FunctionDef) and (f.name=='_formula' if H==32 else f.name.startswith('_n') and f.name[2:].isdigit())]
  for function in selected_functions:
   for node in ast.walk(function):
    chosen=False
    if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name):
     if node.value.id in ('x','T1','T2','rk') and isinstance(node.slice,ast.Tuple):chosen=True
     if H==16 and node.value.id=='F' and isinstance(node.slice,ast.Constant) and node.slice.value in ('x8','x9','R7'):chosen=True
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='top2' and node.args and isinstance(node.args[0],ast.Constant):chosen=True
    if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id in ('am','a2'):chosen=True
    if chosen:position_uses.append(dict(function=function.name,expression=ast.unparse(node)))
  row.update(formula_position_uses=position_uses,position_inventory_is_syntax_not_parameter_count=True,neurons=H,coefficient_budget_max=5,prediction_scalar_checks=checks,subset_literal_occurrences=subsets,distinct_subsets=sorted(set(tuple(s) for s in subsets)),nine_bit_mask_occurrence_bits=9*len(subsets),nine_bit_mask_distinct_bits=9*len(set(tuple(s) for s in subsets)),subset_diagnostic_not_additional=True)
  assert len(fitted_nodes)==(78 if H==16 else 142)
  out['original'].append(row)
 for label,path,families in [('T7',ROOT/'constant_budget/sweep.py',[None]),('T23',T23/'run.py',['fourier','position_polynomial','rank_values'])]:
  out['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();tree=program(path,True)
  wrapper='def predict(x, beta, K, family):\n    phi, names = features(x)\n    return np.maximum(phi[:, :K] @ beta, 0)\n' if label=='T7' else 'def predict(x, beta, K, family):\n    phi, names = features(x, family)\n    return np.maximum(phi[:, :K] @ beta, 0)\n'
  tree.body.extend(ast.parse(wrapper).body);library,decoded=assess(label+'_shared_library',tree,0);out.setdefault('libraries',[]).append(library);scope=environment(tree);restored=environment(decoded)
  for H in (16,32):
   source=ROOT/'constant_budget/sweep_results_coefficients.json' if label=='T7' else T23/f'results_H{H}_seed0.json';out['source_hashes'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest();saved=json.loads(source.read_text());saved=saved if label=='T7' else saved['coefficients']
   for family in families:
    for K in (1,3,5,10,20):
     key=f'{H}_10_k{K}' if label=='T7' else f'{family}_{K}';beta=saved[key];assert len(beta)==K and all(len(r)==H for r in beta);assert all(isinstance(v,float) and math.isfinite(v) for r in beta for v in r)
     # Global family index and dimensions; float matrix has an explicit nested-list structure.
     header=(H,K,families.index(family));extra=(header,beta);bits=C.encode(extra);recovered=C.decode(bits);assert extra==recovered
     a=scope['predict'](x,np.asarray(beta),K,family);b=restored['predict'](x,np.asarray(recovered[1]),K,family);assert np.array_equal(a,b)
     out['universal'].append(dict(source=label,H=H,K=K,family=family,fitted_scalars=H*K,fitted_payload_bits=64*H*K,conditional_header_payload_bits=len(bits),conditional_structure_bits=len(bits)-64*H*K,standalone_bits=len(C.encode((tree,extra))),standalone_framing_bits=len(C.encode((tree,extra)))-library['encoded_bits']-len(bits),shared_library_bits=library['encoded_bits'],prediction_scalar_checks=int(a.size),roundtrip=True))
 out.update(complete=True,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limitations=['All code lengths are upper bounds relative to the frozen Python/NumPy decoder.','Reporting names and identifier spellings are included; no shortest-code or intrinsic legibility comparison.','T23 uses seed0 coefficients for byte/dimension controls; all seeds have same frozen H*K count.','No source-model scoring, accuracy rerun, quantization or search-compute measurement.'])
 (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'original':out['original'],'libraries':out['libraries'],'universal_rows':len(out['universal']),'cpu':out['cpu_seconds_including_imports']}))
if __name__=='__main__':main()
