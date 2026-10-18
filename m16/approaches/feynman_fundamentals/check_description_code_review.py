"""Independent recursive length calculation for frozen executable descriptions."""
import ast,hashlib,json,time
from pathlib import Path
start=time.process_time();p=Path('/workspace/work/recover-small_h2_910/algzoo/description_length_audit/results.json');d=json.loads(p.read_text());root=Path('/workspace/clean-neurons/algzoo')
def gamma_length(n):return 2*n.bit_length()-1
def length(x):
 if x is None:return 3
 if isinstance(x,bool):return 4
 if isinstance(x,int):return 3+gamma_length((2*x if x>=0 else -2*x-1)+1)
 if isinstance(x,float):return 67
 if isinstance(x,str):b=x.encode();return 3+gamma_length(len(b)+1)+8*len(b)
 if isinstance(x,(list,tuple)):return 3+gamma_length(len(x)+1)+sum(map(length,x))
 if isinstance(x,ast.AST):return 11+sum(length(getattr(x,f,None)) for f in x._fields)
 raise TypeError(type(x))
class Normalize(ast.NodeTransformer):
 def visit_UnaryOp(self,n):
  n=self.generic_visit(n)
  if isinstance(n.op,ast.USub) and isinstance(n.operand,ast.Constant) and isinstance(n.operand.value,float):return ast.Constant(value=-n.operand.value)
  return n
 def strip(self,n):
  n=self.generic_visit(n);n.body=[v for v in n.body if not(isinstance(v,ast.Expr) and isinstance(v.value,ast.Constant) and isinstance(v.value.value,str))];return n
 visit_Module=strip;visit_FunctionDef=strip
original=[]
for row in d['original']:
 path=root/row['name'];assert hashlib.sha256(path.read_bytes()).hexdigest()==d['source_hashes'][str(path)]
 t=Normalize().visit(ast.parse(path.read_text()));t.body=[v for v in t.body if not(isinstance(v,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='LABELS' for a in v.targets))]
 if row['neurons']==16:regions=[f for f in t.body if isinstance(f,ast.FunctionDef) and f.name.startswith('_n') and f.name[2:].isdigit()]
 else:regions=[next(v.value for v in t.body if isinstance(v,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='P' for a in v.targets))]
 fitted=sum(isinstance(v,ast.Constant) and isinstance(v.value,float) for r in regions for v in ast.walk(r));assert fitted==row['fitted_float_count'];bits=length((t,None));assert bits==row['encoded_bits'];original.append({'model':row['name'],'fitted':fitted,'standalone_bits':bits})
for row in d['universal']:
 label,H,K=row['source'],row['H'],row['K'];source=root/'constant_budget/sweep_results_coefficients.json' if label=='T7' else Path(f'/workspace/work/recover-small_h2_56/algzoo/constant_budget_multifamily/results_H{H}_seed0.json');saved=json.loads(source.read_text());family_index=0 if label=='T7' else ['fourier','position_polynomial','rank_values'].index(row['family']);key=f'{H}_10_k{K}' if label=='T7' else f'{row["family"]}_{K}';beta=saved[key] if label=='T7' else saved['coefficients'][key];extra=((H,K,family_index),beta)
 assert length(extra)==row['conditional_header_payload_bits'];assert row['standalone_bits']==row['shared_library_bits']+length(extra)+6;assert row['fitted_payload_bits']==64*H*K
out={'independent_original_lengths':original,'conditional_dictionary_rows_checked':len(d['universal']),'prefix_framing_bits':{'original_addition':9,'dictionary_addition':6},'cpu_seconds':time.process_time()-start,'scope':'declared code lengths, not minimal complexity or semantic legibility'}
Path(__file__).with_name('description_code_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
