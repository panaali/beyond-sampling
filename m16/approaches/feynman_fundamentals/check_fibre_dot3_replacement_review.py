import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import ast,json,hashlib,struct,math
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native/outer_coverage/dot3/applicability')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(P/'results.json');p=read(P/'protocol.json');assert r['complete']
assert r['protocol_sha256']==sha(P/'protocol.json') and r['source_sha256']==sha(P/'contract.py')
for path,h in p['pins'].items():assert sha(path)==h
c=read(p['census']);assert c['complete']
identity=read(p['identity'])
def leaves(x):
 if isinstance(x,list):
  for y in x:yield from leaves(y)
 else:yield x
v=list(leaves(identity['model']));assert len(v)==104 and all(type(x) is float and math.isfinite(x) for x in v)
assert hashlib.sha256(b''.join(struct.pack('<d',x) for x in v)).hexdigest()==identity['model_sha256']==r['model_metadata']['model_sha256']
assert len(r['selected'])==10 and len({q['site'] for q in r['selected']})==10
normcalls=dotcalls=clipcalls=old=new=0
for q in r['selected']:
 cats=c['native_sites'][q['site']];n=q['calls']
 if q['kind']=='norm3':assert cats==dict(norm_mul=3*n,norm_add=2*n,special_sqrt=n);normcalls+=n;oldunit=6
 else:
  if q['site'].startswith('clip:'):assert cats==dict(matrix_mul=3*n,matrix_add=2*n);clipcalls+=n
  else:assert cats==dict(dot_mul=3*n,dot_add=2*n);dotcalls+=n
  oldunit=5
 assert q['old_units']==oldunit*n and q['new_upper']==8*n;old+=oldunit*n;new+=8*n
for q in r['source_checks']:
 tree=ast.parse(Path(p['numerical_sources'][q['file']]).read_text());fn,line,_=q['site'].split(':',2);line=int(line[1:]);f=next(z for z in tree.body if isinstance(z,ast.FunctionDef) and z.name==fn)
 assert sum(isinstance(z,(ast.Call,ast.BinOp)) and z.lineno==line and ast.unparse(z)==q['ast_expression'] for z in ast.walk(f))==1
assert normcalls==20501487==c['by_name']['np.linalg.norm']['calls'] and dotcalls==183552==c['by_name']['np.dot']['calls']
assert clipcalls==74509590
assert old==r['selected_old']==496474632 and new==r['selected_new_numeric_upper']==761557032
assert r['added_upper']==new-old==265082400 and r['conditional_revised_hybrid']==r['prior_hybrid_named']+new-old==2103777710
out=dict(complete=True,norm_calls=normcalls,explicit_dot_calls=dotcalls,clip_dot_calls=clipcalls,removed_source_units=old,selected_numeric_upper=new,added_upper=new-old,conditional_hybrid=r['conditional_revised_hybrid'],results_sha256=sha(P/'results.json'),scope='Pinned model metadata, ten exact AST sites and original saved counts independently contracted. Source float64 three-vector invariant and selected all-stride numeric envelope accepted conditionally on historical binding. No full native/hardware/sampler inference; all other source/library/control/memory gaps retained.',new_numerical_calls=0)
Path(__file__).with_name('fibre_dot3_replacement_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
