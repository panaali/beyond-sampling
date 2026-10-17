"""Read-only static native-source inventory; no model execution."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'recipe112_recovery/dependencies/understand_m43/est75.py'
SCALAR=HERE.parent/'recipe112_recovery/snapshot/rec112.py'
PATHS=[SOURCE,SCALAR,Path('/workspace/work/recover-small_h4_4/m16/approaches/plackett_cost_review/inventory.py'),Path('/workspace/work/recover-small_h4_4/m16/approaches/recipe_cost_review/run.py'),Path('/workspace/work/recover-small_h2_78/m16/approaches/fibre_cost_review/arithmetic.py')]
REACH={'est75','_est75','_resolve_margin2','_cells_margin','_margin','_arc','clip','frac'}
def main():
 tree=ast.parse(SOURCE.read_text());rows=[]
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and node.name in REACH:
   calls={};ops={}
   for n in ast.walk(node):
    if isinstance(n,ast.Call):calls.setdefault(ast.unparse(n.func),[]).append(n.lineno)
    if isinstance(n,(ast.BinOp,ast.UnaryOp,ast.AugAssign)):
     ops.setdefault(type(n.op).__name__,[]).append(n.lineno)
    if isinstance(n,ast.Compare):
     for op in n.ops:ops.setdefault(type(op).__name__,[]).append(n.lineno)
   rows.append(dict(function=node.name,line=node.lineno,calls=calls,operators=ops))
 out=dict(pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in PATHS},reachable=sorted(REACH),functions=rows,scope='Static source coverage inventory; operations inside ops.add are diagnostic, not estimator arithmetic. No numerical cost inference from static occurrence counts.')
 assert out['pins'][str(SOURCE)]=='426020c67a52678e5cccb20c552837fe2d233c3fe5e13557916034a3c46877d3'
 (HERE/'inventory.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(functions=len(rows),calls=sorted({name for r in rows for name in r['calls']}))))
if __name__=='__main__':main()
