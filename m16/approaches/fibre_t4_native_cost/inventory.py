"""Read-only function/native-call inventory for pinned T4 numerical sources."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'fibre_t4_recovery';inv=json.loads((PRIOR/'inventory.json').read_text());SRC=Path(inv['source_root'])
paths=[SRC/'fibre4.py',SRC/'fibre4b.py',SRC/'fibre.py',SRC/'fiber3.py',PRIOR/'observer.py',PRIOR/'noise_identity.json',PRIOR/'noise_preflight.json',PRIOR/'noise_summary.json',PRIOR/'reference_summary.json',HERE.parent/'plackett_cost_review/inventory.py',HERE.parent/'recipe_cost_review/run.py',Path('/workspace/work/recover-small_h2_78/m16/approaches/fibre_cost_review/arithmetic.py')]
rows=[]
for path in paths:
 item={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 if path.suffix=='.py':
  tree=ast.parse(path.read_text());calls={}
  for node in ast.walk(tree):
   if isinstance(node,ast.Call):
    name=ast.unparse(node.func);calls.setdefault(name,[]).append(node.lineno)
  item.update(calls=calls,functions=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))])
 rows.append(item)
out={'files':rows,'source_root':str(SRC),'candidate_functions':{'fibre4.py':'all top-level functions and State, model-independent setup separately retained','fibre4b.py':'all top-level functions','fibre.py':['rows_at']},'legacy_gap':'rows_at dense matmul arrays are charged as active gates / elapsed input columns; masked Ag multiplication also executed','scope':'Read-only inventory, no cost conclusion'}
(HERE/'inventory.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows[:3]:print(Path(r['path']).name,sorted(k for k in r['calls'] if k.startswith(('np.','math.'))))
