"""Read-only source and fixed prior-case inventory."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'recipe112_native_cost'
TRAIN=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery')
SOURCE=HERE.parent/'recipe112_recovery/dependencies/understand_m43/est75.py'
def main():
 paths=[SOURCE,PRIOR/'trace.py',PRIOR/'base_snapshot.py',PRIOR/'inventory.json',TRAIN/'frozen/est75_t112.py',TRAIN/'continuation_manifest.json',TRAIN/'continuation.py',TRAIN/'continuation_pilot.json']
 rows=[]
 for seed,noise,copy in ((1,20,0),(4,23,255)):
  p=TRAIN/f'k30_s{seed}_noise{noise}.json';paths.append(p);d=json.loads(p.read_text());rows.append(dict(seed=seed,noise=noise,copy=copy,path=str(p),keys=list(d),structure_type=type(d).__name__))
 src=SOURCE.read_text();other=(TRAIN/'frozen/est75_t112.py').read_text();assert '\n'.join(other.splitlines()[2:])+'\n'==src
 tree=ast.parse(src);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_resolve_inv')
 out=dict(pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},cases=rows,recipe=dict(K=[4,4],rule='margin_inv2',eps=.1,cliptol=1e-12),resolve_inv_calls=sorted(set(ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call))))
 (HERE/'inventory.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['cases']))
if __name__=='__main__':main()
