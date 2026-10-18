"""Pin reviewed Hopf source and enumerate executed arithmetic/native sites."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/work/recover-small_h2_78/m16/approaches/bulk_hopf_review')
PATHS=[ROOT/'frozen/hopf.py',ROOT/'frozen/zoo44.py',ROOT/'manifest.json',ROOT/'COST_SCOPE.md',ROOT/'root_controls.json',ROOT/'full_circle_control.json',HERE.parent/'plackett_cost_review/inventory.py',HERE.parent/'fibre_t4_native_cost/tracer.py',Path('/workspace/work/recover-small_h2_78/m16/approaches/fibre_cost_review/arithmetic.py')]
def main():
 out={'claim':'bf7491f5','files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in PATHS],'source_root':str(ROOT),'native_sites':[]}
 for p in PATHS[:2]:
  for fun in ast.parse(p.read_text()).body:
   if not isinstance(fun,ast.FunctionDef):continue
   for node in ast.walk(fun):
    if isinstance(node,ast.Call):out['native_sites'].append({'file':p.name,'function':fun.name,'line':node.lineno,'call':ast.unparse(node.func)})
 dst=HERE/'inventory.json';assert not dst.exists();dst.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
