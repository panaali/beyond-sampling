"""Replay only original model-free assignments in isolated namespaces."""
import ast,hashlib,importlib,importlib.util,json,resource,sys,time,types
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;inv=json.loads((HERE/'inventory.json').read_text())
for path,sha in inv['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
native=Path(inv['native']);spec=importlib.util.spec_from_file_location('cold_t180_tracer',native/'tracer.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
sys.path.insert(0,str(T.SRC))
# These baseline imports are harness work; their numerical assignments are
# replayed exactly once below. Unrelated checkpoint I/O remains excluded.
mods={n:importlib.import_module(n) for n in ('fiber3','fibre','fibre4','fibre4b')}

def same(a,b):
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 if isinstance(a,(float,np.floating)):return isinstance(b,(float,np.floating)) and float(a).hex()==float(b).hex()
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b

path=HERE/'cold.json';assert not path.exists(),'Preserve previous attempts';start=time.process_time();out={'complete':False,'modules':{},'inventory_sha256':hashlib.sha256((HERE/'inventory.json').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
path.write_text(json.dumps(out,indent=2)+'\n')
try:
 for name,original in mods.items():
  source=T.SRC/(name+'.py');tree=ast.parse(source.read_text());omitted=[];body=[];targets=[]
  for node in tree.body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):body.append(node)
   elif isinstance(node,ast.Assign) and not any(isinstance(t,ast.Name) and t.id=='HERE' for t in node.targets):
    body.append(node);targets.extend(n.id for target in node.targets for n in ast.walk(target) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store))
   else:omitted.append({'line':node.lineno,'kind':type(node).__name__,'source':ast.unparse(node)[:200]})
  module=types.ModuleType('cold_'+name);module.__dict__.update(original.__dict__);ledger=T.Ledger();module.__dict__['_arithmetic']=ledger
  tree.body=body;tree=T.Rewrite().visit(tree)
  exec(compile(ast.fix_missing_locations(tree),str(source)+' [cold census]','exec'),module.__dict__)
  comparisons={key:same(module.__dict__[key],original.__dict__[key]) for key in targets}
  row={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'targets':comparisons,'omitted':omitted,'ledger':ledger.report()};out['modules'][name]=row
  assert all(comparisons.values()),row
 calls=[x for row in out['modules'].values() for x in row['ledger']['opaque_calls']]
 def count(n):return sum(x['count'] for x in calls if x['key'][1]==n)
 assert count('np.linalg.svd')==6 and count('np.linalg.inv')==48 and count('np.polynomial.legendre.leggauss')==6,calls
 out['complete']=True
except Exception as exc:out['failure']=repr(exc);raise
finally:
 out.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='modules'}),flush=True)
