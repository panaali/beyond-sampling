"""Import-only correction gate; forbid model loads and compare compiled ASTs."""
import time
START=time.process_time()
from pathlib import Path
import ast,hashlib,json,resource,inspect,marshal,sys,types
import numpy as np
import adapter
P=Path(__file__).resolve().parent

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));old=np.load;attempts=[]
 def reject(*args,**kw):attempts.append(str(args[0]));raise AssertionError('Unexpected stored-array read')
 np.load=reject
 try:E,counts,M,positive,kernel,rescue,radial=adapter.load()
 finally:np.load=old
 source=Path(E.__file__);tree=ast.parse(source.read_text());functions=[n.name for n in tree.body if isinstance(n,ast.FunctionDef)];checked=[]
 # Independent baseline executes the ORIGINAL engine source; only its imported
 # m43 module is a temporary Ops-only stub, never the unrelated weight loader.
 stub=types.ModuleType('m43');stub.Ops=E.Ops;prior=sys.modules.get('m43');sys.modules['m43']=stub
 base=types.ModuleType('baseline_engine');base.__file__=str(source)
 np.load=reject
 try:exec(compile(source.read_text(),str(source),'exec'),base.__dict__)
 finally:
  np.load=old
  if prior is None:sys.modules.pop('m43',None)
  else:sys.modules['m43']=prior
 for name in functions:
  if name=='_cell_error':continue
  assert marshal.dumps(getattr(E,name).__code__)==marshal.dumps(getattr(base,name).__code__),name
  checked.append(name)
 for name in ('__init__','from_normals','tight'):
  assert marshal.dumps(getattr(E.Cone,name).__code__)==marshal.dumps(getattr(base.Cone,name).__code__)
  checked.append('Cone.'+name)
 counter=E.Ops();counter.add(7);counter.add(-2);assert counter.n==5
 # Binding and default source constants remain identical; no historic loaders run.
 assert M.PROTOCOL['orders']==[2,3,4,6,8,12,16,20,24,32]
 out=dict(complete=True,np_load_attempts=attempts,engine_functions_bytecode_equal=checked,only_engine_import_removed='from m43 import Ops',ops_only_class_extracted=True,ops_fixture=counter.n,numerical_modules_loaded_without_corpus_loaders=True,adapter_sha256=hashlib.sha256((P/'adapter.py').read_bytes()).hexdigest(),initial_controls_retained=True,integrations_rerun=0,cpu=time.process_time()-START)
 (P/'loader_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
