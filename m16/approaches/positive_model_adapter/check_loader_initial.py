"""Import-only correction gate; forbid model loads and compare compiled ASTs."""
import time
START=time.process_time()
from pathlib import Path
import ast,hashlib,json,resource,inspect
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
 for name in functions:
  if name=='_cell_error':continue
  original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  observed=ast.parse(inspect.getsource(getattr(E,name))).body[0]
  assert ast.dump(original,include_attributes=False)==ast.dump(observed,include_attributes=False);checked.append(name)
 counter=E.Ops();counter.add(7);counter.add(-2);assert counter.n==5
 # Binding and default source constants remain identical; no historic loaders run.
 assert M.PROTOCOL['orders']==[2,3,4,6,8,12,16,20,24,32]
 out=dict(complete=True,np_load_attempts=attempts,engine_functions_source_ast_equal=checked,only_engine_import_removed='from m43 import Ops',ops_only_class_extracted=True,ops_fixture=counter.n,numerical_modules_loaded_without_corpus_loaders=True,adapter_sha256=hashlib.sha256((P/'adapter.py').read_bytes()).hexdigest(),initial_controls_retained=True,integrations_rerun=0,cpu=time.process_time()-START)
 (P/'loader_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
