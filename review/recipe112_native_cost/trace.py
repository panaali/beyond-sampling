"""Isolated unchanged-source AST trace with explicit numeric/memory categories."""
import ast,hashlib,importlib.util,json,sys,types
from collections import Counter
from pathlib import Path
import numpy as np
import base_snapshot as B
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'recipe112_recovery/dependencies/understand_m43/est75.py'
PINS=json.loads((HERE/'inventory.json').read_text())['pins']
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==PINS[str(SOURCE)]
assert hashlib.sha256((HERE/'base_snapshot.py').read_bytes()).hexdigest()=='48bf04ead5510fc301a8db9732d90e00179d0ae6524f515344f28df2a94385db'
METHODS={'min','max','sum','mean','any','all','copy','astype'}
class Ledger(B.Ledger):
 def __init__(self):super().__init__();self.memory=Counter();self.calls=Counter()
 def call(self,site,name,fn,*args,**kwargs):
  if name in ('min','max') and len(args)==1:
   seen=0;cat='integer_control'
   def values():
    nonlocal seen,cat
    for x in args[0]:
     seen+=1
     if self.category(x)!='integer_control':cat='scalar'
     yield x
   result=fn(values(),**kwargs);self.charge(site,cat+'_compare',max(seen-1,0));return result
  if name=='sum':
   def values():
    for x in args[0]:self.charge(site,(self.category(x) or 'integer_control')+'_sum');yield x
   return fn(values(),*args[1:],**kwargs)
  result=fn(*args,**kwargs)
  receiver=fn.__self__ if name.startswith('.') else (args[0] if args else None)
  n=np.size(receiver);m=np.size(result);self.calls[name]+=1
  if name in ('abs','np.abs','np.arctan2'):self.charge(site,'special_'+name,m)
  elif name=='np.linalg.norm':
   assert np.ndim(receiver)==1 and len(args)==1 and not kwargs
   self.charge(site,'norm_mul',n);self.charge(site,'norm_add',max(n-1,0));self.charge(site,'special_sqrt',1)
  elif name=='np.cross':
   assert np.shape(args[0])==(3,) and np.shape(args[1])==(3,) and not kwargs
   self.charge(site,'cross_mul',6);self.charge(site,'cross_sub',3)
  elif name=='np.outer':self.charge(site,'array_Mult',m)
  elif name=='np.maximum':self.charge(site,'array_compare',m)
  elif name in ('np.min','np.max','np.sum','np.mean','.min','.max','.sum','.mean'):
   self.charge(site,'reduce_'+name,max(n-m,0))
   if name.endswith('mean'):self.charge(site,'array_Div',m)
  elif name in ('np.any','np.all','.any','.all'):
   # Nonboolean truth conversion compares each numeric element to zero.
   # Charge all elements even if an implementation can short-circuit.
   if np.asarray(receiver).dtype.kind!='b':self.charge(site,'array_truth_compare',n)
   self.charge(site,'integer_control_reduce_'+name,max(n-m,0))
  elif name in ('min','max'):
   cat='integer_control' if all(isinstance(x,(int,np.integer)) for x in args) else 'scalar';self.charge(site,cat+'_compare',len(args)-1)
  elif name in ('np.array','np.asarray','np.zeros','np.eye','np.full','np.flatnonzero','np.delete','.copy','.astype'):
   self.memory[(site,name,str(np.shape(result)))]+=1
  else:raise ValueError('Unsupported call '+name)
  return result
 def report(self):
  out=super().report();out.update(calls=dict(self.calls),memory_calls=[dict(site=k[0],call=k[1],shape=k[2],count=v) for k,v in self.memory.items()]);return out
class Rewrite(B.Rewrite):
 def visit_AugAssign(self,node):
  if isinstance(node.target,ast.Subscript) and isinstance(node.target.value,ast.Name) and node.target.value.id=='STATS':return node
  return super().visit_AugAssign(node)
 def visit_Call(self,node):
  name=ast.unparse(node.func)
  if isinstance(node.func,ast.Attribute) and node.func.attr in METHODS and not (isinstance(node.func.value,ast.Name) and node.func.value.id=='np'):name='.'+node.func.attr
  if name.startswith('np.') or name in METHODS|{'sum','min','max','abs'} or name.startswith('.'):
   site=self.site(node);node=self.generic_visit(node);out=self.wrap(node,'call',[ast.Constant(site),ast.Constant(name),node.func]+node.args);out.keywords=node.keywords;return out
  return super().visit_Call(node)
def load(instrument):
 dep=SOURCE.parent
 for path in (dep,dep.parent/'arc_m43_repro'):
  if str(path) not in sys.path:sys.path.insert(0,str(path))
 import est_mech,m43
 assert Path(est_mech.__file__).resolve()==dep/'est_mech.py'
 for row in json.loads((HERE.parent/'recipe112_recovery/dependencies.json').read_text()):
  path=HERE.parents[1]/row['snapshot'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
 module=types.ModuleType('isolated_native_recipe');module.__file__=str(SOURCE)
 ledger=Ledger();module.__dict__.update(np=np,HALVES=est_mech.HALVES,Ops=m43.Ops,_arithmetic=ledger)
 tree=ast.parse(SOURCE.read_text());tree.body=[x for x in tree.body if isinstance(x,(ast.Assign,ast.FunctionDef,ast.ClassDef))]
 if instrument:tree=Rewrite().visit(tree)
 exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),module.__dict__)
 startup=ledger.report();module._arithmetic=ledger=Ledger()
 return module,ledger,startup

def evaluate(model,instrument):
 mod,ledger,startup=load(instrument);mod.STATS={'cut':0,'resolved':0,'cells':0};ops=mod.Ops();info={}
 value=mod.est75(model,ops,info=info,**mod.RECIPE112)
 return dict(value=value,hex=float(value).hex(),legacy=ops.n,stats=mod.STATS,info=[dict(key=list(k),value=v) for k,v in info['err_by'].items()],ledger=ledger.report(),startup=startup)
