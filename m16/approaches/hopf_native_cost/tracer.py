"""Original native calls with explicit arithmetic and opaque sorting/search."""
import ast,hashlib,importlib.util,json,types
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
INV=json.loads((HERE/'inventory.json').read_text())
for row in INV['files']:
 p=Path(row['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)
ROOT=Path(INV['source_root'])
spec=importlib.util.spec_from_file_location('t183_reviewed_t146',HERE.parent/'plackett_cost_review/inventory.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)

class Ledger(prior.Ledger):
 def call(self,site,name,fn,*args,**kwargs):
  special=('np.cos','np.sin','np.arctan2','np.log2')
  memory=('np.concatenate','np.repeat','np.tile','np.arange','np.empty_like','np.where','np.triu_indices')
  opaque=('np.lexsort','np.searchsorted','np.argsort')
  if name not in special+memory+opaque+('np.einsum','.argmax'):
   return super().call(site,name,fn,*args,**kwargs)
  result=fn(*args,**kwargs)
  receiver=fn.__self__ if name.startswith('.') else (args[0] if args else None)
  shape=prior.shape_of(receiver);outshape=prior.shape_of(result)
  self.calls[(name,str(shape),str(outshape))]+=1
  if name in special:self.charge(site,'special_'+name,np.size(result))
  elif name=='.argmax':self.charge(site,'reduce_argmax',max(np.size(receiver)-np.size(result),0))
  elif name=='np.einsum':
   assert args[0]=='ij,njk->nik' and len(args)==3 and not kwargs
   assert np.ndim(args[1])==2 and np.ndim(args[2])==3
   k=args[1].shape[1];assert k==args[2].shape[1]
   n=np.size(result);self.charge(site,'einsum_mul',n*k);self.charge(site,'einsum_add',n*max(k-1,0))
  elif name in opaque:
   # Include every operand dimension, not merely the first key-array shape.
   signature=tuple(prior.shape_of(v) for v in args)
   self.opaque[(site,name,str(signature),str(outshape),str(kwargs))]+=1
  else:self.memory[(site,name,str(shape),str(outshape))]+=1
  return result

class Rewrite(prior.Rewrite):
 def visit_Assign(self,node):
  if any(isinstance(t,ast.Name) and t.id=='n_ops' for t in node.targets):return node
  return self.generic_visit(node)
 def visit_AugAssign(self,node):
  if isinstance(node.target,ast.Name) and node.target.id=='n_ops':return node
  return super().visit_AugAssign(node)
 def visit_Call(self,node):
  if isinstance(node.func,ast.Attribute) and node.func.attr in ('min','max','sum','mean','any','all','copy','astype','argmax') and not (isinstance(node.func.value,ast.Name) and node.func.value.id in ('np','math')):
   site=self.site(node);name='.'+node.func.attr;node=self.generic_visit(node)
   out=self.wrap(node,'call',[ast.Constant(site),ast.Constant(name),node.func]+node.args);out.keywords=node.keywords;return out
  return super().visit_Call(node)

def load(instrument=True,rescue=False,sampler=False):
 path=ROOT/'frozen'/('zoo44.py' if sampler else 'hopf.py');src=path.read_text()
 if rescue:
  assert not sampler
  old='np.repeat((A != 0) | (B != 0), 2, 1)';assert src.count(old)==1
  src=src.replace(old,'np.tile((A != 0) | (B != 0), (1, 2))')
 tree=ast.parse(src)
 if sampler:tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('mc_acc','ops_per_sample')]
 else:tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign))]
 ledger=Ledger();mod=types.ModuleType('isolated_hopf_cost');mod.__dict__.update(np=np,_arithmetic=ledger)
 if instrument:tree=Rewrite().visit(tree)
 exec(compile(ast.fix_missing_locations(tree),str(path)+(' [tile only]' if rescue else '')+(' [count]' if instrument else ''),'exec'),mod.__dict__)
 startup=ledger.report();ledger=Ledger();mod._arithmetic=ledger
 return mod,ledger,startup
Ops=prior.Ops
