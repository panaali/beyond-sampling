"""Pinned numerical-expression adapter; native results and operands stay unchanged."""
import ast,hashlib,importlib.util,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;INV=json.loads((HERE/'inventory.json').read_text());SRC=Path(INV['source_root'])
for item in INV['files']:
 p=Path(item['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],str(p)
path=HERE.parent/'plackett_cost_review/inventory.py'
spec=importlib.util.spec_from_file_location('t180_reviewed_inventory',path);inherited=importlib.util.module_from_spec(spec);spec.loader.exec_module(inherited)

class Ledger(inherited.Ledger):
 def call(self,site,name,fn,*args,**kwargs):
  # Other calls retain the independently reviewed T146 numeric rules.
  special=('np.arccos','np.arctan')
  memory=('np.ones','np.where','np.meshgrid')
  if name not in special+memory+('np.cross','np.dot','np.mean','np.linalg.svd','np.errstate'):return super().call(site,name,fn,*args,**kwargs)
  result=fn(*args,**kwargs)
  if name=='np.errstate':
   self.memory[(site,name,'context manager','context manager')]+=1;return result
  shape=inherited.shape_of(args[0]) if args else ();outshape=inherited.shape_of(result);self.calls[(name,str(shape),str(outshape))]+=1
  if name in special:self.charge(site,'special_'+name,np.size(result))
  elif name=='np.cross':
   assert np.shape(args[0])[-1]==np.shape(args[1])[-1]==3 and not kwargs
   n=np.size(result)//3;self.charge(site,'cross_mul',6*n);self.charge(site,'cross_sub',3*n)
  elif name=='np.dot':
   assert np.ndim(args[0])==np.ndim(args[1])==1 and np.shape(args[0])==np.shape(args[1]) and not kwargs
   n=np.size(args[0]);self.charge(site,'dot_mul',n);self.charge(site,'dot_add',max(n-1,0))
  elif name=='np.mean':
   n=np.size(args[0]);m=np.size(result);self.charge(site,'mean_add',max(n-m,0));self.charge(site,'mean_div',m)
  elif name=='np.linalg.svd':self.opaque[(site,name,str(shape),str(outshape),str(kwargs))]+=1
  else:self.memory[(site,name,str(shape),str(outshape))]+=1
  return result

class Rewrite(inherited.Rewrite):
 # T153's computed-receiver repair, generalized only to the same method list.
 def visit_Call(self,node):
  if isinstance(node.func,ast.Attribute) and node.func.attr in ('min','max','sum','mean','any','all','copy','astype') and not isinstance(node.func.value,ast.Name):
   site=self.site(node);name='.'+node.func.attr;node=self.generic_visit(node)
   out=self.wrap(node,'call',[ast.Constant(site),ast.Constant(name),node.func]+node.args);out.keywords=node.keywords;return out
  return super().visit_Call(node)

GUARD='''
def guarded_clip(poly,n,ops=None):
 if ops is not None:ops.add(2*len(n))
 if bool(np.all(np.asarray(n)==0)):return poly,[]
 return _original_clip(poly,n,ops)
'''

def compile_functions(module,path,ledger,enabled,names=None):
 tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and (names is None or n.name in names)]
 module.__dict__['_arithmetic']=ledger
 if enabled:tree=Rewrite().visit(tree)
 exec(compile(ast.fix_missing_locations(tree),str(path)+(' [native count]' if enabled else ' [unmodified numerical source]'),'exec'),module.__dict__)

def load(enabled=True):
 sys.path.insert(0,str(SRC));import fibre4 as FX;import fibre4b as F3;import fibre
 from fiber3 import Ops
 ledger=Ledger()
 # Original model-independent FAMS/rules have already been initialized. Their
 # numerical/native cost is excluded here and remains a separate cold inventory.
 compile_functions(FX,SRC/'fibre4.py',ledger,enabled)
 FX.X=fibre.X;compile_functions(FX,SRC/'fibre.py',ledger,enabled,{'rows_at'})
 compile_functions(F3,SRC/'fibre4b.py',ledger,enabled)
 FX._original_clip=FX.clip;tree=ast.parse(GUARD)
 if enabled:tree=Rewrite().visit(tree)
 exec(compile(ast.fix_missing_locations(tree),'T100 exact-zero guard [native count]' if enabled else 'T100 exact-zero guard','exec'),FX.__dict__);FX.clip=FX.guarded_clip
 assert F3.LOOK==3 and FX.ADAPT is True and FX.NQ_CORR==3 and FX.EMAX==.25
 return FX,F3,Ops,ledger
