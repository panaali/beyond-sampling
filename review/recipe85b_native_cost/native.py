"""Isolated RECIPE85B adapter with explicit flat-tuple comparison accounting."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'recipe112_native_cost'
PINS=json.loads((HERE/'inventory.json').read_text())['pins']
for path in (PRIOR/'trace.py',PRIOR/'base_snapshot.py',PRIOR/'inventory.json'):
 assert hashlib.sha256(path.read_bytes()).hexdigest()==PINS[str(path)]
sys.path.insert(0,str(PRIOR))
spec=importlib.util.spec_from_file_location('t176_prior_native',PRIOR/'trace.py');N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)

class Ledger(N.Ledger):
 def __init__(self):super().__init__();self.sort_comparisons=0
 def primitive(self,site,kind,a,b):
  assert isinstance(a,(float,np.floating,int,np.integer,bool,np.bool_)) and isinstance(b,(float,np.floating,int,np.integer,bool,np.bool_)),(type(a),type(b))
  cat=self.category(a,b);self.charge(site,cat+'_sort_'+kind)
  return a==b if kind=='eq' else a<b
 def lesser(self,site,a,b):
  if isinstance(a,tuple) and isinstance(b,tuple):
   for x,y in zip(a,b):
    # CPython tuple equality first uses identity, then rich equality.
    if x is y:continue
    if self.primitive(site,'eq',x,y):continue
    return self.primitive(site,'lt',x,y)
   self.charge(site,'integer_control_sort_length');return len(a)<len(b)
  return self.primitive(site,'lt',a,b)
 def sort(self,site,fn,*args,**kwargs):
  key=kwargs.get('key',lambda x:x);ledger=self
  class Key:
   def __init__(self,value):self.value=value
   def __lt__(self,other):
    ledger.sort_comparisons+=1
    return ledger.lesser(site,self.value,other.value)
  kwargs['key']=lambda x:Key(key(x))
  return fn(*args,**kwargs)
 def report(self):
  out=super().report();out['sort_dispatch_calls']=self.sort_comparisons;return out
N.Ledger=Ledger

def evaluate(model,instrument):
 mod,ledger,startup=N.load(instrument);mod.STATS={'cut':0,'resolved':0,'cells':0};ops=mod.Ops();info={}
 value=mod.est75(model,ops,info=info,**mod.RECIPE85B)
 return dict(value=float(value),hex=float(value).hex(),legacy=ops.n,stats=mod.STATS,info=[dict(key=list(k),value=float(v)) for k,v in info['err_by'].items()],ledger=ledger.report(),startup=startup)
