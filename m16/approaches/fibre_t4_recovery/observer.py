"""Pure observers around original numerical statements; owned-process zero guard."""
import ast,hashlib,inspect,json,sys,textwrap
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def load():
 inv=json.loads((HERE/'inventory.json').read_text())
 for item in inv['files']:
  assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
 for path,sha in json.loads((HERE/'analytic_pins.json').read_text()).items():
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
 sys.path.insert(0,inv['source_root'])
 import fibre4 as FX
 import fibre4b as F3
 from fiber3 import Ops
 assert F3.LOOK==3 and FX.ADAPT is True and FX.NQ_CORR==3 and FX.EMAX==.25
 return FX,F3,Ops

class Observer:
 def __init__(self):self.counts={}
 def tick(self,name,index):
  d=self.counts.setdefault(name,{'calls':0,'iterations':0,'max_iterations':0,'exhaustions':0})
  d['calls']+=int(index==0);d['iterations']+=1;d['max_iterations']=max(d['max_iterations'],index+1)
 def exhausted(self,name):self.counts[name]['exhaustions']+=1
 def subdivision(self,depth,e,limit):
  d=self.counts.setdefault('subdivide',{'calls':0,'max_depth':0,'unsatisfied_depth_caps':0})
  d['calls']+=1;d['max_depth']=max(d['max_depth'],depth)
  d['unsatisfied_depth_caps']+=int(depth>=6 and e>limit)

def install(FX,F3):
 obs=Observer();guard={'calls':0,'exact_zero':0};original=FX.clip
 def clip(poly,n,ops=None):
  guard['calls']+=1
  if ops is not None:ops.add(2*len(n))
  if bool(np.all(np.asarray(n)==0)):
   guard['exact_zero']+=1;return poly,[]
  return original(poly,n,ops)
 FX.clip=clip
 for module,name in ((FX,'trace'),(F3,'_walk_down')):
  tree=ast.parse(textwrap.dedent(inspect.getsource(getattr(module,name))))
  loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(64)']
  assert len(loops)==1;loop=loops[0];assert not loop.orelse
  loop.body.insert(0,ast.parse(f'_review_observer.tick({name!r}, _)').body[0])
  loop.orelse=[ast.parse(f'_review_observer.exhausted({name!r})').body[0]]
  module.__dict__['_review_observer']=obs
  exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(module)+' [cap observer]','exec'),module.__dict__)
 tree=ast.parse(textwrap.dedent(inspect.getsource(FX._subdivide)))
 branches=[n for n in tree.body[0].body if isinstance(n,ast.If) and ast.unparse(n.test)=='e <= EMAX or depth >= 6']
 assert len(branches)==1
 i=tree.body[0].body.index(branches[0]);tree.body[0].body.insert(i,ast.parse('_review_observer.subdivision(depth,e,EMAX)').body[0])
 exec(compile(ast.fix_missing_locations(tree),inspect.getsourcefile(FX)+' [cap observer]','exec'),FX.__dict__)
 return obs,guard
