"""Execute unchanged orthant AST; read-only NumPy call observation preserves values."""
from pathlib import Path
import ast, hashlib, json, math, time, types
import numpy as np
HERE=Path(__file__).resolve().parent
PROTOCOL=json.loads((HERE/'protocol.json').read_text())
class Ops:
 def __init__(self):self.n=0
 def add(self,n):self.n+=n

def engine(observer=None):
 tree=ast.parse((HERE/'estT_snapshot.py').read_text());node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='orthant')
 x,w=np.polynomial.legendre.leggauss(128);v=35.0*(x+1)
 env=dict(np=np if observer is None else observer,math=math,STATS={'simplices':0},_LOG_S=np.exp(-v),_LOG_T=1-np.exp(-v),_LOG_W=35.0*w*np.exp(-v))
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(HERE/'estT_snapshot.py'),'exec'),env)
 return env['orthant']
class Observer:
 def __init__(self):
  self.dets=[];self.clips=[];self.floors=[];self.inv_shapes=[]
  self.linalg=types.SimpleNamespace(det=self.det,inv=self.inv)
 def __getattr__(self,name):return getattr(np,name)
 def det(self,a):
  v=np.linalg.det(a);self.dets.append((a.shape,np.array(v,copy=True)));return v
 def inv(self,a):self.inv_shapes.append(a.shape);return np.linalg.inv(a)
 def clip(self,a,lo,hi):self.clips.append(np.array(a,copy=True));return np.clip(a,lo,hi)
 def maximum(self,a,b):
  # Source uses maximum(products,1e-300), then maximum(1e-300,om).
  raw=np.asarray(a if np.ndim(a)>0 else b);self.floors.append(np.array(raw,copy=True));return np.maximum(a,b)
 def diagnostics(self):
  ds=[v for sh,v in self.dets if sh==(128,3,3)]
  if not ds:return {'cutoff':True,'inverse_calls':len(self.inv_shapes),'determinant_shapes':[list(s) for s,_ in self.dets]},{}
  co=np.stack(ds).reshape(6,3,128);pc=np.stack([a for a in self.clips if a.shape==(128,)])
  product=np.stack(self.floors[::2]);om=np.stack(self.floors[1::2])
  d={'cutoff':False,'inverse_calls':len(self.inv_shapes),'batched_3det_calls':len(ds),'scalar_3det_calls':len(ds)*128,'scalar_4det_calls':sum(s==(4,4) for s,_ in self.dets),'negative_principal_minors':int(np.sum(co[:,:2]<0)),'zero_principal_minors':int(np.sum(co[:,:2]==0)),'product_floor_hits':int(np.sum(product<1e-300)),'denominator_floor_hits':int(np.sum(om<1e-300)),'pc_clip_hits':int(np.sum(np.abs(pc)>1)),'pc_nonfinite':int(np.sum(~np.isfinite(pc))),'max_abs_pc':float(np.max(np.abs(pc))),'min_principal':float(np.min(co[:,:2])),'min_product':float(np.min(product))}
  return d,dict(cofactors=co,products=product,raw_pc=pc,om=om)

def check_case(R,label,truth,bound=0):
 plain=engine();obs=Observer();observed=engine(obs);a=Ops();b=Ops()
 t=time.process_time();value=plain(R,a);cpu=time.process_time()-t
 t=time.process_time();seen=observed(R,b);diagcpu=time.process_time()-t
 diag,arrays=obs.diagnostics()
 return dict(label=label,value=float(value),reference=float(truth),reference_bound=float(bound),difference=float(value-truth),parity=bool(np.float64(value).tobytes()==np.float64(seen).tobytes()),legacy_ops=a.n,observed_ops=b.n,source_cpu=cpu,diagnostic_cpu=diagcpu,diagnostics=diag),arrays

def verify_pins():
 for f,h in PROTOCOL['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
