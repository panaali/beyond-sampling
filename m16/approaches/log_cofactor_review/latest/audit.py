"""Execute pinned unchanged hybrid orthant with pass-through numerical observation."""
from pathlib import Path
import ast,json,math,time,types,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'protocol.json').read_text())
class Ops:
 def __init__(self):self.n=0
 def add(self,n):self.n+=n
class Observer:
 def __init__(self):
  self.linalg=types.SimpleNamespace(inv=self.inv,det=self.det,norm=self.norm);self.clear()
 def clear(self):self.dets=[];self.inputs=[];self.inv_shapes=[];self.norm_shapes=[];self.floors=[];self.angles=[];self.q=[]
 def __getattr__(self,name):return getattr(np,name)
 def inv(self,a):self.inv_shapes.append(a.shape);return np.linalg.inv(a)
 def norm(self,a,*args,**kwargs):self.norm_shapes.append(a.shape);return np.linalg.norm(a,*args,**kwargs)
 def det(self,a):
  v=np.linalg.det(a);self.dets.append((a.shape,np.array(v,copy=True)))
  if a.shape==(128,4,4):self.inputs.append(a.copy())
  return v
 def maximum(self,a,b):
  self.floors.append(np.array(a if np.ndim(a)>0 else b,copy=True));return np.maximum(a,b)
 def arctan2(self,y,x):self.angles.append((np.array(y,copy=True),np.array(x,copy=True)));return np.arctan2(y,x)
 def sum(self,a,*args,**kwargs):
  v=np.sum(a,*args,**kwargs)
  if np.shape(a)==(1000,4) and np.shape(v)==(1000,):self.q.append(np.array(v,copy=True))
  return v
 def diagnostics(self):
  shapes=[sh for sh,v in self.dets];thin=not self.inputs
  d=dict(branch='duffy10' if thin else 'log128_atan2',inverse_calls=len(self.inv_shapes),scalar3det_calls=sum(math.prod(sh[:-2]) if len(sh)>2 else 1 for sh in shapes if sh[-2:]==(3,3)),scalar4det_calls=sum(math.prod(sh[:-2]) if len(sh)>2 else 1 for sh in shapes if sh[-2:]==(4,4)),norm_calls=len(self.norm_shapes))
  arrays={}
  if thin:d['q_min']=float(np.min(self.q[0]));arrays['q']=self.q[0]
  else:
   Rt=self.inputs[0];det=next(v for sh,v in self.dets if sh==(128,4,4));angles=np.array(self.angles)
   d.update(raw_negative_detRt=int(np.sum(det<0)),raw_zero_detRt=int(np.sum(det==0)),min_detRt=float(np.min(det)),max_unit_diagonal_drift=float(np.max(abs(np.diagonal(Rt,axis1=1,axis2=2)-1))),angle_nonfinite=int(np.sum(~np.isfinite(angles))),denominator_floor_hits=sum(int(np.sum(v<1e-300)) for v in self.floors[1:]))
   arrays=dict(Rt=Rt,detRt=det,atan2_arguments=angles)
  return d,arrays

def engine(obs=None):
 src=(HERE/'estT_snapshot.py').read_text();assert hashlib.sha256(src.encode()).hexdigest()==P['source_sha256'];tree=ast.parse(src)
 funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['orthant','_duffy']]
 x,w=np.polynomial.legendre.leggauss(128);v=35.*(x+1)
 env=dict(np=np if obs is None else obs,math=math,THIN_DET=1e-6,STATS={'simplices':0},_LOG_S=np.exp(-v),_LOG_T=1-np.exp(-v),_LOG_W=35.*w*np.exp(-v))
 exec(compile(ast.Module(body=funcs,type_ignores=[]),str(HERE/'estT_snapshot.py'),'exec'),env)
 # _duffy is identical source; observer has no effects on setup.
 env['_DU'],env['_DW']=env['_duffy'](10)
 if obs is not None:obs.clear()
 return env['orthant']
class Run:
 def __init__(self):self.obs=Observer();self.plain=engine();self.observed=engine(self.obs)
 def case(self,R,label,p,b=0):
  self.obs.clear();a=Ops();o=Ops();t=time.process_time();v=self.plain(R,a);cpu=time.process_time()-t
  t=time.process_time();w=self.observed(R,o);dcpu=time.process_time()-t;diag,arrays=self.obs.diagnostics()
  return dict(label=label,value=float(v),reference=float(p),reference_bound=float(b),difference=float(v-p),parity=np.float64(v).tobytes()==np.float64(w).tobytes(),legacy_ops=a.n,observed_ops=o.n,source_cpu=cpu,diagnostic_cpu=dcpu,diagnostics=diag),arrays
