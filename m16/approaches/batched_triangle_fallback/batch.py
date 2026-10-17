"""Preserved scalar decisions, grouped positive nodes; source-pinned local adapter."""
import hashlib,math,sys,time,types
from pathlib import Path
import numpy as np
ROOT=Path('/workspace/clean-neurons');sys.path.insert(0,str(ROOT))
from m16.approaches.small_ladder_review.portable_cost.t6.triangle import Engine
SOURCE=Path(__file__).resolve().parents[1]/'t6_edge_algebra/primitive.py'

def load_evaluator():
 assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()=='f8620876672fb9dac954a7d0f6ce73fe783c2c1b0556dfc042d676b95bacee32'
 src=SOURCE.read_text();old='''    fallback_stats=[]
    for i in np.flatnonzero(~good):
        values[i],meta=positive(vv[i],p+3,absolute_tolerance,128)
        fallback_stats.append(meta)
''';new='''    fallback_ids=np.flatnonzero(~good)
    if hasattr(getattr(positive, '__self__', None), 'many'):
        fallback_values,fallback_stats=positive.__self__.many(vv[fallback_ids],p+3,absolute_tolerance,128)
        values[fallback_ids]=fallback_values
    else:
        fallback_stats=[]
        for i in fallback_ids:
            values[i],meta=positive(vv[i],p+3,absolute_tolerance,128)
            fallback_stats.append(meta)
'''
 assert src.count(old)==1
 m=types.ModuleType('batched_edges');m.__file__=str(SOURCE)
 exec(compile(src.replace(old,new),str(SOURCE)+' [grouped positive callback]','exec'),m.__dict__)
 return m
EDGE=load_evaluator()

class Grouped(Engine):
 def __init__(self,audit,max_nodes=65536):
  super().__init__(audit);self.max_nodes=max_nodes;self.selection_cpu=0.;self.node_cpu=0.;self.groups=0
 def select(self,vertices,exponent,tolerance,max_order):
  if exponent!=6 or vertices.shape!=(3,6):raise ValueError('counted fallback supports dimension6 only')
  self.audit.events['positive_fallback_calls']+=1
  center=vertices.sum(axis=0);center/=math.sqrt(float(center@center))
  lo=float(np.min(vertices@center))**2
  signs=np.sign(vertices.sum(axis=0));projection=float(np.min(vertices@signs))
  if projection>0:lo=max(lo,projection*projection/float(signs@signs))
  hi=float(np.max(np.sum(vertices*vertices,axis=1)));lo=min(lo,hi)
  mid=(lo+hi)/2;ratio=(hi-lo)/(hi+lo)
  self.audit.add('positive_radial_setup',512);choice=None;trials=0
  for n in (4,8,12,16,24,32,48,64,96,128):
   if n>max_order:continue
   next_ratio=ratio*(n+3)/(n+1);trials+=1;self.audit.add('positive_bound_trials',120)
   if next_ratio>=1:continue
   error=0. if ratio==0 else ((n+1)*(n+2)/2)*(ratio**n)/(mid*mid*mid*(1-next_ratio))
   if error<=tolerance:choice=n,error;break
  if choice is None:raise ArithmeticError('positive triangle error bound unavailable')
  n,error=choice
  return dict(quadrature_nodes=n*n,quadrature_error_bound=error,quadrature_order=n,legendre_rule_generations=0,legendre_rule_generation_nodes=0,qmin=lo,qmax=hi,binomial_ratio=ratio,bound_trials=trials,bound_lgamma_calls=0,bound_exp_calls=0,bound_log_calls=0)
 def many(self,vertices,exponent,tolerance,max_order):
  vertices=np.asarray(vertices);values=np.empty(len(vertices));start=time.process_time()
  metas=[self.select(v,exponent,tolerance,max_order) for v in vertices]
  if any(m['quadrature_nodes']>self.max_nodes for m in metas):raise ValueError('one selected rule exceeds node cap')
  groups={}
  for i,m in enumerate(metas):groups.setdefault(m['quadrature_order'],[]).append(i)
  self.audit.add('positive_grouping_scatter',8*len(vertices)+32*len(groups))
  self.selection_cpu+=time.process_time()-start;start=time.process_time()
  for n,indices in groups.items():
   (bary,w),generated=self.rule(n);first=metas[indices[0]]
   first['legendre_rule_generations']=generated;first['legendre_rule_generation_nodes']=n*generated
   chunk=max(1,self.max_nodes//(n*n))
   for begin in range(0,len(indices),chunk):
    ids=indices[begin:begin+chunk];vv=vertices[ids]
    points=vv.transpose(0,2,1)@bary
    q=np.sum(points*points,axis=1);values[ids]=(1/(q*q*q))@w
    self.groups+=1
   self.audit.add('positive_nodes',100*n*n*len(indices));self.audit.add('positive_metadata',100*len(indices))
   self.audit.events['positive_quadrature_nodes']+=n*n*len(indices)
  self.node_cpu+=time.process_time()-start
  return values,metas
