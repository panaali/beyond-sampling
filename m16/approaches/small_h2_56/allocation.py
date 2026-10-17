"""Finite geometry-only quadrature planning; no model inputs or fitted advice."""
from math import gamma,pi,factorial
import time
import numpy as np
ORDERS=np.array((4,6,8,10,12,16,20,24,28,32))

def bound_table(rays,orders=ORDERS,blocksize=4096):
 """Vectorized identical Bernstein ellipse bounds, with component inventory."""
 start=time.process_time();d=rays.shape[1];p=d/2;out=np.empty((len(rays),len(orders)))
 for startidx in range(0,len(rays),blocksize):
  v=rays[startidx:startidx+blocksize];g=np.swapaxes(v,1,2)@v;det=np.abs(np.linalg.det(v))
  center=v.sum(axis=2);center/=np.sqrt(np.sum(center*center,axis=1))[:,None]
  dot=np.einsum('bij,bi->bj',v,center).min(axis=1);support=np.maximum(dot,0)**2
  a=np.maximum((1-1e-12)/d,g.min(axis=(1,2))-1e-12);a=np.maximum(a,support-1e-12)
  b=np.maximum(a,np.diagonal(g,axis1=1,axis2=2).max(axis=1)+1e-12)
  m=(a+b)/2;delta=(b-a)/2
  limit=(np.sqrt(b)+np.sqrt(a))/(np.sqrt(b)-np.sqrt(a))
  rr=1+(limit[:,None]-1)*np.linspace(.1,.95,32)
  lower=m[:,None]-delta[:,None]*(rr+1/rr)/2
  logs=-p*np.log(lower)-np.log(rr-1)
  terms=logs[:,None,:]-(orders[None,:,None]-1)*np.log(rr)[:,None,:]
  out[startidx:startidx+len(v)]=gamma(p)/(2*pi**p)*4/factorial(d-1)*det[:,None]*np.exp(terms.min(axis=2))
 n=len(rays)
 return out,dict(cpu_seconds=time.process_time()-start,determinants=n,gram_arithmetic_ops=n*d*d*(2*d-1),supporting_arithmetic_ops=n*(3*d*d+d),supporting_sqrt_calls=n,ellipse_candidates=n*32,bound_entries=n*len(orders),candidate_bound_evaluations=n*len(orders)*32,unmeasured_operations=['determinant implementation','numpy reductions and interval/ellipse arithmetic'])

def allocate(table,tolerance,power,orders=ORDERS):
 start=time.process_time();cost=orders.astype(float)**power;rows=np.arange(len(table));searches=0
 def choose(multiplier):
  nonlocal searches
  searches+=1;idx=np.argmin(cost[None,:]+multiplier*(table/tolerance),axis=1)
  return idx,float(table[rows,idx].sum())
 # Equal allocation uses the same finite table, with explicit best-order fallback.
 ok=table<=tolerance/len(table);equal=np.where(ok.any(axis=1),ok.argmax(axis=1),table.argmin(axis=1))
 best=table.argmin(axis=1)
 if table[rows,best].sum()>tolerance:
  idx=best;feasible=False
 else:
  lo=0.;hi=1.;idx,error=choose(hi)
  while error>tolerance:
   lo=hi;hi*=2;idx,error=choose(hi)
   if searches>100:raise ArithmeticError('multiplier doubling exhausted')
  for _ in range(48):
   mid=(lo+hi)/2;trial,error=choose(mid)
   if error<=tolerance:hi=mid;idx=trial
   else:lo=mid
  feasible=True
 selected=orders[idx]
 result=dict(tolerance=tolerance,feasible=feasible,conditional_quadrature_error_bound=float(table[rows,idx].sum()),predicted_nodes=int(cost[idx].sum()),equal_allocation_bound=float(table[rows,equal].sum()),equal_allocation_nodes=int(cost[equal].sum()),order_histogram={str(n):int(np.sum(selected==n)) for n in np.unique(selected)},multiplier_searches=searches,search_candidate_evaluations=searches*table.size,search_arithmetic_ops=searches*(3*table.size+max(0,len(table)-1)),search_comparisons=searches*len(table)*(len(orders)-1),cpu_seconds=time.process_time()-start,node_power=power)
 return selected,table[rows,idx],result
