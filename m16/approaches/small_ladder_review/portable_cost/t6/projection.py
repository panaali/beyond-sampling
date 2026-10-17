"""Explicit orthant projection with an additive analytic mass-shift allowance."""
import math
import numpy as np

def project(rays,audit):
 dominant=np.take_along_axis(rays,np.argmax(abs(rays),axis=2)[:,:,None],axis=2)[:,:,0]
 signs=np.where(dominant<0,-1.,1.);signed=rays*signs[:,:,None]
 projected=signs[:,:,None]*np.maximum(signed,0);norms=np.sum(abs(projected),axis=1)
 if np.any(norms==0):raise ArithmeticError('projection erased an entire ray')
 projected/=norms[:,None,:]
 shifts=np.sqrt(np.sum((rays-projected)**2,axis=1));epsilon=shifts.max(axis=1)
 l1=np.sum(abs(projected),axis=1);radius=l1.min(axis=1)/math.sqrt(6)-epsilon
 if radius.min()<=0:raise ArithmeticError('no positive projection interpolation radius')
 original_norm=np.sqrt(np.sum(rays*rays,axis=1));new_norm=np.sqrt(np.sum(projected*projected,axis=1))
 maximum=np.maximum(original_norm.max(axis=1),new_norm.max(axis=1))
 bound=(maximum**5*shifts.sum(axis=1)/radius**6+maximum**6*6*epsilon/radius**7)/(120*math.pi**3)
 audit.add('orthant_projection_and_bound',2048*len(rays)+100)
 stats=dict(cones=len(rays),original_mixed_cones=int(np.sum(np.any(signed<0,axis=(1,2)))),maximum_column_l2_shift=float(epsilon.max()),mass_shift_bound=float(bound.sum()),projected_l1_error=float(np.max(abs(l1-1))),projected_mixed_cones=int(np.sum(np.any((projected.min(axis=2)<0)&(projected.max(axis=2)>0),axis=1))),roundoff_certified=False)
 return projected,bound,stats
