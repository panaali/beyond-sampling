import json,time,resource
from pathlib import Path
import numpy as np
import mpmath as mp
from exact_tail import bounds
BASE=Path(__file__).parent;start=time.process_time();mp.mp.dps=160;out={'primitive_checks':[],'rebounded_results':[],'initial_check_failure':'80-digit x=-1 subtraction cannot resolve the near-constant interval tail at N=32; rerun with160 digits, same formulas and cases'}
def save():
 out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'cycle15.json').write_text(json.dumps(out,indent=2)+'\n')
for a,b in ((mp.mpf(1)/6,mp.mpf(1)),(mp.mpf('.3'),mp.mpf('.7')),(mp.mpf('.499'),mp.mpf('.501'))):
 m=(a+b)/2;delta=(b-a)/2;s=mp.sqrt(a*b);alpha=m/s;r=(mp.sqrt(b)-mp.sqrt(a))/(mp.sqrt(b)+mp.sqrt(a))
 for k in range(6):
  direct=mp.quad(lambda t:mp.cos(k*t)/(m+delta*mp.cos(t))**3,[0,mp.pi])*(1 if k==0 else 2)/mp.pi
  closed=(3*alpha**2-1)/(2*s**3) if k==0 else (-r)**k/s**3*(k*k+3*alpha*k+3*alpha**2-1)
  error=abs(direct-closed)/max(abs(closed),mp.mpf('1e-70'));assert error<mp.mpf('1e-55')
  out['primitive_checks'].append(dict(a=str(a),b=str(b),coefficient=k,relative_error=float(error)))
 for n in (4,8,16,32):
  truncated=(3*alpha**2-1)/(2*s**3)+sum(r**k/s**3*(k*k+3*alpha*k+3*alpha**2-1) for k in range(1,n))
  exact=a**-3-truncated
  formula=r**n/(s**3*(1-r))*(n*n+3*alpha*n+3*alpha**2-1+(2*n+3*alpha)*r/(1-r)+r*(1+r)/(1-r)**2)
  error=abs(exact-formula)/max(abs(formula),mp.mpf('1e-150'));assert error<mp.mpf('1e-50')
  out['primitive_checks'].append(dict(a=str(a),b=str(b),tail_start=n,relative_error=float(error)))
save()
for label,rayfile,planfile,resultfile in (('partition','t6_partition','t6_partition_allocation_p3','cycle12'),('reference','t6_retained_cones','t6_retained_cones_allocation_p3','cycle13'),('cheap','cycle14_cold_rays','cycle14_cold_plan','cycle14')):
 rays=np.load(BASE/(rayfile+'.npz'))['rays'];orders=np.load(BASE/(planfile+'.npz'))['orders'];old=json.loads((BASE/(resultfile+'.json')).read_text());s=time.process_time();new=float(bounds(rays,orders).sum())
 omission=0 if label=='partition' else 5.003269270003126e-6
 total=new+old['inner_positive_quadrature_bound']+old['inner_series_truncation_bound']+omission
 row=dict(name=label,unchanged_estimate=old['estimate'],old_outer_bound=old['conditional_outer_quadrature_bound'],exact_coefficient_outer_bound=new,inner_positive_bound=old['inner_positive_quadrature_bound'],inner_series_bound=old['inner_series_truncation_bound'],pruned_mass_bound=omission,total_conditional_error_bound=total,bound_cpu_seconds=time.process_time()-s,cones=len(rays),determinants=len(rays),gram_arithmetic_ops=len(rays)*396,supporting_arithmetic_ops=len(rays)*114,method='exact inverse-cube Chebyshev coefficients, existing orders')
 out['rebounded_results'].append(row);save();print(row,flush=True)
ref=out['rebounded_results'][1];cheap=out['rebounded_results'][2];out['cheap_reference_conditional_error_ceiling']=abs(ref['unchanged_estimate']-cheap['unchanged_estimate'])+ref['total_conditional_error_bound'];save()
