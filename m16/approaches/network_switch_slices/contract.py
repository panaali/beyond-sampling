"""Frozen complete T4 face contraction with exact-form inner slice geometry."""
from collections import Counter
import time
import numpy as np
from .slices import Slice,walker


def contract(model,outer_order=3,inner_order=16,diagnostic_order=32,deadline=float('inf')):
    if len(model[2])!=4: raise ValueError('This bounded contraction is frozen to T4')
    start=time.process_time();n,w=np.polynomial.legendre.leggauss(outer_order);n=(n+1)/2;w=w/2
    _,z23,z12=walker.tables(4);families=[]
    for kind,factor in [('23',z23),('12',z12)]:
        count=len(walker.geometry.configs(4,kind));candidate=diagnostic=partition=0.
        work=Counter();diagwork=Counter();construction_cpu=integral_cpu=diagnostic_cpu=0.
        maximum_discrepancy=0.;slices=0;maxcells=0;maxpanels=0
        for cfg in range(count):
            for i,u in enumerate(n):
                for j,v in enumerate(n):
                    if time.process_time()>deadline: raise TimeoutError('Full contraction CPU deadline')
                    t=time.process_time();fixed=[.5,u,.5,v]
                    s=Slice(model,kind,cfg,qlo=1e-6,qhi=1-1e-6,fixed_quantiles=fixed)
                    construction_cpu+=time.process_time()-t
                    maxcells=max(maxcells,len(s.cells));maxpanels=max(maxpanels,len(s.knots)-1)
                    t=time.process_time()
                    try: value,norm,total=s.integrate_gaussian(s.knots,inner_order,return_partition=True)
                    except Exception as error:
                        raise RuntimeError(dict(kind=kind,configuration=cfg,outer_indices=[i,j],
                                                fixed_quantiles=fixed,outer_order=outer_order,
                                                completed_slices=slices,elapsed_cpu=time.process_time()-start,
                                                cause=str(error))) from error
                    integral_cpu+=time.process_time()-t;work.update(s.work)
                    weight=w[i]*w[j]/count;candidate+=weight*value;partition+=weight*total
                    if diagnostic_order:
                        before=Counter(s.work);t=time.process_time()
                        val2,norm2=s.integrate_gaussian(s.knots,diagnostic_order)
                        diagnostic_cpu+=time.process_time()-t;diagwork.update(s.work-before)
                        diagnostic+=weight*val2;maximum_discrepancy=max(maximum_discrepancy,abs(val2-value))
                    slices+=1
        families.append(dict(kind=kind,factor=factor,integral=candidate,diagnostic_integral=diagnostic,partition_integral=partition,
                             slices=slices,max_gate_cells=maxcells,max_panels=maxpanels,
                             maximum_inner_discrepancy=maximum_discrepancy,construction_cpu_seconds=construction_cpu,
                             integration_cpu_seconds=integral_cpu,diagnostic_cpu_seconds=diagnostic_cpu,
                             construction_and_integration_work=dict(work),diagnostic_work=dict(diagwork)))
    estimate=1-sum(f['factor']*f['integral'] for f in families)
    diagnostic=1-sum(f['factor']*f['diagnostic_integral'] for f in families) if diagnostic_order else None
    mass=sum(f['factor']*f['partition_integral'] for f in families)
    return dict(outer_order=outer_order,inner_order=inner_order,diagnostic_order=diagnostic_order,
                estimate=estimate,diagnostic_estimate=diagnostic,partition_mass=mass,normalized_estimate=1-(1-estimate)/mass,
                endpoint_omitted_mass_bound=2e-6*(z23+z12),
                families=families,total_cpu_seconds=time.process_time()-start)
