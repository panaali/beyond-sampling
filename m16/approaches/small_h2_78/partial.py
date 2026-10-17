"""Partial reset decomposition; outer quadrature is explicitly approximate."""
from itertools import product
import time
from fractions import Fraction
import numpy as np
from scipy.special import ndtri
from envelope import reset_certificate
from prefix import sign_rank_masses,last_reset_masses
from scalar_tail import conditional_tail


def partial(model,max_suffix=3,order=16,max_cpu=None):
    started=time.process_time();stopped=False
    cert=reset_certificate(model)
    if not cert['holds']:raise ValueError('model lacks this two-negative reset certificate')
    T=len(model[2]);max_suffix=min(max_suffix,T-2)
    if max_suffix<1:raise ValueError('include at least suffix lengths0 and1')
    mass0=sign_rank_masses(T,{T-2:0,T-1:0})[0]
    mass1=sign_rank_masses(T,{T-3:0,T-2:0,T-1:1})[cert['negative_negative_positive_winner']]
    rows=[{'suffix_length':0,'contribution':float(mass0),'exact':str(mass0)},
          {'suffix_length':1,'contribution':float(mass1),'exact':str(mass1)}]
    base_order=2*((order+1)//2)
    stats={'conditional_tail_calls':0,'prefix_recurrent_steps':0,'gate_cells':0,
           'readout_cells':0,'polynomial_probability_calls':0,'logit_pair_roots':0,
           'envelope_builds':0,'rational_arithmetic_operations':0,'rational_comparisons':0}
    class BudgetStop(Exception):pass
    for length in range(2,max_suffix+1):
        dimension=length-1;m=T-length-2;contribution=0.
        axis_orders=[base_order+2*j for j in range(dimension)]
        rules=[np.polynomial.legendre.leggauss(n) for n in axis_orders]
        valid_signs=[s for s in product((0,1),repeat=dimension) if s[0] and not any(s[i-1:i+1]==(0,0) for i in range(1,dimension))]
        try:
            for signs in valid_signs:
                coords=[ndtri((rules[j][0]+1)/4+(.5 if s else 0)) for j,s in enumerate(signs)]
                for ix in product(*[range(n) for n in axis_orders]):
                    if max_cpu is not None and stats['conditional_tail_calls']%256==0 and time.process_time()-started>=max_cpu:
                        raise BudgetStop
                    known=[coords[j][i] for j,i in enumerate(ix)]
                    weight=float(np.prod([rules[j][1][i]/4 for j,i in enumerate(ix)]))
                    p,c=conditional_tail(model,m,known)
                    contribution+=.25*weight*p
                    stats['conditional_tail_calls']+=1
                    for key in stats:
                        if key!='conditional_tail_calls':stats[key]+=c.get(key,0)
        except BudgetStop:
            stopped=True
            break
        rows.append({'suffix_length':length,'contribution':contribution,'outer_dimension':dimension,
                     'outer_orders':axis_orders,'known_sign_patterns':len(valid_signs)})
    masses=last_reset_masses(T)
    resolved_mass=sum(Fraction(v) for v in masses['suffix_masses'][:rows[-1]['suffix_length']+1])
    total=sum(row['contribution'] for row in rows)
    return {'rows':rows,'partial_correct_mass':total,'resolved_event_mass':str(resolved_mass),
            'unresolved_event_mass':str(1-resolved_mass),'no_reset_mass':masses['no_reset_mass'],
            'interval_ignoring_outer_quadrature_error':[total,total+float(1-resolved_mass)],
            'stats':stats,'stopped_due_cpu_budget':stopped,'cpu_seconds':time.process_time()-started,'classification':'analytic scalar elimination plus approximate outer quadrature'}
