"""Fixed original-domain quadrature validation, including degenerate triangles."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from triangle import triangle_inverse_norm


def reference(v,p):
    count=0
    def outer(u):
        def inner(t):
            nonlocal count
            count+=1
            x=u*v[0]+(1-u)*(t*v[1]+(1-t)*v[2])
            return (1-u)*float(x@x)**(-p/2)
        return quad(inner,0,1,epsabs=2e-13,epsrel=2e-13)[0]
    value,error=quad(outer,0,1,epsabs=2e-13,epsrel=2e-13)
    return value,error,count


def main():
    start=time.process_time()
    rows=[]
    all_stats=[]
    known={3:np.pi/2,4:3*np.pi/4,5:np.pi/2+2,6:3*np.pi/2+.75}
    for p in [3,4,5,6]:
        base=np.eye(3,p)
        variants=[('orthogonal',base),('ordinary',base+0.2),
                  ('thin',np.ones((3,p))+1e-7*base),
                  ('collinear',np.array([1.,1.25,1.5])[:,None]*np.ones((3,p)))]
        for name,v in variants:
            val,stats=triangle_inverse_norm(v,p)
            ref,err,nodes=reference(v,p)
            scaled,scaled_stats=triangle_inverse_norm(v*3,p,absolute_tolerance=1e-11/3**p)
            permuted,permuted_stats=triangle_inverse_norm(v[[2,0,1]],p)
            all_stats.extend([stats,scaled_stats,permuted_stats])
            rows.append(dict(exponent=p,case=name,actual=val,reference=ref,
                             reference_error_estimate=err,reference_nodes=nodes,
                             absolute_error=abs(val-ref),relative_error=abs(val/ref-1),
                             scaling_relative_error=abs(scaled*3**p/val-1),
                             permutation_relative_error=abs(permuted/val-1),
                             known_orthogonal_value=known[p] if name=='orthogonal' else None,
                             stats=stats))
    result={'cases':rows,'max_relative_error':max(x['relative_error'] for x in rows),
            'max_scaling_relative_error':max(x['scaling_relative_error'] for x in rows),
            'max_permutation_relative_error':max(x['permutation_relative_error'] for x in rows),
            'primary_analytic_calls':sum(x['stats']['positive_quadrature_fallbacks']==0 for x in rows),
            'primary_fallback_calls':sum(x['stats']['positive_quadrature_fallbacks'] for x in rows),
            'primary_fallback_nodes':sum(x['stats']['quadrature_nodes'] for x in rows),
            'independent_reference_nodes':sum(x['reference_nodes'] for x in rows),
            'total_primitive_calls_including_scale_and_permutation':3*len(rows),
            'model_forward_passes':0,'sampled_model_inputs':0,
            'cpu_seconds':time.process_time()-start}
    counters=('qr_factorizations','svd_condition_estimates','linear_solves','determinants',
              'spherical_moment_calls','spherical_area_calls','great_circle_edges',
              'great_circle_arctan2_calls','solid_angle_arctan2_calls',
              'moment_complex_exp_calls','moment_sinc_calls',
              'positive_quadrature_fallbacks','quadrature_nodes',
              'legendre_rule_generations','legendre_rule_generation_nodes',
              'bound_trials','bound_lgamma_calls','bound_exp_calls','bound_log_calls')
    result['all_call_operation_counts']={key:sum(s.get(key,0) for s in all_stats) for key in counters}
    result['fallback_observed_errors_exceeding_quadrature_only_bound']=sum(
        x['stats']['positive_quadrature_fallbacks'] and
        x['absolute_error']>x['stats']['quadrature_error_bound'] for x in rows)
    normal_rows=[]
    for p in [4,5,6]:
        base=np.eye(3,p)
        for name,raw in [('orthogonal',base),('ordinary',base+.2),
                         ('thin',np.ones((3,p))+1e-7*base),
                         ('collinear',np.ones((3,p)))]:
            v=raw/np.sum(raw,axis=1)[:,None]
            val,meta=triangle_inverse_norm(v,p,known_plane_normal=np.ones(p))
            ref,err,nodes=reference(v,p)
            normal_rows.append({'exponent':p,'case':name,'value':val,'reference':ref,
                                'reference_nodes':nodes,'reference_error_estimate':err,
                                'relative_error':abs(val/ref-1),'stats':meta})
    result['known_normal_cases']=normal_rows
    result['known_normal_max_relative_error']=max(x['relative_error'] for x in normal_rows)
    result['known_normal_operation_counts']={key:sum(x['stats'].get(key,0) for x in normal_rows)
        for key in counters+('known_normal_projections',)}
    result['known_normal_reference_nodes']=sum(x['reference_nodes'] for x in normal_rows)
    result['cpu_seconds']=time.process_time()-start
    Path(__file__).with_name('triangle_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','known_normal_cases')},indent=2))


if __name__=='__main__':
    main()
