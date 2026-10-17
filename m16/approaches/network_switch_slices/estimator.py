"""Unreviewed fixed T4 slice estimator; returned cost is explicitly partial."""
from m16.approaches.network_switch_slices.contract import contract
from m16.approaches.network_switch_slices.slices import walker

CLASS='unreviewed'
DEDUCTIVE=True
FIXED_COST=True
COST_REVIEWED=False
COST_IS_PARTIAL_PROXY=True
SUPPORTS=[(2,4),(4,4)]
ADVICE=None


def estimate_details(model):
    result=contract(model,outer_order=3,inner_order=16,diagnostic_order=0)
    # Arithmetic proxies only: exact Fraction work, allocation, sorting and
    # library implementations remain named but unbounded in this research adapter.
    rates={'affine_coefficient_proxy':1,'readout_coefficient_proxy':1,
           'tested_vertices':5,'intersections':10,'cdf_calls':50,'quantile_calls':50,
           'gaussian_exp_calls':30,'vertical_halfspace_queries':6,
           'stored_halfspace_rows':3,'convexity_turn_tests':10,
           'partition_weighted_sum_proxy':1}
    proxy=sum(sum(rate*f['construction_and_integration_work'].get(key,0)
                  for key,rate in rates.items()) for f in result['families'])
    result['partial_float_operation_proxy']=proxy
    result['partial_fpe_proxy']=proxy/walker.fpe_flops(len(model[0]),len(model[2]))
    result['uncosted_named_work']=['exact binary-rational area fallback','allocation and sorting',
                                 'quadrature/table setup','remaining bookkeeping','library implementation details']
    return result


def est(model,budget,rng=None):
    result=estimate_details(model)
    return result['estimate'],result['partial_fpe_proxy']
