"""Exact bound-based selection; knows nothing about observed quadrature errors."""
from fractions import Fraction as F

def select(keys,orders,tolerance,getter,emit=lambda result:None):
    assert keys and len(set(keys))==len(keys)
    assert list(orders)==sorted(set(orders)) and all(isinstance(n,int) and n>=2 for n in orders)
    tolerance=F(tolerance);assert tolerance>0
    result=dict(status='started',steps=[]);emit(result)
    for n in orders:
        step=dict(order=n,status='started',entries=[]);result['steps'].append(step);emit(result)
        total=F(0)
        for key in keys:
            try:
                value=getter(key,n)
                entry={k:value[k] for k in ('status','index','origin','upper_bound') if k in value};step['entries'].append(entry);emit(result)
                if value['status']!='completed':raise ArithmeticError('bound evaluation unresolved')
                upper=F(value['upper_bound'])
                if upper<0:raise ArithmeticError('negative bound rejected')
                total+=upper
            except Exception as e:
                step.update(status='unknown',exception=repr(e));result.update(status='unknown',order=n);emit(result);return result
        step.update(status='completed',bound_sum=str(total),qualifies=total<=tolerance);emit(result)
        if step['qualifies']:
            m=len(keys);result.update(status='certified_ideal',selected_order=n,bound_sum=str(total),predicted_nodes=m*n**4,
                predicted_source_kernel_proxy=m*(125+40*n**4),assembly_additions=m-1)
            emit(result);return result
    result.update(status='capped',maximum_order=orders[-1]);emit(result);return result
