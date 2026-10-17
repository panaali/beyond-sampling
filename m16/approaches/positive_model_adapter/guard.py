"""Local exact-coefficient-zero output tie ownership; source remains unchanged."""
from collections import Counter
import inspect
import numpy as np

def install(E):
    original=E._cell_error
    source=inspect.getsource(original)
    old='''        ci, co = clip(q, L[truth] - L[o], ops)'''
    new='''        margin = L[truth] - L[o]
        GUARD_COUNTS['scalar_equalities'] += len(margin)
        GUARD_COUNTS['boolean_reductions'] += len(margin) - 1
        if np.all(margin == 0):
            GUARD_COUNTS['integer_orientation_comparisons'] += 1
            if o < truth:
                GUARD_COUNTS['earlier_tie'] += 1
                err += mass(q, ops)
                break
            GUARD_COUNTS['later_tie'] += 1
            continue
        ci, co = clip(q, margin, ops)'''
    assert source.count(old)==1
    namespace=E.__dict__
    namespace['GUARD_COUNTS']=Counter()
    exec(compile(source.replace(old,new),'<local_exact_tie_guard>','exec'),namespace)
    return original,namespace['GUARD_COUNTS'],source.replace(old,new)
