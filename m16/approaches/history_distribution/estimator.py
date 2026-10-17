"""Unreviewed fixed coupled T4 recipe; returned work is explicitly partial."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

CLASS='unreviewed'
DEDUCTIVE=True
FIXED_COST=True
COST_REVIEWED=False
COST_IS_PARTIAL_PROXY=True
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('owned_history_distribution_engine',HERE/'engine.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
sys.path.pop(0)

def est(model,budget,rng):
    wi,wh,wo=(np.asarray(w,dtype=float) for w in model)
    if len(wi)!=2 or wo.shape!=(4,2):raise ValueError('frozen H2T4 recipe only')
    value,stats=engine.calculate((wi,wh,wo),4,4,16,rank_aligned=True,coupled=True)
    # Unit costs on explicitly counted primitive operations only. Library
    # factorization, sorting, Python/bookkeeping and uncounted work are excluded.
    work=stats['work'];setup=stats['distribution']['work']
    partial=sum(v for k,v in work.items() if k.endswith('_scalar_ops') or k.endswith('_comparisons') or k.endswith('_cdf_values'))
    partial+=sum(v for k,v in setup.items() if k.endswith('_products'))
    forward=2*(4*2*(2+1)+4*2)
    return float(value),float(partial/forward)
