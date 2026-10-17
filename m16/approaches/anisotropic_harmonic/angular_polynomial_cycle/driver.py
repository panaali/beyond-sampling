"""Private shared branching driver with the fixed polynomial cut space."""
import hashlib
import importlib.util
from pathlib import Path
from poly_space import Space

HERE=Path(__file__).resolve().parent
source=HERE.parent/'operator_cycle'/'adapter.py'
spec=importlib.util.spec_from_file_location('polynomial_base_adapter',source)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def estimate(model,**kwargs):
    base.driver.Space=Space
    value,stats=base.driver.estimate(model,optimized=True,**kwargs)
    w=stats['cut_work']
    stats['accounted_scalar_arithmetic_upper']+=2*(
        w.get('real_operator_multiply_adds',0)+w.get('angular_gram_multiply_adds',0))
    stats['representation']='nonperiodic angular Legendre, matched81DOF'
    stats['angular_degree']=2*stats['M']
    stats['opaque_kernels'].append(dict(name='angular Gauss-Legendre setup',order=2*stats['M']+1,calls=1))
    stats['private_adapter_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    return value,stats


def fourier_estimate(model,**kwargs):
    return base.estimate(model,vectorized=True,**kwargs)
