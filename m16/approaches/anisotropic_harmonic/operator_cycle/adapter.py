"""Private unchanged driver, with real cut-space substitution only."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from operator_space import Space,dense

HERE=Path(__file__).resolve().parent
source=HERE.parent/'lune_cycle'/'engine.py'
spec=importlib.util.spec_from_file_location('real_operator_driver',source)
driver=importlib.util.module_from_spec(spec)
previous=sys.modules.get('space')
sys.modules['space']=dense
try:spec.loader.exec_module(driver)
finally:
    if previous is None:sys.modules.pop('space',None)
    else:sys.modules['space']=previous


def estimate(model,vectorized=False,**kwargs):
    driver.Space=lambda *a,**kw:Space(*a,**kw,vectorized=vectorized)
    value,stats=driver.estimate(model,optimized=True,**kwargs)
    count=stats['cut_work'].get('real_operator_multiply_adds',0.)
    stats['accounted_scalar_arithmetic_upper']+=2*count
    stats['representation']='real matrix-free equivalent cut'
    stats['vectorized_angular_assembly']=vectorized
    stats['driver_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    return value,stats


def dense_estimate(model,**kwargs):
    driver.Space=dense.Space
    return driver.estimate(model,optimized=True,**kwargs)
