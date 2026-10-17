"""Private existing q-panel driver, union topology and label0 override only."""
import importlib.util
from pathlib import Path
import sys,time
import numpy as np
from core import moving_knots,stream
from breakpoints import prepare
from polynomial import fallback_rule
HERE=Path(__file__).resolve().parent


def estimate(model,h=9,order=2,direct=True,seconds=100.):
    start=time.process_time()
    sys.path.insert(0,str(HERE.parent/'source_mesh_cycle'))
    try:
        spec=importlib.util.spec_from_file_location('direct_first_private_outer',HERE.parent/'source_mesh_cycle/driver.py')
        outer=importlib.util.module_from_spec(spec);spec.loader.exec_module(outer)
    finally:sys.path.pop(0)
    def setup(*args,**kwargs):
        data=prepare(*args,**kwargs)
        extra=moving_knots(model,data['axes'][2],data['limit'],data['target_limit'],data['stats'])
        data['stats']['fixed_topology_before_moving']=len(data['points'])-1
        data['points']=np.unique(np.r_[data['points'],extra])
        data['stats']['complete_union_panels']=len(data['points'])-1
        return data
    outer.prepare=setup
    outer.stream=lambda data,q,w,deadline:stream(data,q,w,deadline,direct)
    fallback_rule.cache_clear();fallback_rule()
    value,stats=outer.estimate(model,h,order,'baseline',seconds=seconds-(time.process_time()-start))
    stats['direct_first']=direct
    stats['work']['fallback_gauss12_rules_constructed']=1
    stats['cpu_seconds']=time.process_time()-start
    return value,stats
