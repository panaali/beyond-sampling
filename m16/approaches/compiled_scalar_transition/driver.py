"""Private StagePieces globals; canonical module objects and files unchanged."""
from collections import defaultdict
import importlib.util
from pathlib import Path
import sys,time,types
from native import NativeBasis
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'rank_conditioned_transport'))
from breakpoints import prepare
from polynomial import StagePieces,fallback_rule


def estimate(model,h=9,order=2,native=True,seconds=100.):
    start=time.process_time();setup=defaultdict(float)
    sys.path.insert(0,str(HERE.parent/'rank_conditioned_transport/source_mesh_cycle'))
    try:
        path=HERE.parent/'rank_conditioned_transport/source_mesh_cycle/driver.py'
        spec=importlib.util.spec_from_file_location('native_scalar_private_outer',path)
        outer=importlib.util.module_from_spec(spec);spec.loader.exec_module(outer)
    finally:sys.path.pop(0)
    if native:
        backend=NativeBasis(setup);namespace=dict(prepare.__globals__)
        namespace['StagePieces']=backend.stage_class(StagePieces)
        constructor=types.FunctionType(prepare.__code__,namespace,prepare.__name__,prepare.__defaults__,prepare.__closure__)
    else:
        constructor=prepare;fallback_rule.cache_clear();fallback_rule()
        setup['python_gauss12_rules_constructed']+=1
    def make(*args,**kwargs):
        data=constructor(*args,**kwargs)
        for k,v in setup.items():data['stats'][k]+=v
        return data
    outer.prepare=make
    value,stats=outer.estimate(model,h,order,'baseline',seconds=seconds-(time.process_time()-start))
    stats['native']=native;stats['cpu_seconds']=time.process_time()-start
    return value,stats
