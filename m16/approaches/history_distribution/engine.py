"""Decorrelated sign-state mixture scored by the preserved exact rank law."""
from collections import defaultdict
import hashlib,importlib.util,sys,time
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from distribution import atoms,ROOT2PI

HERE=Path(__file__).parent
OLD=Path('/workspace/work/recover-small_h4_4/m16/approaches/nonzero_history/sign_cycle')
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('preserved_sign_engine',OLD/'engine.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
sys.path.pop(0)

def calculate(model,subdivisions=2,radial_order=2,order=16,collapse=False,limit=6.,seconds=100.):
    start=time.process_time();stats=defaultdict(float)
    distributions,dreport=atoms(model,subdivisions,radial_order)
    nodes,weights=leggauss(order);stats['legendre_rule_constructions']+=1
    signs=[]
    for sign in (1,-1):
        states,masses=distributions[sign]
        if collapse:states=np.array([masses@states]);masses=np.ones(1)
        # Exact duplicate states only, including origin atoms; no tolerance clustering.
        unique={}
        for state,mass in zip(states,masses):
            key=tuple(float(x) for x in state);unique[key]=unique.get(key,0.)+mass
            stats['exact_atom_key_coordinates']+=len(state)
        values=[];sign_value=0.;sign_partition=0.
        for key,mass in unique.items():
            if time.process_time()-start>seconds:raise TimeoutError('bounded mixture CPU')
            state=np.array(key)
            pieces=prior.landmark_pieces(prior.arrangement(model,state,limit,stats),stats)
            value=0.;partition=0.
            for yl,yr,lower,upper,label in pieces:
                y=(yl+yr)/2+(yr-yl)/2*nodes
                lo=lower[0]*y+lower[1];hi=upper[0]*y+upper[1]
                total,first,last,special,regular=prior.scalar_integrals(y,lo,hi,2,sign,stats)
                truth=special if label==0 else first if label==2 else last if label==3 else regular
                weight=weights*(yr-yl)/2*np.exp(-y*y/2)/ROOT2PI
                value+=float(weight@truth);partition+=float(weight@total)
                stats['outer_normal_density_exponentials']+=len(y);stats['outer_quadrature_nodes']+=len(y)
            sign_value+=mass*value;sign_partition+=mass*partition
            values.append(dict(state=list(key),weight=float(mass),estimate=value,partition_mass=partition))
        signs.append(dict(sign=sign,estimate=sign_value,partition_mass=sign_partition,components=values))
        stats['scored_components']+=len(unique)
        if time.process_time()-start>seconds:raise TimeoutError('bounded mixture CPU')
    value=sum(row['estimate'] for row in signs)/2
    return value,dict(subdivisions=subdivisions,radial_order=radial_order,order=order,collapse=collapse,signs=signs,
        distribution=dreport,work=dict(stats),omitted_y_mass_bound=float(2*ndtr(-limit)),cpu_seconds=time.process_time()-start,
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (OLD/'engine.py',OLD/'ranks.py',OLD.parent/'affine.py')},
        decorrelated_surrogate=True,portable_flops_certified=False,total_error_certified=False)
