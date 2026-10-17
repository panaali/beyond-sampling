"""Private unchanged recurrence with target-last terminal cache substitution."""
from collections import defaultdict
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import ndtr
from terminal_cache import TerminalCache

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from engine import grids,possible_labels,batched_terminal
from polynomial import StagePieces
from panel import density_nodes
sys.path.pop(0)
source=HERE.parent/'engine.py'
spec=importlib.util.spec_from_file_location('target_last_private_engine',source)
private=importlib.util.module_from_spec(spec);spec.loader.exec_module(private)
original_terminal=private.terminal


def estimate(model,hidden_order=5,order=2,cached=True,seconds=100.):
    start=time.process_time();deadline=start+seconds;stats=defaultdict(float)
    T=len(model[2]);limit=6.;target_limit=4.
    axes,states,bounds=grids(model,hidden_order,limit,stats,2.)
    last=TerminalCache(model,states[T-2],limit,target_limit,seconds)
    parts=[last.knots];piece_cache={}
    for t in range(T-1):
        piece_cache[t]=StagePieces(model,states[t],axes[t+1],limit,stats,deadline)
        parts.extend((piece_cache[t].lo,piece_cache[t].hi))
    labels=possible_labels(model[2]);terminal_cache={}
    for label in labels:
        if label==T-1:continue
        batched_terminal(model,states[T-1],0.,.5,label,limit,stats,terminal_cache)
        parts.extend(terminal_cache[label][1:])
    raw=np.concatenate(parts);points=np.unique(raw[(raw>=-target_limit)&(raw<=target_limit)])
    if len(points)-1>2000:raise ValueError('frozen complete-panel cap exceeded')
    lo,hi=points[:-1],points[1:]
    tiny=hi-lo<=8*np.finfo(float).eps*np.maximum(1.,abs((lo+hi)/2))
    omitted_width=float((hi[tiny]-lo[tiny]).sum());lo,hi=lo[~tiny],hi[~tiny]
    mode=(T-2)/(T-1)
    density_upper=T*(T-1)*mode**(T-2)*(1-mode)/(np.sqrt(2*np.pi)*(1-2*ndtr(-limit)))
    omission=density_upper*omitted_width
    q,weights,target_omission=density_nodes(T,limit,target_limit,order,(lo,hi))
    stats['target_density_cdf_values']+=len(q);stats['target_density_exponentials']+=len(q)
    lookup={tuple(h):i for i,h in enumerate(states[T-2])}
    current=[None,None]
    def terminal_override(model_arg,h,u,qq,label,lim,work):
        if label!=T-1:return original_terminal(model_arg,h,u,qq,label,lim,work)
        if current[0]!=u:
            current[0]=u;current[1]=last.values(u,qq)
        work['target_last_row_lookups']+=1
        return current[1][lookup[tuple(h)]]
    private.terminal=terminal_override if cached else original_terminal
    contributions={j:0. for j in labels};transition_error=0.
    try:
        for qq,weight in zip(q,weights):
            operator_cache={}
            for label in labels:
                value,bound=private.fixed_target(model,label,float(qq),axes,states,8,
                    limit,stats,deadline,'polynomial_piece_cached',operator_cache,
                    piece_cache,terminal_cache)
                contributions[label]+=float(weight)*value/T
                transition_error+=float(weight)*bound/T
            stats['streaming_target_nodes']+=1
    finally:private.terminal=original_terminal
    return sum(contributions.values()),dict(hidden_order=hidden_order,order=order,cached=cached,
        complete_panels=len(points)-1,used_panels=len(lo),target_nodes=len(q),
        omitted_panel_mass_bound=omission,omitted_target_mass_bound=target_omission,
        omitted_input_mass_bound=float(1-(1-2*ndtr(-limit))**T),
        raw_target_rule_mass=float(weights.sum()),
        raw_mass_defect_upper=abs(float(weights.sum())-(1-target_omission))+omission,
        transition_quadrature_only_bound=transition_error,contributions=contributions,
        work=dict(stats),target_last_work=dict(last.stats),cpu_seconds=time.process_time()-start,
        driver_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        total_error_certified=False,portable_flops_certified=False)
