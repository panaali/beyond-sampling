"""Complete target-value formula transitions for the nonfinal-target engine."""
from collections import defaultdict
import time
import numpy as np
from engine import grids, possible_labels, batched_terminal
from polynomial import StagePieces


def prepare(model, hidden_order, limit=6., target_limit=4., seconds=100., mesh_builder=None):
    start=time.process_time();deadline=start+seconds
    stats=defaultdict(float)
    T=len(model[2]);labels=possible_labels(model[2])
    if T-1 in labels:
        raise ValueError('target-last terminal topology not covered by this breakpoint proof')
    axes,states,bounds=grids(model,hidden_order,limit,stats,2.)
    if mesh_builder is not None:
        axes,states,_=mesh_builder(model,axes,states,limit,stats)
    piece_cache={}
    stage_edges=[]
    for t in range(T-1):
        piece_cache[t]=StagePieces(model,states[t],axes[t+1],limit,stats,deadline)
        stage_edges.extend((piece_cache[t].lo,piece_cache[t].hi))
    terminal_cache={}
    terminal_edges=[]
    for label in labels:
        if time.process_time()>deadline:raise TimeoutError('breakpoint preparation deadline')
        batched_terminal(model,states[T-1],0.,.5,label,limit,stats,terminal_cache)
        _,lo,hi=terminal_cache[label]
        terminal_edges.extend((lo,hi))
    def points(parts):
        raw=np.concatenate([np.array([-target_limit,0.,target_limit]),*parts])
        stats['target_breakpoint_candidates']+=len(raw)
        selected=raw[(raw>=-target_limit)&(raw<=target_limit)]
        return np.unique(selected)
    grid=points(stage_edges)
    terminal=points(terminal_edges)
    all_points=points([grid,terminal])
    stats['target_breakpoint_unique_count']=len(all_points)
    stats['target_breakpoint_setup_cpu']=time.process_time()-start
    return dict(model=model,axes=axes,states=states,bounds=bounds,labels=labels,
                piece_cache=piece_cache,terminal_cache=terminal_cache,stats=stats,
                grid_points=grid,terminal_points=terminal,points=all_points,
                limit=limit,target_limit=target_limit)
