"""Frozen h5/h9 topology counts; no outer transport integration."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from terminal_cache import TerminalCache

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from engine import grids,possible_labels,batched_terminal
from polynomial import StagePieces
sys.path.pop(0)
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=(a[0].ravel(),a[1],a[2][[0,1,3,2]])
start=time.process_time();out=dict(rows=[],fixture='H2T4 seed0 readout rows2/3 swapped',
    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256((HERE/'terminal_cache.py').read_bytes()).hexdigest())
for h in (5,9):
    case_start=time.process_time();stats=defaultdict(float)
    axes,states,bounds=grids(model,h,6.,stats,2.)
    cache=TerminalCache(model,states[2],seconds=100-(time.process_time()-start))
    parts=[np.array([-4.,0.,4.]),cache.knots]
    for t in range(3):
        pieces=StagePieces(model,states[t],axes[t+1],6.,stats,start+100)
        parts.extend((pieces.lo,pieces.hi))
    terminal_cache={}
    for label in possible_labels(model[2]):
        if label==3:continue
        batched_terminal(model,states[3],0.,.5,label,6.,stats,terminal_cache)
        parts.extend(terminal_cache[label][1:])
    raw=np.concatenate(parts);points=np.unique(raw[(raw>=-4)&(raw<=4)])
    widths=np.diff(points)
    row=dict(hidden_order=h,labels=possible_labels(model[2]),
        complete_panel_count=len(points)-1,target_last_knots=len(cache.knots),
        target_last_pieces=len(cache.left),target_last_work=dict(cache.stats),
        remaining_work=dict(stats),minimum_panel_width=float(widths.min()),
        tiny_panel_count=int(np.count_nonzero(widths<=8*np.finfo(float).eps*np.maximum(1.,abs((points[:-1]+points[1:])/2)))),
        cpu_seconds=time.process_time()-case_start)
    out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
    (HERE/'topology.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ('target_last_work','remaining_work')}),flush=True)
