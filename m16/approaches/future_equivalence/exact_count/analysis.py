"""Dimension-generalized saved-state analysis; pilot arithmetic unchanged."""
from fractions import Fraction as F
from collections import defaultdict
import time
from pilot import frozen_state,Ledger,pack

def analyze(states,weights,record,start):
    Wi,Wh,Wo=weights;cache={};rows=[];L=Ledger()
    for i,s in enumerate(states):
        if time.process_time()-start>17:raise RuntimeError('17CPU pilot whole-phase analysis cap')
        record(dict(stage='state_attempt',index=i))
        hist=tuple(tuple(g) for g in s['history'])
        assert len(hist)==3 and all(len(g)==len(Wi) and all(v in (0,1) for v in g) for g in hist)
        if hist not in cache:cache[hist]=frozen_state(Wi,Wh,hist,4,L)
        exact=cache[hist]
        stored=tuple(tuple(F(float.fromhex(v)) for v in row[:3]) for row in s['state'])
        assert len(stored)==len(Wi) and all(len(row)==3 for row in stored)
        row=dict(index=i,parent=s['parent'],truth=s['truth'],depth=s['depth'],history=hist,
                 occurrence_id=[s['parent'],i],exact_state=exact,stored_state=stored,
                 map_equal=exact==stored,max_map_difference=max(abs(a-b) for r,q in zip(exact,stored,strict=True) for a,b in zip(r,q,strict=True)))
        rows.append(row);record(dict(stage='state_result',row=pack(row)))
    groups={}
    for field in ('history','exact_state','stored_state'):
        for scope in ('global','same_truth'):
            bucket=defaultdict(list)
            for i,r in enumerate(rows):bucket[(r['truth'],r[field]) if scope=='same_truth' else r[field]].append(i)
            groups[scope+'_'+field]=list(bucket.values())
    resets={k:[g for g in gs if len({rows[i]['history'] for i in g})>1] for k,gs in groups.items() if k.endswith('_state')}
    return dict(rows=rows,groups=groups,distinct_history_reset_groups=resets,
                summary=dict(prefix_pieces=len(rows),group_counts={k:len(v) for k,v in groups.items()},
                             reset_group_counts={k:len(v) for k,v in resets.items()},
                             stored_maps_equal_exact=sum(r['map_equal'] for r in rows),
                             max_map_difference=max((r['max_map_difference'] for r in rows),default=F(0))),
                formation_counts=dict(L.counts),formation_observed_bits=L.bits,
                unique_history_reconstructions=len(cache),cached_history_reuses=len(rows)-len(cache))
