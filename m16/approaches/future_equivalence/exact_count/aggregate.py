"""Saved-only full-prefix census, checking exactly-once parent coverage."""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict,Counter
import hashlib,json,time,resource
from pilot import H,C,pack


def decode_matrix(a):return tuple(tuple(F(int(x[0]),int(x[1])) for x in r) for r in a)


def main():
    out=H/'full.json';progress=H/'full_progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'full_protocol.json').read_text());assert p['source_gate']!='pending'
    start=time.process_time()
    def record(row):
        with progress.open('a') as f:f.write(json.dumps(pack(row),allow_nan=False)+'\n')
    result=dict(complete=False,models=[],protocol_sha256=hashlib.sha256((H/'full_protocol.json').read_bytes()).hexdigest())
    record(dict(stage='attempt',protocol_sha256=result['protocol_sha256']))
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        for width in (4,8):
            record(dict(stage='model_attempt',H=width))
            paths=([H/'pilot.json'] if width==8 else [])+[H/f'chunk_h{width}_{i}.json' for i in range(8)]
            inherited=None;model=None
            allrows=[];parent_rows=[];newformation=Counter();obs=Counter();newcpu=[];inputs=[];newdecodes=0
            for j,path in enumerate(paths):
                d=json.loads(path.read_text());assert d['complete'] and d['observer']['mass_or_final_calls']==0
                is_reused=width==8 and j==0
                assert d['observer']['lookahead_calls']==d['observer']['lookahead_points']==0
                if model is None:model=d['model']
                if is_reused:
                    inherited=dict(formation_counts=d['analysis']['formation_counts'],canonical_counts=d['analysis'].get('canonical_counts',{}),cpu_seconds=d['cpu_seconds'],path=str(path))
                    assert [r['parent'] for r in d['parents']]==[0,1,20,21]
                else:
                    batch=j-1 if width==8 else j
                    assert d['H']==width and d['batch']==batch and d['protocol_sha256']==result['protocol_sha256']
                    assert d['model']==model and [r['parent'] for r in d['parents']]==p['chunks'][str(width)][batch]
                    newformation.update(d['analysis']['formation_counts']);newcpu.append(d['cpu_seconds']);newdecodes+=d['rational_weight_decodes']
                assert d['checkpoint_sha256']==p['models'][str(width)]['checkpoint_sha256']
                assert len(d['states'])==len(d['analysis']['rows'])==d['observer']['states']
                perparent=Counter()
                for r in d['analysis']['rows']:
                    raw=d['states'][r['index']]
                    assert all(r[k]==raw[k] for k in ('parent','truth','depth','history'))
                    assert decode_matrix(r['stored_state'])==tuple(tuple(F(float.fromhex(x)) for x in row[:3]) for row in raw['state'])
                    hist=tuple(tuple(x) for x in r['history']);assert len(hist)==3 and all(len(g)==width for g in hist)
                    exact=decode_matrix(r['exact_state']);stored=decode_matrix(r['stored_state'])
                    assert len(exact)==len(stored)==width and all(len(v)==3 for v in exact+stored)
                    delta=max(abs(a-b) for q,z in zip(exact,stored,strict=True) for a,b in zip(q,z,strict=True))
                    assert (exact==stored)==r['map_equal'] and delta==F(*map(int,r['max_map_difference']))
                    assert r['depth']==3 and r['truth'] in range(4)
                    row=dict(index=len(allrows),source=str(path),source_index=r['index'],occurrence_id=[width,r['parent'],r['index']],parent=r['parent'],truth=r['truth'],history=hist,exact_state=exact,stored_state=stored,map_equal=exact==stored,max_map_difference=delta)
                    allrows.append(row);perparent[r['parent']]+=1
                assert perparent==Counter({r['parent']:r['states'] for r in d['parents']})
                parent_rows.extend(d['parents'])
                for k in ('proxy','finish_calls','lookahead_calls','lookahead_points','states','mass_or_final_calls'):obs[k]+=d['observer'][k]
                inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reused=is_reused,pieces=d['observer']['states']))
            assert len(parent_rows)==96 and sorted(r['parent'] for r in parent_rows)==list(range(96))
            assert len({tuple(r['occurrence_id']) for r in allrows})==len(allrows)==obs['states']
            groups={}
            for field in ('history','exact_state','stored_state'):
                for scope in ('global','same_truth'):
                    b=defaultdict(list)
                    for i,r in enumerate(allrows):b[(r['truth'],r[field]) if scope=='same_truth' else r[field]].append(i)
                    groups[scope+'_'+field]=list(b.values())
            resets={k:[g for g in gs if len({allrows[i]['history'] for i in g})>1] for k,gs in groups.items() if k.endswith('_state')}
            chart_repeats={k:[g for g in groups[k] if len({allrows[i]['parent'] for i in g})>1] for k in ('global_history','same_truth_history')}
            same_parent_repeats=defaultdict(list)
            for r in allrows:same_parent_repeats[(r['parent'],r['history'])].append(r['index'])
            summary=dict(prefix_pieces=len(allrows),parents=96,group_counts={k:len(gs) for k,gs in groups.items()},reset_group_counts={k:len(gs) for k,gs in resets.items()},cross_chart_history_group_counts={k:len(gs) for k,gs in chart_repeats.items()},same_parent_repeated_history_groups=sum(len(g)>1 for g in same_parent_repeats.values()),stored_maps_equal_exact=sum(r['map_equal'] for r in allrows),max_map_difference=max(r['max_map_difference'] for r in allrows),distinct_global_states_per_prefix_piece=F(len(groups['global_exact_state']),len(allrows)),distinct_same_truth_states_per_prefix_piece=F(len(groups['same_truth_exact_state']),len(allrows)),terminal_cells=None)
            saved=dict(H=width,checkpoint_sha256=p['models'][str(width)]['checkpoint_sha256'],summary=summary,rows=allrows,groups=groups,distinct_history_reset_groups=resets,cross_chart_history_groups=chart_repeats,inputs=inputs,parents=parent_rows,combined_observer=dict(obs),new_formation_counts=dict(newformation),new_weight_decodes=newdecodes,new_chunk_cpu_seconds=sum(newcpu),inherited_analysis_cost=inherited)
            result['models'].append(pack(saved));record(dict(stage='model_result',H=width,summary=summary))
        result.update(complete=True,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_geometry_mass_reference_calls=0,scope=p['scope'])
        record(dict(stage='complete'));out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(complete=True,cpu=result['cpu_seconds'],models=[dict(H=m['H'],summary=m['summary']) for m in result['models']])))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);record(dict(stage='failure',message=repr(e)));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise

if __name__=='__main__':main()
