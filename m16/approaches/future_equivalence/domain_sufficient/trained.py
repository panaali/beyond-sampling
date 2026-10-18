"""Fixed64 distinct-state pairs from immutable exact-prefix archives; no geometry."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import itertools,json,hashlib,time,sys,resource
from primitive import certify,Ledger,pack
H=Path(__file__).resolve().parent

def rat(x):return F(int(x[0]),int(x[1]))
def matrix(x):return tuple(tuple(rat(v) for v in r) for r in x)
def state_key(row):return tuple(tuple(tuple(v) for v in r) for r in row['exact_state'])
def rep_id(row):return (row['parent'],row['source_index'])

def select(model):
    selected=[];populations=[]
    for truth in range(4):
        groups={}
        for row in model['rows']:
            if row['truth']!=truth:continue
            key=state_key(row)
            if key not in groups or rep_id(row)<rep_id(groups[key]):groups[key]=row
        representatives=sorted(groups.values(),key=rep_id)
        pairs=list(itertools.combinations(range(len(representatives)),2));assert len(pairs)>=8
        indices=[k*(len(pairs)-1)//7 for k in range(8)];assert len(set(indices))==8
        populations.append(dict(truth=truth,representatives=[list(rep_id(r)) for r in representatives],pair_count=len(pairs),selected_pair_indices=indices))
        for rank,index in enumerate(indices):
            i,j=pairs[index];selected.append(dict(truth=truth,rank=rank,pair_index=index,a=representatives[i],b=representatives[j]))
    assert len(selected)==32;return selected,populations


def main():
    out=H/'trained.json';progress=H/'trained_progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'trained_protocol.json').read_text());assert p['source_gate']!='pending'
    start=time.process_time()
    def record(row):
        with progress.open('a') as f:f.write(json.dumps(pack(row),allow_nan=False)+'\n')
    def cap():
        if time.process_time()-start>17:raise RuntimeError('17CPU whole-phase cap')
    result=dict(complete=False,protocol_sha256=hashlib.sha256((H/'trained_protocol.json').read_bytes()).hexdigest());record(dict(stage='phase_attempt',protocol_sha256=result['protocol_sha256']))
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        assert str(Path(sys.executable).resolve())==p['runtime_python']
        controls=json.loads((H/'controls.json').read_text());assert controls['complete'] and controls['all_pass'] and controls['cases']==20
        full=json.loads(Path(p['full']).read_text());assert full['complete'] and [m['H'] for m in full['models']]==[4,8]
        models=[];source_hashes={}
        for m in full['models']:
            cap();width=m['H'];chosen,populations=select(m);record(dict(stage='selection',H=width,populations=populations))
            # Retain original source hashes/objects; each phase archive is loaded once.
            sources={}
            for entry in m['inputs']:
                path=Path(entry['path']);actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==entry['sha256'];source_hashes[str(path)]=actual
                d=json.loads(path.read_text());assert d['complete'] and d['checkpoint_sha256']==m['checkpoint_sha256'];sources[str(path)]=d
            first=next(iter(sources.values()));encoded=first['model'];assert all(d['model']==encoded for d in sources.values())
            wi=tuple(map(rat,encoded[0]));wh=matrix(encoded[1]);wo=matrix(encoded[2])
            assert len(wi)==width and len(wh)==width and all(len(r)==width for r in wh) and len(wo)==4 and all(len(r)==width for r in wo)
            L=Ledger();V=[[L.op('sub',wo[k][i],wo[0][i]) for i in range(width)] for k in range(1,4)]
            cache={};rows=[];state_decodes=0;ray_decodes=0;restriction=Counter();canonical=Counter();comparisons=0
            def load_side(row):
                nonlocal state_decodes,ray_decodes
                d=sources[row['source']];raw=d['states'][row['source_index']]
                assert all(raw[k]==row[k] for k in ('parent','truth','history')) and raw['depth']==3
                assert raw['history']==d['analysis']['rows'][row['source_index']]['history']
                assert row['exact_state']==d['analysis']['rows'][row['source_index']]['exact_state']
                key=state_key(row)
                if key not in cache:
                    M=matrix(row['exact_state']);assert len(M)==width and all(len(r)==3 for r in M);state_decodes+=width*3
                    B=[]
                    for i in range(width):
                        form=[]
                        for j in range(3):
                            z=F(0)
                            for k in range(width):z=L.op('add',z,L.op('mul',wh[i][k],M[k][j]))
                            form.append(z)
                        B.append(form+[wi[i]])
                    cache[key]=dict(B=B,V=V)
                rays=[tuple(F(float.fromhex(x)) for x in r) for r in raw['cone']['R']]
                assert rays and all(len(r)==4 for r in rays);ray_decodes+=sum(map(len,rays))
                return cache[key],rays,dict(occurrence_id=row['occurrence_id'],source=row['source'],source_index=row['source_index'],parent=row['parent'],truth=row['truth'],history=row['history'],exact_state=row['exact_state'])
            for c in chosen:
                cap();record(dict(stage='pair_attempt',H=width,truth=c['truth'],rank=c['rank'],pair_index=c['pair_index']))
                assert state_key(c['a'])!=state_key(c['b']) and c['a']['truth']==c['b']['truth']==c['truth']
                a,ra,ba=load_side(c['a']);b,rb,bb=load_side(c['b']);answer=certify(a,b,ra,rb)
                for dom in answer['domains']:
                    restriction.update(dom['restriction_counts']);canonical.update(dom['canonical']['counts']);comparisons+=dom['sign_comparison_allowance']
                row=dict(H=width,truth=c['truth'],rank=c['rank'],pair_index=c['pair_index'],a=ba,b=bb,function_a=a,function_b=b,result=answer)
                rows.append(row);record(dict(stage='pair_result',row=row));cap()
            summary=dict(pairs=len(rows),both_hulls_certified=sum(r['result']['certified'] for r in rows),left_hull_certified=sum(r['result']['domains'][0]['certified'] for r in rows),right_hull_certified=sum(r['result']['domains'][1]['certified'] for r in rows),unknown_both=sum(not r['result']['certified'] for r in rows))
            assert len(rows)==32
            saved=dict(H=width,checkpoint_sha256=m['checkpoint_sha256'],model=encoded,populations=populations,rows=rows,summary=summary,formation_counts=dict(L.counts),formation_observed_bits=L.bits,unique_selected_state_forms=len(cache),weight_decodes=width+width*width+4*width,state_decodes=state_decodes,ray_decodes=ray_decodes,restriction_arithmetic=dict(restriction),canonical_arithmetic_including_unused_projective=dict(canonical),sign_comparison_allowance=comparisons)
            models.append(saved);record(dict(stage='model_result',H=width,summary=summary))
        result.update(complete=True,models=models,source_hashes=source_hashes,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_geometry_network_forward_random_reference_mass_calls=0,scope=p['scope'])
        record(dict(stage='complete'));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=True,cpu=result['cpu_seconds'],models=[m['summary'] for m in models])))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);record(dict(stage='failure',message=repr(e)));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise

if __name__=='__main__':main()
