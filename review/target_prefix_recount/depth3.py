"""One fixed extension of the 34 already saved exact target depth2 cones."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import resource
import time
import traceback
import core

HERE=Path(__file__).resolve().parent
OUT=HERE/'depth3'
PROTOCOL=HERE/'depth3_protocol.json'
GATE=HERE/'depth3_gate.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,obj,compressed=False):
    temp=path.with_suffix(path.suffix+'.tmp')
    data=(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n').encode()
    with temp.open('wb') as f:
        if compressed:
            with gzip.GzipFile(filename='',fileobj=f,mode='wb',compresslevel=1,mtime=0) as z:z.write(data)
        else:f.write(data)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def event(phase,**kw):
    with (OUT/'progress.jsonl').open('a') as f:
        f.write(json.dumps(dict(phase=phase,**kw),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def decode(s):
    return core.State(core.Cone(tuple(map(tuple,s['rays'])),tuple(map(tuple,s['normals']))),
                      tuple(map(tuple,s['hidden'])),tuple(map(tuple,s['history'])))

def main():
    assert not OUT.exists(),'one frozen attempt; existing partial results prohibit restart'
    p=json.loads(PROTOCOL.read_text());g=json.loads(GATE.read_text())
    assert g['clearance'] is True and g['protocol_sha256']==sha(PROTOCOL)
    assert p['requires_k2_saved_review'] is True and p['parents']==list(range(34))
    assert p['input_depth']==2 and p['output_depth']==3
    for name,digest in {**p['pins'],**g['pins']}.items():assert sha(name)==digest,(name,'source/input drift')
    review=g['k2_saved_review'];assert len(review['commit'])>=8 and all(c in '0123456789abcdef' for c in review['commit'])
    assert g['pins'][review['path']]==review['sha256']==sha(review['path'])
    checked=json.loads(Path(review['path']).read_text());assert checked['complete'] and checked['pass_all']
    assert 0<resource.getrlimit(resource.RLIMIT_CPU)[0]<=110
    assert 0<resource.getrlimit(resource.RLIMIT_AS)[0]<=2147483648
    start=time.process_time();OUT.mkdir();result=dict(status='running',protocol_sha256=sha(PROTOCOL),gate=g,
        parents=[],counts={},max_retained_bits=0,framework_forward_calls=0,checkpoint_decodes=0,mass_calls=0,random_calls=0,
        resource_limits=dict(cpu=resource.getrlimit(resource.RLIMIT_CPU),address_space=resource.getrlimit(resource.RLIMIT_AS)))
    def save():write(OUT/'summary.json',result)
    def cap():assert time.process_time()-start<95,'fixed95CPU soft cap; no automatic restart'
    save();event('attempt',stage='load_saved_parent_records')
    try:
        old=json.loads(Path(p['input']).read_text())
        assert old['status']=='complete' and old['count']==34 and len(old['cone_levels'])==2
        parents=old['cone_levels'][1]['states'];assert len(parents)==34
        wi=tuple(old['weights']['wi']);wh=tuple(map(tuple,old['weights']['wh']));D=old['weights']['D']
        core.validate_weights(wi,wh,D);assert len(wi)==16
        result['weights_input_sha256']=sha(p['input']);save()
        event('result',stage='load_saved_parent_records',count=34)
        all_histories=set()
        for idx,raw in enumerate(parents):
            cap();S=decode(raw);assert S.C.d==2 and len(S.history)==2
            path=OUT/f'parent_{idx:02d}.json.gz';L=core.Ledger();t=time.process_time()
            row=dict(parent_index=idx,parent_history=S.history,status='running',children=[],certificates=[])
            write(path,row,True);event('attempt',parent=idx)
            try:
                children=core.extend(S,wi,wh,D,L)
                row['children']=[core.encode(c) for c in children];write(path,row,True)
                local_histories=set()
                for child in children:
                    cap();assert child.history[:-1]==S.history and len(child.history)==3 and child.C.d==3
                    assert child.history not in local_histories and child.history not in all_histories
                    local_histories.add(child.history)
                    row['certificates'].append(core.certify_state(child,wi,wh,D,L))
                assert children and len(local_histories)==len(children)==len(row['certificates'])
                all_histories.update(local_histories);row['status']='complete'
            except BaseException as e:
                row['status']='failure';row['failure']=repr(e);row['traceback']=traceback.format_exc();raise
            finally:
                row['counts']=L.counts;row['max_retained_bits']=L.max_bits;row['cpu_seconds']=time.process_time()-t
                write(path,row,True)
                item=dict(index=idx,path=path.name,sha256=sha(path),status=row['status'],children=len(row['children']),
                          certificates=len(row['certificates']),cpu_seconds=row['cpu_seconds'],counts=L.counts,max_retained_bits=L.max_bits)
                result['parents'].append(item)
                for name,n in L.counts.items():result['counts'][name]=result['counts'].get(name,0)+n
                result['max_retained_bits']=max(result['max_retained_bits'],L.max_bits);save()
            event('result',parent=idx,children=len(children),status=row['status'])
        assert len(result['parents'])==34 and all(x['status']=='complete' for x in result['parents'])
        result['count']=len(all_histories);assert result['count']==sum(x['children'] for x in result['parents'])
        result['author_reported_k3']=792;result['author_count_agrees']=result['count']==792
        result['status']='complete';event('complete',count=result['count'],author_count_agrees=result['author_count_agrees'])
    except BaseException as e:
        result['status']='failure';result['failure']=repr(e);result['traceback']=traceback.format_exc()
        event('failure',error=repr(e));raise
    finally:
        result['cpu_seconds']=time.process_time()-start;result['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save();print(json.dumps(dict(status=result['status'],cpu_after_final_save=time.process_time()-start)),flush=True)

if __name__=='__main__':main()
