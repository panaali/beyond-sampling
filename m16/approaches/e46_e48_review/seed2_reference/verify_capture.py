"""Read-only archive round-trip check; no geometry or probability evaluation."""
from pathlib import Path
import hashlib,json,time
import numpy as np
P=Path(__file__).resolve().parent

def main():
    start=time.process_time();output=P/'serialization_check.json';assert not output.exists()
    d=json.loads((P/'capture.json').read_text());assert d['complete'] and len(d['parents'])==96
    out=dict(complete=False,rows=[],capture_sha256=hashlib.sha256((P/'capture.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    total=0;stream=hashlib.sha256()
    try:
        for i,row in enumerate(d['parents']):
            assert row['index']==i
            path=P/'parents'/row['archive'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            with np.load(path,allow_pickle=False) as a:
                assert a.files==['rays'];r=a['rays']
            assert r.shape==(row['N'],4,4) and r.dtype==np.float64 and np.isfinite(r).all()
            raw=r.tobytes(order='C');assert len(raw)==row['ray_copy_bytes']==128*row['N']
            stream.update(raw);total+=len(r)
            out['rows'].append(dict(index=i,N=len(r),payload_sha256=hashlib.sha256(raw).hexdigest()))
        assert total==d['N']==d['stats']['simplices']==37739
        assert sum(r['stats_delta']['cells'] for r in d['parents'])==9470
        assert sum(r['stats_delta']['resolved'] for r in d['parents'])==9374
        out.update(complete=True,N=total,ordered_float64_payload_sha256=stream.hexdigest())
    except BaseException as e:
        out['failure']=repr(e);raise
    finally:
        out['cpu']=time.process_time()-start
        output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))

if __name__=='__main__':main()
