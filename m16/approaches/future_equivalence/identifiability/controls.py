"""Frozen rational condition controls, with failed predicates retained."""
from pathlib import Path
import json,time,hashlib,resource
from primitive import conditions,pack
H=Path(__file__).resolve().parent

def main():
    out=H/'controls.json';progress=H/'progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'protocol.json').read_text());assert p['source_gate']!='pending';start=time.process_time()
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r),allow_nan=False)+'\n')
    record(dict(stage='attempt',protocol_sha256=hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest()))
    result=dict(complete=False,rows=[])
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        for case in p['cases']:
            record(dict(stage='case_attempt',name=case['name']))
            try:v=conditions(*case['model'])
            except (AssertionError,ValueError,ZeroDivisionError) as e:
                if 'reject' not in case:raise
                row=dict(name=case['name'],rejected=type(e).__name__,passed=True)
            else:
                assert 'reject' not in case and v['checks']==case['checks'] and v['identifiable']==all(case['checks'].values())
                row=dict(name=case['name'],result=v,passed=True)
            result['rows'].append(row);record(dict(stage='case_result',row=row))
        assert len(result['rows'])==16
        result.update(complete=True,all_pass=all(r['passed'] for r in result['rows']),cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=True,cases=len(result['rows']),cpu=result['cpu_seconds'])))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);record(dict(stage='failure',message=repr(e)))
        out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise

if __name__=='__main__':main()
