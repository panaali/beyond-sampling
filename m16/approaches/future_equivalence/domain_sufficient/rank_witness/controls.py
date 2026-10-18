"""Eight frozen exact rank controls plus four malformed guards."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,sys
from primitive import witness,pack
H=Path(__file__).resolve().parent

def fixtures():
    E=[[F(i==j) for j in range(4)] for i in range(4)];Z=[F(0)]*4
    return [
      ('zero',[Z],0,None),('duplicate',[E[0],E[0]],1,None),
      ('identity',E,4,F(1)),('pivot_order',[E[1],E[0],E[3],E[2]],4,F(1)),
      ('thin',[E[0],E[1],E[2],[0,0,0,F(1,2**200)]],4,F(1,2**200)),
      ('late_basis',[Z,E[0],E[0],E[1],E[2],E[3]],4,F(1)),
      ('cancellation',[[1,1,0,0],[1,1+F(1,2**80),0,0],E[2],E[3]],4,F(1,2**80)),
      ('rank_three',[E[0],E[1],E[2],[1,2,3,0]],3,None)]

def main():
    out=H/'controls.json';progress=H/'controls_progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'protocol.json').read_text());assert p['source_gate']!='pending'
    start=time.process_time();result=dict(complete=False,protocol_sha256=hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest())
    def log(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x),allow_nan=False)+'\n')
    def cap():
        if time.process_time()-start>17:raise RuntimeError('17CPU phase cap')
    log(dict(stage='phase_attempt',protocol_sha256=result['protocol_sha256']))
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        assert str(Path(sys.executable).resolve())==p['runtime_python']
        rows=[]
        for name,rays,rank,det in fixtures():
            cap();log(dict(stage='attempt',name=name));ans=witness(rays)
            passed=ans['rank']==rank and ((ans['determinant'] is None and det is None) or (ans['determinant'] is not None and ans['determinant']['value']==det))
            row=dict(name=name,rays=rays,expected_rank=rank,expected_determinant=det,result=ans,passed=passed)
            rows.append(row);log(dict(stage='result',row=row));assert passed;cap()
        guards=[]
        for name,rays in [('empty',[]),('wrong_width',[[1,0,0]]),('ragged',[[1,0,0,0],[1]]),('nonfinite',[[float('inf'),0,0,0]])]:
            log(dict(stage='attempt',name=name));error=None
            try:witness(rays)
            except (ValueError,OverflowError) as e:error=type(e).__name__
            row=dict(name=name,error=error,passed=error is not None);guards.append(row);log(dict(stage='result',row=row));assert row['passed'];cap()
        assert len(rows)==8 and len(guards)==4
        result.update(complete=True,all_pass=True,cases=12,rows=rows,guards=guards,cpu_seconds=time.process_time()-start)
        log(dict(stage='complete'));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=True,cases=12,cpu=result['cpu_seconds'])))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);log(dict(stage='failure',message=repr(e)));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise
if __name__=='__main__':main()
