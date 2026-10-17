"""Eight deterministic selector semantics controls; no bound or mass work."""
from pathlib import Path
import hashlib,json,time
from selector import select
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dest=P/'controls.json';assert not dest.exists();t=time.process_time()
    out=dict(complete=False,source_sha256=sha(__file__),selector_sha256=sha(P/'selector.py'),protocol_sha256=sha(P/'protocol.json'),rows=[])
    def save():out['cpu']=time.process_time()-t;dest.write_text(json.dumps(out,indent=2)+'\n')
    save()
    cases=[
        ('first_order',['a'],{6:'0',8:'2',10:'2'},'certified_ideal',6,1),
        ('exact_boundary',['a'],{6:'1',8:'2',10:'2'},'certified_ideal',6,1),
        ('fail_then_pass',['a'],{6:'2',8:'1/2',10:'1/4'},'certified_ideal',8,2),
        ('two_child_sum',['a','b'],{6:'3/4',8:'1/2',10:'1/4'},'certified_ideal',8,4),
        ('nonmonotone_tail',['a'],{6:'2',8:'1/2',10:'3'},'certified_ideal',8,2),
        ('all_capped',['a'],{6:'2',8:'2',10:'2'},'capped',None,3),
        ('unknown',['a'],{6:None,8:'0',10:'0'},'unknown',None,1),
        ('negative_bound',['a'],{6:'-1',8:'0',10:'0'},'unknown',None,1),
    ]
    try:
        p=json.loads((P/'protocol.json').read_text());assert [c[0] for c in cases]==p['controls']
        for name,keys,values,status,chosen,calls in cases:
            row=dict(name=name,status='started',calls=[]);out['rows'].append(row);save()
            def get(key,n):
                row['calls'].append([key,n]);v=values[n]
                return dict(status='unknown' if v is None else 'completed',upper_bound=v,index=len(row['calls'])-1,origin='cached' if n==6 else 'new')
            def emit(result):row['result']=result;save()
            result=select(keys,[6,8,10],'1',get,emit)
            assert result['status']==status and result.get('selected_order')==chosen and len(row['calls'])==calls
            if chosen:assert result['predicted_nodes']==len(keys)*chosen**4 and result['assembly_additions']==len(keys)-1
            row['status']='passed';save()
        out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:save();print(json.dumps(dict(complete=out['complete'],cpu=out['cpu'])))
if __name__=='__main__':main()
