"""Independent saved T190 audit: basis recurrence, not author template code."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,itertools
H=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/recover-small_h2_910/review/time_shift_reuse')

def decode(x):
    if isinstance(x,list):
        if len(x)==2 and all(isinstance(t,str) for t in x):return F(int(x[0]),int(x[1]))
        return [decode(t) for t in x]
    if isinstance(x,dict):return {k:decode(v) for k,v in x.items()}
    return x
def dot(a,b):return sum((x*y for x,y in zip(a,b,strict=True)),F(0))
def recurrence(W,w,g,z,actual=False):
    n=len(w);state=list(z[:n]);pre=[];hist=[]
    for t,mask in enumerate(g):
        a=[sum((W[i][j]*state[j] for j in range(n)),F(0))+w[i]*z[n+t] for i in range(n)]
        pre+=a;hist.append([int(x>0) for x in a]);state=[max(x,F(0)) if actual else x*mask[i] for i,x in enumerate(a)]
    return state,pre,hist
def base_count(k):return 4*k*(k+2)
def expected_counts(k):
    mul=add=base_count(k)
    for a in range(1,k):
        b=k-a;d=2+k;mul+=base_count(a)+base_count(b)+2*(1+b)*d*(2+b)
        add+=base_count(a)+base_count(b)+2*(1+b)*d*(1+b)
    return dict(mul=mul,add=add)

def main():
    out=H/'result.json';progress=H/'progress.jsonl';assert not out.exists() and not progress.exists();start=time.process_time()
    def record(x):
        with progress.open('a') as f:f.write(json.dumps(x)+'\n')
    result=dict(complete=False,rows=0,vectors=0,shifts=0,compositions=0,review_scope_claim='a98dcd8b',new_trained_mass_author_calls=0)
    record(dict(stage='attempt'))
    try:
        frozen=json.loads((H/'protocol.json').read_text())
        for path,sha in frozen['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        p=json.loads((SOURCE/'protocol.json').read_text());raw=json.loads((SOURCE/'controls.json').read_text());data=decode(raw)
        result['pins']={str(SOURCE/n):hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ['protocol.json','controls.json','progress.jsonl','template.py','controls.py','PROOF.md']}
        assert raw['complete'] and raw['protocol_sha256']==result['pins'][str(SOURCE/'protocol.json')]
        assert raw['progress_sha256']==result['pins'][str(SOURCE/'progress.jsonl')]
        assert raw['source_sha256']==result['pins'][str(SOURCE/'controls.py')]
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
        logs=[json.loads(x) for x in (SOURCE/'progress.jsonl').read_text().splitlines()]
        assert len(logs)==842 and sum(x['stage']=='started' for x in logs)==420 and sum(x['stage']=='complete' for x in logs)==420
        for r,l in zip(raw['rows'],[x['row'] for x in logs if x['stage']=='complete'],strict=True):assert r==l
        models={m['name']:([list(map(F,r)) for r in m['Wh']],list(map(F,m['Wi']))) for m in p['models']}
        indexed={(r['model'],tuple(map(tuple,r['word']))):r for r in data['rows']};assert len(indexed)==420
        for row in data['rows']:
            record(dict(stage='row_attempt',key=row['key']));W,w=models[row['model']];word=row['word'];k=len(word);d=2+k;c=row['template']
            basis=[[F(int(i==j)) for i in range(d)] for j in range(d)]
            evaluations=[recurrence(W,w,word,z) for z in basis]
            output=[[v[0][i] for v in evaluations] for i in range(2)]
            planes=[[v[1][i] for v in evaluations] for i in range(2*k)]
            assert output==c['output'] and planes==[x[0] for x in c['planes']]
            assert [x[1] for x in c['planes']]==[g for step in word for g in step]
            assert c['impossible_identically_zero_on']==any(on and not any(r) for r,on in c['planes'])
            assert c['counts']==dict(mul=base_count(k),add=base_count(k)) and row['control_template_and_composition_counts']==expected_counts(k)
            expected=[[F(0)]*d]+[[F(sign if j==q else 0) for j in range(d)] for q in range(d) for sign in (1,-1)]+[[F(1)]*d,[F((j+1)*(-1 if j%2 else 1)) for j in range(d)]]
            assert [v['vector'] for v in row['checks']]==expected
            for saved in row['checks']:
                z=saved['vector'];masked,pre,_=recurrence(W,w,word,z);real,_,actual=recurrence(W,w,word,z,True)
                compatible=all((v>0)==bool(on) for v,(_,on) in zip(pre,c['planes'],strict=True))
                assert masked==saved['masked'] and pre==saved['preactivation'] and real==saved['direct_relu'] and actual==saved['direct_gate_word'] and compatible==saved['compatible']
                assert saved['nonzero_form_zero_values']==sum(bool(any(r)) and v==0 for v,(r,on) in zip(pre,c['planes'],strict=True))
                if compatible:assert real==masked and actual==word
                for shift in (0,2):
                    ambient=[F(13+j) for j in range(d+2)];ambient[:2]=z[:2];ambient[2+shift:2+shift+k]=z[2:]
                    def value(r):return dot(r[:2],ambient[:2])+dot(r[2:],ambient[2+shift:2+shift+k])
                    assert [value(r) for r in output]==masked and [value(r) for r in planes]==pre
                    result['shifts']+=1
                result['vectors']+=1
            for split in range(1,k):
                left=indexed[(row['model'],tuple(map(tuple,word[:split])))]['template'];right=indexed[(row['model'],tuple(map(tuple,word[split:])))]['template'];b=k-split
                def transform(r):return [sum((r[i]*left['output'][i][q] for i in range(2)),F(0)) if q<2+split else r[2+q-(2+split)] for q in range(d)]
                assert [transform(r) for r in right['output']]==output
                cp=[r+[F(0)]*b for r,on in left['planes']]+[transform(r) for r,on in right['planes']]
                assert cp==planes;result['compositions']+=1
            result['rows']+=1;record(dict(stage='row_pass',key=row['key']))
        assert (result['rows'],result['vectors'],result['shifts'],result['compositions'])==(420,5220,10440,720)
        assert data['counts']==dict(words=420,direct_masked=5220,direct_relu=5220,shift_checks=10440,composition_checks=720)
        a,b=data['counterexamples'];assert a['incoming']==[F(1),F(2)] and a['input']==F(-3,2) and a['preactivation']==[F(-1,2),F(1,2)] and a['actual_gates']==[0,1]
        assert b['inputs']==[[F(3),F(2),F(1)],[F(2),F(3),F(1)]] and b['histories']==[[1,1,1],[1,1,1]] and b['states']==[F(1),F(1)] and b['truth']==[1,0] and b['correct']==[False,True]
        result.update(complete=True,cpu_seconds=time.process_time()-start,scope='Exact saved rational semantics/counts verified without author imports or trained computations. Source observed bit-width ledger not independently replayed; no cone feasibility, cache-hit or speed claim.')
        record(dict(stage='complete'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);record(dict(stage='failure',message=repr(e)));out.write_text(json.dumps(result,indent=2)+'\n');raise

if __name__=='__main__':main()
