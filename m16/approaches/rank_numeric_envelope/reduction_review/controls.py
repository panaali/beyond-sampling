"""1152 frozen parent controls; shared native children, no rank replay."""
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from application import call,translate
HERE=Path(__file__).resolve().parent
PAIRS=[(1,1),(1,3),(2,1),(2,3),(4,1),(4,3),(6,3),(24,3)]
def bits(a):return np.asarray(a).ravel(order='F').view(np.uint64).tolist()
def main():
    out=HERE/'controls.json';progress=HERE/'controls_progress.jsonl'
    assert not out.exists() and not progress.exists()
    rows=[];start=time.process_time()
    for side,(length,other),tau,vpattern,apattern,inc in itertools.product(['L','R'],PAIRS,[0.,.25,1.],range(3),range(4),[1,3]):
        m,n=(length,other) if side=='L' else (other,length);lda=m+1
        v=np.full(length*inc,19.);v[::inc]=[(-1.)**i/(i+1) for i in range(length)]
        if vpattern==1:v[(max(1,length//2))*inc::inc]=0
        if vpattern==2:v[inc::inc]=0
        v[0]=np.nan # implicit unit entry MUST NOT be accessed
        a=np.full((lda,n),37.,order='F');a[:m,:]=np.arange(1,m*n+1).reshape(m,n,order='F')/8
        if apattern==1:a[:m,-1]=0
        if apattern==2:a[m-1,:]=0
        if apattern==3:a[:m,:]=0
        aa=a.copy(order='F');bb=a.copy(order='F');va=v.copy();vb=v.copy();wa=np.full(max(m,n)+2,43.);wb=wa.copy()
        args=dict(side=side,length=length,other=other,tau=tau,vpattern=vpattern,apattern=apattern,inc=inc)
        with progress.open('a') as f:f.write(json.dumps(dict(start=args))+'\n')
        meta=translate(side,m,n,va,inc,tau,aa,lda,wa);call(side,m,n,vb,inc,tau,bb,lda,wb)
        upper=(length+4+length*other) if side=='L' else (2*length+4+length*other)
        row=dict(args=args,expected=bits(aa),actual=bits(bb),expected_work=bits(wa),actual_work=bits(wb),input_v=bits(v),expected_v=bits(va),actual_v=bits(vb),upper=upper,meta=meta)
        row['match']=row['expected']==row['actual'] and row['expected_work']==row['actual_work'] and row['input_v']==row['expected_v']==row['actual_v'] and np.array_equal(aa[m,:],a[m,:]) and np.array_equal(bb[m,:],a[m,:])
        rows.append(row)
        with progress.open('a') as f:f.write(json.dumps(row)+'\n')
    result=dict(rows=rows,cases=len(rows),failures=sum(not r['match'] or r['meta']['count']>r['upper'] for r in rows),cpu_seconds=time.process_time()-start,pins={x:hashlib.sha256((HERE/x).read_bytes()).hexdigest() for x in ['application.py','controls.py']},scope='Parent scan/branch/BLAS-call semantics only; children shared, not independently validated or priced')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}));assert result['cases']==1152 and result['failures']==0
if __name__=='__main__':main()
