"""Nonoverlapping child envelopes for saved reducer-application shapes."""
import hashlib,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def price(row):
    side,l,c=row['side'],row['length'],row['opposite']
    assert side in ['L','R'] and l>=1 and c>=1
    if side=='L':assert l<=24 and c<=3
    else:assert l<=4 and c<=5
    if l==1:return dict(scal=c+3,gemv=0,axpy=0,ger=0,total=c+3)
    if side=='L':
        m,n=l-1,c;gemv=2*m*n+7*n+7;ger=2*m*n+n+1
    else:
        m,n=c,l-1;gemv=3*m*n+n+3*m+4;ger=2*m*n+n+1
    axpy=2*(2*c+1)
    # LASTV may shrink to1; full-dimension polynomial still dominates scale.
    assert gemv+axpy+ger>=c+3
    return dict(scal=0,gemv=gemv,axpy=axpy,ger=ger,total=gemv+axpy+ger)

def main():
    out=HERE/'blas_contraction.json';assert not out.exists()
    controls=json.loads((HERE/'blas_controls.json').read_text());assert controls['cases']==1344 and controls['failures']==0
    start=time.process_time();prior=json.loads((HERE/'contraction.json').read_text());modes={}
    for mode,data in prior['modes'].items():
        modes[mode]={}
        for phase,info in data.items():
            rows=[];totals={k:0 for k in ['scal','gemv','axpy','ger','total']}
            for r in info['rows']:
                units=price(r)
                rows.append(dict(shape=r,per_call_child_upper=units))
                for key,value in units.items():totals[key]+=r['calls']*value
            modes[mode][phase]=dict(child_upper=totals,rows=rows)
        for key in totals:assert modes[mode]['total']['child_upper'][key]==modes[mode]['geometry']['child_upper'][key]+modes[mode]['cold_once']['child_upper'][key]
    result=dict(modes=modes,cpu_seconds=time.process_time()-start,pins={x:hashlib.sha256((HERE/x).read_bytes()).hexdigest() for x in ['contraction.json','blas_controls.json','BLAS_CANDIDATE.md']},scope='Only three named reducers application BLAS children, fixed small dimensions and historical SKYLAKEX binding. FMA2, named arithmetic/comparison/source upper. Does not add DLARFG, application scans or other global BLAS calls; one cold setup. Finite source/native controls and historical binding qualifications retained; integer/control/memory/runtime and other drivers remain unpriced. Not full rank cost.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({m:d['total']['child_upper'] for m,d in modes.items()}))
if __name__=='__main__':main()
