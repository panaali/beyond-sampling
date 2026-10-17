import itertools,json,math,time
from pathlib import Path
from .conditional import conditional_rank
from .reset import masses

def main():
    start=time.process_time();rows=[]
    for k,N,suffix in itertools.product((0.,.1,math.inf),(0,2),((),(.2,),(-.4,.2),(.2,1.))):
        values=[];oc=pc=0
        for p in range(N+2+len(suffix)):
            v,s=conditional_rank(k,N,suffix,p);values.append(v);oc+=s['outer_callbacks'];pc+=s['pair_callbacks']
        r=masses(k,2)['pair_reset_mass'];err=sum(values)-r
        assert abs(err)<2e-10,(k,N,suffix,values,r)
        assert min(values)>=0
        if N>1:assert values[0]==values[1]
        rows.append(dict(kappa=str(k),older_count=N,suffix=suffix,label_masses=values,partition_error=err,outer_callbacks=oc,pair_callbacks=pc))
    out=dict(complete=True,cases=rows,cpu_seconds=time.process_time()-start,random_model_inputs=0)
    assert out['cpu_seconds']<18
    Path(__file__).with_name('conditional_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),max_partition_error=max(abs(r['partition_error']) for r in rows),cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
