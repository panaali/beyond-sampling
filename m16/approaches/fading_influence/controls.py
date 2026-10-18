"""Fixed exact analytic and function-preserving transformation controls."""
from pathlib import Path
from fractions import Fraction as F
import json,time,resource,hashlib
from core import integer_model,envelope,witness,pack
HERE=Path(__file__).resolve().parent

def rational_forward(m,x):
    h=[F(0)]*m['H'];gates=[];counts={'mul':0,'add':0,'division':0}
    for z in x:
        values=[]
        for i,row in enumerate(m['wh']):
            v=F(0)
            for a,b in zip(row,h):v+=F(a,m['D'])*b;counts['division']+=1;counts['mul']+=1;counts['add']+=1
            v+=F(m['wi'][i],m['D'])*z;counts['division']+=1;counts['mul']+=1;counts['add']+=1;values.append(v)
        gates.append([v>0 for v in values]);h=[max(v,0) for v in values]
    logits=[]
    for row in m['wo']:
        v=F(0)
        for a,b in zip(row,h):v+=F(a,m['D'])*b;counts['division']+=1;counts['mul']+=1;counts['add']+=1
        logits.append(v)
    return logits,gates,counts

def main():
    start=time.process_time();P=json.loads((HERE/'control_protocol.json').read_text());assert P['source_gate']
    for name,sha in P['pins'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
    out=HERE/'controls.json';progress=HERE/'controls_progress.jsonl';assert not out.exists() and not progress.exists()
    result=dict(complete=False,rows=[],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'core.py',HERE/'controls.py')})
    def log(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r))+'\n')
    def save():
        result.update(cpu=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        p=out.with_suffix('.tmp');p.write_text(json.dumps(pack(result),indent=2)+'\n');p.replace(out)
    def one(name,model,x,expected=None,strict=True):
        log(dict(stage='attempt',name=name));m=integer_model(model);u=envelope(m);r=witness(m,x,u)
        row=dict(name=name,model=m,envelope=u,witness=r);result['rows'].append(row);log(dict(stage='result',row=row));save()
        assert r['strict']==strict
        if expected is not None:assert r['sensitivities_by_input']==expected
        if strict:
            baseline,_,count=rational_forward(m,x);checks=[]
            for j in range(m['T']):
                for sign in (-1,1):
                    delta=sign*r['radius']/2;y=list(map(F,x));y[j]+=delta
                    value,gates,extra=rational_forward(m,y)
                    ok=gates==r['history'] and all(value[a]-baseline[a]==F(r['logit_map_numerators'][a][j],r['logit_denominator'])*delta for a in range(m['T']))
                    check=dict(j=j,sign=sign,delta=delta,passed=ok,counts=extra);checks.append(check)
                    log(dict(stage='radius_check',name=name,check=check));assert ok
            row['radius_checks']=checks;row['radius_baseline_counts']=count;save()
        return row
    try:
        for recurrent in (F(0),F(1,2),F(1),F(2)):
            model=([1],[[recurrent]],[[1],[0],[-1],[2]])
            r=one('positive_r_'+str(recurrent),model,[1,2,3,4],[3*recurrent**(3-j) for j in range(4)])
            assert [z['maximum'] for z in r['envelope']['rows']]==[3*recurrent**j for j in range(4)]
        one('zero_input',([0],[[2]],[[1],[0],[-1],[2]]),[1,-2,3,-4],[F(0)]*4)
        one('negative_reset',([1],[[2]],[[1],[0],[-1],[2]]),[-1,-2,-3,-4],[F(0)]*4)
        one('nonzero_gate_boundary',([1],[[1]],[[1],[0],[-1],[2]]),[0,1,2,3],strict=False)
        one('rank_boundary',([1],[[1]],[[1],[0],[-1],[2]]),[1,1,2,3],strict=False)
        base=([F(1),F(-1)],[[F(1,2),F(-1,4)],[F(1,4),F(1,2)]],[[1,0],[0,1],[-1,1],[2,-1]])
        x=[1,-2,3,-4];b=one('mixed_base',base,x)
        scales=[F(2),F(1,4)];wi,wh,wo=base
        scaled=([scales[i]*wi[i] for i in range(2)],[[scales[i]*wh[i][j]/scales[j] for j in range(2)] for i in range(2)],[[F(r[j])/scales[j] for j in range(2)] for r in wo])
        perm=([wi[1],wi[0]],[[wh[1-i][1-j] for j in range(2)] for i in range(2)],[[r[1],r[0]] for r in wo])
        shifted=(wi,wh,[[F(r[0])+3,F(r[1])-2] for r in wo])
        for name,model in [('positive_neuron_scale',scaled),('neuron_permutation',perm),('common_readout',shifted)]:
            r=one(name,model,x)
            assert r['witness']['sensitivities_by_input']==b['witness']['sensitivities_by_input']
            assert [z['maximum'] for z in r['envelope']['rows']]==[z['maximum'] for z in b['envelope']['rows']]
        assert len(result['rows'])==12;result['complete']=True;save()
    except BaseException as e:result['failure']=repr(e);log(dict(stage='failure',error=repr(e)));save();raise
    print(json.dumps(dict(complete=True,controls=len(result['rows']),cpu=result['cpu'])))
if __name__=='__main__':main()
