"""Primitive shape counts and unchanged three-branch orthant behavior."""
import ast,inspect,json,time
from pathlib import Path
import numpy as np
from inventory import HERE,Ledger,Rewrite,Ops,load,PINS

def scalar_sampler(model,x,truth):
    wi,wh,wo=model;H=len(wi);T=len(wo);zero=0.
    h=[x[0]*z for z in wi];h=[z if z>zero else zero for z in h]
    for t in range(1,T):
        new=[]
        for i in range(H):
            value=x[t]*wi[i]
            for j in range(H):value=value+wh[i][j]*h[j]
            new.append(value if value>zero else zero)
        h=new
    logits=[]
    for row in wo:
        value=row[0]*h[0]
        for j in range(1,H):value=value+row[j]*h[j]
        logits.append(value)
    winner=0
    for i in range(1,T):
        if logits[i]>logits[winner]:winner=i
    return winner,winner==truth

def main():
    start=time.process_time();baseline,_,_=load(False);traced,_,startup=load(True)
    C=np.eye(4);C[0,1]=C[1,0]=.99;B=np.linalg.cholesky(C);thin=np.linalg.inv(B).T
    cases=[('wide',np.eye(4)),('thin',thin),('cutoff',np.diag([1e-13,1,1,1]))];rows=[]
    for name,rays in cases:
        ledger=Ledger();traced._arithmetic=ledger;oa=Ops();ob=Ops();a=baseline.orthant(rays,oa);b=traced.orthant(rays,ob)
        assert float(a).hex()==float(b).hex() and oa.n==ob.n,(name,a,b,oa.n,ob.n)
        if name=='wide':assert a==1/16
        if name=='thin':assert abs(a-(1/4+np.arcsin(.99)/(2*np.pi))/4)<1e-14
        if name=='cutoff':assert a==0
        rows.append(dict(name=name,value=float(a),legacy=oa.n,ledger=ledger.report()))
    ledger=Ledger();a=np.arange(6.).reshape(2,3);b=np.ones((3,4));assert np.array_equal(ledger.binary('dot','MatMult',a,b),a@b)
    assert ledger.rows['dot']=={'matrix_mul':24,'matrix_add':16}
    for name,fn,args,kwargs,expected in [('np.sum',np.sum,(a,),{'axis':0},3),('np.linalg.norm',np.linalg.norm,(a,),{'axis':0},12),('np.clip',np.clip,(a,0,1),{},12)]:
        test=Ledger();test.call('test',name,fn,*args,**kwargs);assert test.report()['inclusive_explicit_total']==expected,(name,test.report())
    scope={'_arithmetic':Ledger()};tree=Rewrite().visit(ast.parse(inspect.getsource(scalar_sampler)));exec(compile(ast.fix_missing_locations(tree),'same native sampler','exec'),scope)
    model=(np.ones(8),np.eye(8),np.ones((4,8)));x=(.3,-1.2,.7,2.1)
    assert scalar_sampler(model,x,0)==scope['scalar_sampler'](model,x,0)
    sample=scope['_arithmetic'].report();assert sample['inclusive_explicit_total']==512,sample
    result=dict(rows=rows,startup=startup,sampler=sample,pins={str(k):v for k,v in PINS.items()},cpu_seconds=time.process_time()-start)
    (HERE/'controls.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cpu=result['cpu_seconds'],mass_counts=[(x['name'],x['ledger']['inclusive_explicit_total']) for x in rows],sampler=512)))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        (HERE/'controls_failure.json').write_text(json.dumps(dict(failure=repr(exc)),indent=2)+'\n');raise
