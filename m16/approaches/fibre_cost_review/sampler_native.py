"""Same scalar-forward convention, executed through the estimator's AST ledger."""
import ast,inspect,json
from pathlib import Path
import numpy as np
from arithmetic import Ledger,Rewrite

def sampler(model,x,truth):
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

def count(model,x=(.3,-1.2,.7),truth=0):
    L=Ledger();scope={'_arithmetic':L};tree=Rewrite().visit(ast.parse(inspect.getsource(sampler)));exec(compile(ast.fix_missing_locations(tree),'sampler native inventory','exec'),scope)
    expected=sampler(model,x,truth);actual=scope['sampler'](model,x,truth);assert actual==expected
    result=L.report();H=len(model[0]);T=len(model[2]);assert result['named_numeric_total']+result['integer_control_total']==4*T*H+2*(T-1)*H*H,result
    return result

def main():
    from panel import frozen_models
    rows={name:count(model) for name,model,_ in frozen_models()}
    assert all(r['named_numeric_total']==111 and r['integer_control_total']==1 for r in rows.values())
    Path(__file__).with_name('sampler_native_results.json').write_text(json.dumps(rows,indent=2)+'\n');print('All ten: 111 floating operations/comparisons plus one integer correctness comparison = 112.')
if __name__=='__main__':main()
