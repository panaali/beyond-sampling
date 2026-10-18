"""Six source-declared budget arithmetic rows; no author imports/evaluations."""
from pathlib import Path
from fractions import Fraction as F
import ast,json,hashlib,math,sys,time,resource
H=Path(__file__).resolve().parent

def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [pack(v) for v in x]
    return x

def assignment(tree,name):
    a=[n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(a)==1;return a[0]

def batch_floor(m,b):
    assert isinstance(m,int) and m>0 and isinstance(b,int) and b>0
    return ((m+b-1)//b)*b

def main():
    out=H/'results.json';progress=H/'progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'run_protocol.json').read_text());assert p['source_gate']!='pending' and p['theorem_gate']!='pending'
    start=time.process_time();result=dict(complete=False,protocol_sha256=hashlib.sha256((H/'run_protocol.json').read_bytes()).hexdigest())
    def log(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x),allow_nan=False)+'\n')
    def cap():
        if time.process_time()-start>17:raise RuntimeError('17CPU phase cap')
    log(dict(stage='phase_attempt',protocol_sha256=result['protocol_sha256']))
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        assert str(Path(sys.executable).resolve())==p['runtime_python']
        assert (sys.float_info.radix,sys.float_info.mant_dig,sys.float_info.rounds)==(2,53,1)
        rb=ast.parse(Path(p['rb9']).read_text());cv=ast.parse(Path(p['cv']).read_text());lb=ast.parse(Path(p['leaderboard']).read_text())
        cq=ast.literal_eval(assignment(rb,'COST_Q'));cd=ast.literal_eval(assignment(cv,'COST_DRAW'));assert cq==1.6 and cd==.1
        budgets=ast.literal_eval(assignment(lb,'CURVE_BUDGETS'));assert list(budgets)==p['budgets']
        draw=next(n for n in cv.body if isinstance(n,ast.FunctionDef) and n.name=='draw_all');batch=ast.literal_eval(draw.args.defaults[-1]);assert batch==50000
        overrides=assignment(lb,'OVERRIDES');assert isinstance(overrides,ast.Dict)
        choices=[v for k,v in zip(overrides.keys,overrides.values,strict=True) if isinstance(k,ast.Constant) and k.value=='control_variate'];assert len(choices)==1 and isinstance(choices[0],ast.Call)
        factor_nodes=[x.value for x in choices[0].keywords if x.arg=='cost_factor'];assert len(factor_nodes)==1
        expr=factor_nodes[0];assert isinstance(expr,ast.BinOp) and isinstance(expr.op,ast.Div)
        numerator=ast.literal_eval(expr.left);denominator=ast.literal_eval(expr.right);assert (numerator,denominator)==(3.3,1.6)
        factor=numerator/denominator;c=cq+cd
        alloc=json.loads(Path(p['allocation']).read_text())['plain'];assert len(alloc)==7 and all(isinstance(a,float) and math.isfinite(a) and a>0 for a in alloc)
        result['constants']=dict(c_hex=c.hex(),factor_hex=factor.hex(),cost_q_hex=cq.hex(),cost_draw_hex=cd.hex(),allocation_hex=[a.hex() for a in alloc],batch=batch,fpe_flops=6000,rule5_extra=F(1,25))
        controls=[]
        for m,expected in [(1,50000),(50000,50000),(50001,100000),(100000,100000)]:
            log(dict(stage='control_attempt',M=m));value=batch_floor(m,batch);row=dict(M=m,expected=expected,value=value,passed=value==expected);controls.append(row);log(dict(stage='control_result',row=row));assert row['passed'];cap()
        rows=[]
        for budget in budgets:
            cap();log(dict(stage='budget_attempt',budget=budget));assert isinstance(budget,int) and budget>0
            n=int(budget//c);assert 0<=n<2**53
            products=[a*n for a in alloc];assert all(float(F(a)*n)==v for a,v in zip(alloc,products,strict=True))
            quotas=[max(4,math.floor(x)) for x in products];M=sum(quotas);assert 0<M<2**53
            product=M*c;assert product==float(F(c)*M)
            raw=math.ceil(product);charged=raw*factor;assert math.isfinite(charged) and charged==float(F(factor)*raw)
            report=F(charged);dmin=batch_floor(M,batch);assert dmin>=M and dmin%batch==0
            dense=F(7040*M,6000);classic=F(6820*M,6000);extra=F(dmin-M,25)
            primary=dense+extra;alternate=classic+extra
            row=dict(budget=budget,n_source=n,n_exact_binary_c_floor=F(budget)//F(c),product_hex=[x.hex() for x in products],quotas=quotas,M=M,D_min=dmin,
                     raw_product_hex=product.hex(),raw_charge=raw,reported_hex=charged.hex(),reported_exact=report,
                     canonical_dense_flops=7040*M,classic_dot_units=6820*M,canonical_dense_FPE=dense,classic_dot_FPE=classic,extra_candidate_FPE=extra,
                     conditional_canonical_FPE=primary,conditional_classic_FPE=alternate,conditional_gap=primary-report,
                     dense_only_status='floor_exceeds_report' if dense>report else 'inconclusive',
                     conditional_status='conditional_undercharge' if primary>report else 'inconclusive',
                     classic_conditional_status='conditional_undercharge' if alternate>report else 'inconclusive',
                     displayed=dict(reported=float(report),canonical_dense=float(dense),extra=float(extra),conditional=float(primary),gap=float(primary-report)))
            rows.append(row);log(dict(stage='budget_result',row=row));cap()
        assert len(rows)==6 and [r['budget'] for r in rows]==p['budgets'] and len(controls)==4
        result.update(complete=True,rows=rows,controls=controls,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      scope=p['scope'],new_author_estimator_model_RNG_native_reference_calls=0,
                      accounting='Six budget arithmetic contractions plus four integer batch checks; source/advice decoding and proof inherited. No runtime/full-cost benchmark.')
        log(dict(stage='complete'));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=True,cpu=result['cpu_seconds'],rows=[dict(budget=r['budget'],M=r['M'],status=r['conditional_status'],**r['displayed']) for r in rows])))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);log(dict(stage='failure',message=repr(e)));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise
if __name__=='__main__':main()
