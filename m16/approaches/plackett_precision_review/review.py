"""Separate finite-rule bias from arithmetic/correlation rounding."""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import ast,hashlib,json,math,time,types
import mpmath as mp
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review')
HASHES={'negative_mass_fixture.json':'ddd0ff28a186837d9f96e2e9413916de1cc674634fc2f84518a74a6472b0fd1b',
        'estT_snapshot.py':'afa94627ff5b860a4b922dc340e90d0d6718c770d74e326dfbf667a2bb0bbc3a',
        'captured.npz':'02f3a5b837e73bdfdac6b93d85be100ddd049664e312ef92a6f603eea861b95d'}


def load_source():
    for name,sha in HASHES.items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==sha
    source=(SOURCE/'estT_snapshot.py').read_text();tree=ast.parse(source);parts=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('_rule','orthant'):parts.append(ast.get_source_segment(source,node))
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_RULES' for t in node.targets):parts.append(ast.get_source_segment(source,node))
    assert len(parts)==3
    module=types.ModuleType('unchanged_source_orthant');module.__dict__.update(np=np,math=math,STATS={'simplices':0})
    exec(compile('\n\n'.join(parts),str(SOURCE/'estT_snapshot.py')+' [unchanged mass-only extraction]','exec'),module.__dict__)
    return module


def exact_float(x):
    f=Fraction(float(x));return mp.mpf(f.numerator)/f.denominator


def mp_matrix(a):
    return mp.matrix([[exact_float(v) for v in row] for row in a])


def correlation(rays,stats):
    B=mp_matrix(rays).T**-1;stats['correlation_inversions']+=1
    S=B*B.T;d=[mp.sqrt(S[i,i]) for i in range(4)]
    return mp.matrix([[S[i,j]/(d[i]*d[j]) for j in range(4)] for i in range(4)])


def rounded_rule(rule):
    return [tuple(exact_float(x) for x in row) for row in rule]


def ideal_rule(n,stats):
    start=time.process_time();x,w=mp.gauss_quadrature(n,'legendre');stats['ideal_legendre_rules']+=1
    stats['ideal_rule_cpu_seconds']+=time.process_time()-start
    return [(1-(1-(x[i]+1)/2)**3,w[i]/2,3*(1-(x[i]+1)/2)**2) for i in range(n)]


def asin_checked(x,where,stats):
    if not -1<=x<=1:raise ArithmeticError('inadmissible asin argument '+where+': '+mp.nstr(x,40))
    stats['arcsines']+=1;return mp.asin(x)


def plackett(C,rule,stats):
    pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    base=mp.mpf(1)/16+sum(asin_checked(C[i,j],'base',stats) for i,j in pairs)/(8*mp.pi)
    acc=mp.mpf(0)
    for t,weight,jac in rule:
        Rt=mp.eye(4)+t*(C-mp.eye(4))
        mp.cholesky(Rt);stats['positive_definite_checks']+=1
        Om=Rt**-1;stats['transformed_inversions']+=1
        for i,j in pairs:
            k,l=[q for q in range(4) if q!=i and q!=j]
            product=Om[k,k]*Om[l,l]
            if product<=0:raise ArithmeticError('nonpositive inverse-diagonal product')
            pc=-Om[k,l]/mp.sqrt(product);r=C[i,j]
            denominator=1-(t*r)**2
            if denominator<=0:raise ArithmeticError('nonpositive Plackett denominator')
            acc+=weight*jac*r*asin_checked(pc,'conditional',stats)/mp.sqrt(denominator)
        stats['quadrature_nodes']+=1
    integral=acc/(4*mp.pi**2)
    return dict(base=str(base),integral=str(integral),mass=str(base+integral))


def controls():
    start=time.process_time();source=load_source();mp.mp.dps=80;stats=defaultdict(float);rows=[]
    identity=mp.eye(4);block=mp.eye(4);block[0,1]=block[1,0]=mp.mpf(1)/2;block[2,3]=block[3,2]=-mp.mpf(1)/3
    truth=(mp.mpf(1)/4+mp.asin(mp.mpf(1)/2)/(2*mp.pi))*(mp.mpf(1)/4-mp.asin(mp.mpf(1)/3)/(2*mp.pi))
    for name,C,expected in (('identity',identity,mp.mpf(1)/16),('independent_blocks',block,truth)):
        for rule_name in ('wide','thin'):
            result=plackett(C,rounded_rule(source._RULES[rule_name]),stats)
            error=abs(mp.mpf(result['mass'])-expected);assert error<mp.mpf('1e-12')
            rows.append(dict(name=name,rule=rule_name,expected=str(expected),error=str(error),**result))
    return dict(rows=rows,work=dict(stats),cpu_seconds=time.process_time()-start,source_hashes=HASHES)


class Ops:
    def __init__(self):self.n=0
    def add(self,n):self.n+=n


def float_diagnostics(C,rule):
    pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    base_out=[(i,j,float(C[i,j])) for i,j in pairs if abs(C[i,j])>1]
    conditional_out=[];denominator_clamps=0
    for index,(t,weight,jac) in enumerate(rule):
        Om=np.linalg.inv(np.eye(4)+t*(C-np.eye(4)))
        for i,j in pairs:
            k,l=[q for q in range(4) if q!=i and q!=j]
            pc=-Om[k,l]/math.sqrt(Om[k,k]*Om[l,l])
            if abs(pc)>1:conditional_out.append(dict(node=index,pair=[i,j],argument=float(pc)))
            denominator_clamps+=int(1-(t*C[i,j])**2<1e-300)
    return dict(base_clipped_arguments=base_out,conditional_clipped_arguments=conditional_out,
                denominator_clamps=denominator_clamps,extra_diagnostic_inversions=len(rule))


def fixture():
    start=time.process_time();source=load_source();saved=json.loads((SOURCE/'negative_mass_fixture.json').read_text())
    rays=np.asarray(saved['rays']);captured=np.load(SOURCE/'captured.npz')
    assert np.array_equal(rays,captured['rays'][saved['index']])
    ops=Ops();float_value=source.orthant(rays,ops);assert abs(float_value-saved['source_value'])<1e-16
    B=np.linalg.inv(rays.T);S=B@B.T;d=np.sqrt(np.diag(S));C=S/np.outer(d,d)
    assert max(abs(C[i,j]) for i in range(4) for j in range(i+1,4))>.95
    assert abs(np.linalg.det(rays))>=1e-12
    diagnostics=float_diagnostics(C,source._RULES['thin']);rows=[]
    for precision in (80,100):
        mp.mp.dps=precision;stats=defaultdict(float);rule=rounded_rule(source._RULES['thin'])
        exactC=correlation(rays,stats)
        arms={
            'rounded_C_rounded_rule':plackett(mp_matrix(C),rule,stats),
            'exact_ray_C_rounded_rule':plackett(exactC,rule,stats),
            'exact_ray_C_ideal_rule':plackett(exactC,ideal_rule(48,stats),stats),
        }
        original=exact_float(float_value);reference=exact_float(saved['positive'])
        for name,result in arms.items():
            value=mp.mpf(result['mass']);result['minus_float_source']=str(value-original);result['minus_positive_reference']=str(value-reference)
        rows.append(dict(precision=precision,arms=arms,work=dict(stats),exact_ray_C=[[str(exactC[i,j]) for j in range(4)] for i in range(4)]))
    stability={name:str(abs(mp.mpf(rows[0]['arms'][name]['mass'])-mp.mpf(rows[1]['arms'][name]['mass']))) for name in rows[0]['arms']}
    assert max(mp.mpf(v) for v in stability.values())<mp.mpf('1e-65')
    return dict(index=saved['index'],source_value=float_value,source_named_ops=ops.n,rounded_C=C.tolist(),
                positive_reference=saved['positive'],positive_conditional_bound=saved['bound'],float_guard_diagnostics=diagnostics,
                rows=rows,precision_stability=stability,source_hashes=HASHES,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cpu_seconds=time.process_time()-start)


if __name__=='__main__':
    import sys
    name='controls' if len(sys.argv)==1 else sys.argv[1]
    if name not in ('controls','fixture'):raise ValueError('unknown frozen phase')
    try:result=controls() if name=='controls' else fixture()
    except Exception as exc:
        (HERE/(name+'_failure.json')).write_text(json.dumps(dict(failure=repr(exc)),indent=2)+'\n');raise
    (HERE/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','rounded_C')}))
