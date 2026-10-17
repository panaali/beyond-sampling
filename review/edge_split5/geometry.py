"""Certify actual binary64 edge splits; no mass evaluation."""
from pathlib import Path
from fractions import Fraction as F
import hashlib, importlib.util, json, time, resource
import numpy as np
P=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pinned():
    p=json.loads((P/'protocol.json').read_text())
    for name,h in p['pins'].items(): assert sha(name)==h, name
    return p

def primitive(p):
    path=Path(p['prior'])/'radial_bound/primitive.py'
    spec=importlib.util.spec_from_file_location('split5_radial',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def hexrows(R): return [[float(x).hex() for x in row] for row in R]
def rational(R): return [[F(float(x)) for x in row] for row in R]

class Rejected(ValueError): pass

def split(R, Exact, candidate=None):
    R=np.array(R,dtype=float,copy=True)
    if R.shape!=(5,5) or not np.isfinite(R).all(): raise Rejected('nonfinite_or_shape')
    with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
        raw=R[0]+R[1]
        if not np.isfinite(raw).all(): raise Rejected('nonfinite_sum')
        if not np.any(raw): raise Rejected('zero_sum')
        norm=float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm==0: raise Rejected('invalid_norm')
        v=raw/norm if candidate is None else np.array(candidate,dtype=float,copy=True)
    if v.shape!=(5,) or not np.isfinite(v).all(): raise Rejected('nonfinite_candidate')
    Q=Exact();A=rational(R);V=[F(float(x)) for x in v]
    for row in A:
        for x in row:Q.record(x)
    det=Q.determinant(A)
    if det==0: raise Rejected('singular_parent')
    coeff=Q.solve([list(row) for row in zip(*A)],V)
    if any(coeff[2:]): raise Rejected('off_edge_span')
    a,b=coeff[:2]
    if a<=0 or b<=0: raise Rejected('nonpositive_edge_coefficient')
    for j in range(5):
        assert Q.total(Q.mul(coeff[i],A[i][j]) for i in range(5))==V[j]
    children=[];child_dets=[]
    for i,c in enumerate((a,b)):
        C=R.copy();C[i]=v;cdet=Q.determinant(rational(C))
        assert cdet==Q.mul(c,det) and cdet!=0
        children.append(hexrows(C));child_dets.append(str(cdet))
    # The same positive dual section cuts parent and both children.
    w=Q.solve(A,[F(1)]*5)
    wv=Q.total(Q.mul(x,y) for x,y in zip(w,V))
    assert wv==Q.add(a,b)>0
    ratios=[Q.div(a,wv),Q.div(b,wv)]
    assert Q.total(ratios)==1
    # Separate raw-sum and normalization rounding from actual-span acceptance.
    exact_sum=[Q.add(A[0][j],A[1][j]) for j in range(5)]
    addition_error=[Q.sub(F(float(raw[j])),exact_sum[j]) for j in range(5)]
    normalization_error=[Q.sub(V[j],Q.div(F(float(raw[j])),F(norm))) for j in range(5)]
    return dict(parent_hex=hexrows(R),candidate_hex=[float(x).hex() for x in v],
        raw_sum_hex=[float(x).hex() for x in raw],norm_hex=norm.hex(),
        coefficients=list(map(str,coeff)),parent_determinant=str(det),
        child_determinants=child_dets,section_dual=list(map(str,w)),
        section_ratios=list(map(str,ratios)),children_hex=children,
        exact_sum=list(map(str,exact_sum)),addition_error=list(map(str,addition_error)),
        normalization_error=list(map(str,normalization_error)),
        exact_counts=dict(Q.counts),max_bits=Q.max_bits,
        floating_setup_named={'addition':5,'norm_multiply':5,'norm_add':4,'sqrt':1,'divide':5},
        scope='actual binary rays; section fractions are not Gaussian masses')

def controls(Exact, rows, save):
    I=np.eye(5);cases=[]
    R=I.copy();R[1]=-R[0];cases.append(('zero_sum',R,None,'zero_sum'))
    R=I.copy();R[0,0]=np.inf;cases.append(('nonfinite',R,None,'nonfinite_or_shape'))
    R=I.copy();R[1]=R[0];cases.append(('singular_parent',R,None,'singular_parent'))
    cases.append(('endpoint',I,I[0],'nonpositive_edge_coefficient'))
    v=(I[0]+I[1])/np.sqrt(2);v[2]=2.**-52;cases.append(('off_edge_span',I,v,'off_edge_span'))
    cases.append(('norm_overflow',I*1e308,None,'invalid_norm'))
    cases.append(('norm_underflow',I*1e-308,None,'invalid_norm'))
    for name,R,v,expected in cases:
        try: split(R,Exact,v)
        except Rejected as e: result=str(e)
        else: result='unexpected_acceptance'
        rows.append(dict(name=name,expected=expected,result=result,pass_control=result==expected,
            rays_hex=hexrows(R),candidate_hex=None if v is None else [float(x).hex() for x in v]))
        save()
    assert all(x['pass_control'] for x in rows)
    return rows

def main():
    out=P/'geometry.json';assert not out.exists()
    start=time.process_time()
    result=dict(complete=False,protocol_sha256=sha(P/'protocol.json'),source_sha256=sha(__file__),
        controls=[],rows=[],accepted=0,
        exclusions='Python control, imports, memory, exact bit-complexity and compiled norm internals not priced; rejected paths do not return exact-operation counts.')
    def save():
        result.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        out.write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        p=pinned();m=primitive(p)
        fixtures=json.loads((Path(p['prior'])/'results.json').read_text())['fixtures']
        assert [f['name'] for f in fixtures]==p['fixture_names']
        controls(m.Exact,result['controls'],save)
        for f in fixtures:
            entry=dict(name=f['name'],status='started',parent_hex=f['rays_hex']);result['rows'].append(entry);save()
            R=[[float.fromhex(x) for x in row] for row in f['rays_hex']]
            try: entry.update(status='accepted',**split(R,m.Exact))
            except Rejected as e: entry.update(status='unresolved',reason=str(e))
            result['accepted']=sum(x['status']=='accepted' for x in result['rows']);save()
        result['complete']=True
    except BaseException as e: result['failure']=repr(e);raise
    finally:save();print(json.dumps({k:result[k] for k in ('complete','accepted','cpu','max_rss_kib')}))
if __name__=='__main__': main()
