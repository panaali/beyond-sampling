"""Saved exact partition and independent Gaussian-CDF checks, without forwards."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys
import mpmath as mp
sys.set_int_max_str_digits(0);mp.mp.dps=180
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/prefix_wedge_remainder/limit')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def m(q):
    q=F(q);return mp.mpf(q.numerator)/q.denominator
def endpoints(d):return F(d['lower']),F(d['upper'])
def inside(x,lo,hi):return (lo is None or lo<x) and (hi is None or x<hi)
def endpoint(x):return None if x is None else F(x)
def cdf(side,x,upper):
    if x is None:return mp.mpf(F(1,8).numerator)/8 if upper and side=='10' else (mp.mpf(3)/8 if upper else mp.mpf(0))
    c=m(x);sign=-1 if side=='10' else 1
    # Independent arcsine orthant form, rather than author's transformed atan.
    return mp.mpf(1)/8+(mp.atan(c)+sign*(mp.pi/4+mp.asin(c/mp.sqrt(2*(1+c*c)))))/(4*mp.pi)
def check(parts,mass):
    A=list(map(F,parts['A']));B=list(map(F,parts['B']));W=[list(map(F,row)) for row in parts['Wo']]
    roots=sorted(set(-a/b for a,b in zip(A,B) if b));assert roots==list(map(F,parts['gate_roots']))
    cells=parts['intervals'];assert cells[0]['lower'] is None and cells[-1]['upper'] is None
    bounds=[]
    for i,row in enumerate(cells):
        lo,hi=endpoint(row['lower']),endpoint(row['upper'])
        if i:assert lo==endpoint(cells[i-1]['upper'])
        if hi is not None:bounds.append(hi)
        assert not any(inside(r,lo,hi) for r in roots)
        mask=row['mask'];aff=[(sum(w*a*g for w,a,g in zip(wr,A,mask)),sum(w*b*g for w,b,g in zip(wr,B,mask))) for wr in W]
        assert aff==[tuple(map(F,x)) for x in row['output_affines']]
        for j in range(4):
            for k in range(j):
                u,v=aff[j][0]-aff[k][0],aff[j][1]-aff[k][1]
                assert not v or not inside(-u/v,lo,hi)
        for rep in row['representatives']:
            x=F(rep['r']);assert inside(x,lo,hi)
            h=[max(F(0),a+b*x) for a,b in zip(A,B)]
            scores=[sum(w*v for w,v in zip(wr,h)) for wr in W]
            winner=next(j for j,s in enumerate(scores) if s==max(scores))
            assert [a+b*x>0 for a,b in zip(A,B)]==mask and h==list(map(F,rep['hidden']))
            assert scores==list(map(F,rep['scores'])) and winner==rep['winner']==row['winner']
    assert bounds==sorted(set(bounds)) and set(roots)<=set(bounds)
    assert list(map(F,(b['r'] for b in parts['boundaries'])))==bounds
    for b in parts['boundaries']:
        x=F(b['r']);h=[max(F(0),a+z*x) for a,z in zip(A,B)]
        scores=[sum(w*v for w,v in zip(wr,h)) for wr in W];winners=[j for j,v in enumerate(scores) if v==max(scores)]
        assert b['maximizers']==winners and b['winner']==winners[0] and h==list(map(F,b['hidden'])) and scores==list(map(F,b['scores']))
    expected=[];value=mp.mpf(0);rational_lo=rational_hi=F(0)
    for i,row in enumerate(cells):
        if row['winner']!=2:continue
        for side in ('10','01'):
            lo,hi=endpoint(row['lower']),endpoint(row['upper'])
            if side=='10':hi=min(hi,F(1)) if hi is not None else F(1)
            else:lo=max(lo,F(1)) if lo is not None else F(1)
            if lo is not None and hi is not None and lo>=hi:continue
            expected.append((i,side,lo,hi))
            c=mass['contributions'][len(expected)-1]
            assert (c['interval_index'],c['rank_side'],endpoint(c['lower']),endpoint(c['upper']))==expected[-1]
            v0,v1=cdf(side,lo,False),cdf(side,hi,True)
            for recorded,true in ((c['cdf_lower'],v0),(c['cdf_upper'],v1)):
                l,h=endpoints(recorded);assert m(l)-mp.mpf("1e-170")<=true<=m(h)+mp.mpf("1e-170")
            l,h=endpoints(c['joint_mass']);assert m(l)-mp.mpf("1e-170")<=(v1-v0)/2<=m(h)+mp.mpf("1e-170")
            rational_lo+=l;rational_hi+=h;value+=(v1-v0)/2
    assert len(expected)==len(mass['contributions'])
    lo,hi=endpoints(mass['joint_correctness_mass'])
    assert (lo,hi)==(max(F(0),rational_lo),min(F(1,12),rational_hi)) and m(lo)-mp.mpf("1e-170")<=value<=m(hi)+mp.mpf("1e-170")
    return value
def main():
    d=read(P/'controls.json');assert d['complete'] and d['network_forward_MC_calls']==0
    for name,h in d['source_sha256'].items():assert sha(P/name)==h
    assert len(d['rounding_controls'])==4 and len(d['rows'])==9
    for r in d['rounding_controls']:
        a,b=endpoints(r['input']);l,h=endpoints(r['output']);scale=2**160
        assert l==F((a.numerator*scale)//a.denominator,scale)
        assert h==F(-((-b.numerator*scale)//b.denominator),scale)
    truths=dict(zero=F(0),always_target=F(1,12),earlier_tie=F(0),later_tie=F(1,12),positive_ratio=F(5,96),negative_ratio=F(1,32),right_rank_boundary=F(1,24),left_rank_boundary=F(1,48),disjoint_tails=F(1,16))
    for row in d['rows']:
        value=check(row['partition'],row['mass']);truth=truths[row['name']]
        assert F(row['exact_truth'])==truth and abs(value-m(truth))<mp.mpf('1e-170')
        lo,hi=endpoints(row['mass']['joint_correctness_mass']);assert lo<=truth<=hi
    out=dict(complete=True,analytic_partitions=9,dyadic_controls=4,controls_sha256=sha(P/'controls.json'),new_forward_quadrature_calls=0,
             scope='Exact saved root completeness, affine coefficients, first-index interval/boundary ownership, dyadic rounding and independent 180-digit arcsine CDF contractions. Primitive interval source theorem supplies rigor; no trained application or original model accuracy claimed.')
    (HERE/'reset_limit_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
