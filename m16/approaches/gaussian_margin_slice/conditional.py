"""Analytic one-dimensional Gaussian residual law on a certified gate cell."""
from pathlib import Path
from fractions import Fraction as F
import json,time,hashlib
import mpmath as mp
HERE=Path(__file__).parent

def num(x):
    if isinstance(x,F):return mp.mpf(x.numerator)/x.denominator
    return mp.mpf(x)
def normal_interval(a,b):
    if not a<b:raise ValueError('empty interval')
    if a>=0:return (mp.erfc(a/mp.sqrt(2))-mp.erfc(b/mp.sqrt(2)))/2
    return (mp.erfc(-b/mp.sqrt(2))-mp.erfc(-a/mp.sqrt(2)))/2

def conditional(mu,variance,a,b,cut):
    mu,variance,a,b,cut=map(num,(mu,variance,a,b,cut))
    if variance<=0:raise ValueError('nonpositive variance or degenerate direction')
    if not a<b:raise ValueError('empty interval')
    sd=mp.sqrt(variance);lo=(a-mu)/sd;hi=(b-mu)/sd;Z=normal_interval(lo,hi)
    P=mp.mpf(0) if cut<=a else mp.mpf(1) if cut>=b else normal_interval(lo,(cut-mu)/sd)/Z
    phi=lambda z:mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)
    mean=mu+sd*(phi(lo)-phi(hi))/Z
    plug=mp.mpf(int(mean<=cut))
    return dict(correct_probability=P,truncated_mean=mean,normal_segment_mass=Z,mean_replacement_correct=plug,mean_replacement_bias=plug-P,mean_minus_boundary=mean-cut)

def controls():
    tiny=mp.mpf('1e-55');r=conditional(0,1,-1,1,0);assert abs(r['truncated_mean'])<tiny and abs(r['correct_probability']-mp.mpf('.5'))<tiny
    r=conditional(F(3,10),F(36,25),F(-2,5),F(17,10),F(4,5));q=conditional(F(-3,10),F(36,25),F(-17,10),F(2,5),F(-4,5));assert abs(r['correct_probability']+q['correct_probability']-1)<tiny and abs(r['truncated_mean']+q['truncated_mean'])<tiny
    assert conditional(0,1,-1,1,-2)['correct_probability']==0 and conditional(0,1,-1,1,2)['correct_probability']==1
    rejected=0
    for args in ((0,0,-1,1,0),(0,1,1,1,0)):
        try:conditional(*args)
        except ValueError:rejected+=1
    assert rejected==2
    return dict(symmetry=True,reflection=True,outside_clamps=True,degenerate_rejections=rejected)

def main():
    stamp=time.process_time()
    for name,sha in json.loads((HERE/'manifest.json').read_text()).items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
    c=json.loads((HERE/'certificate.json').read_text())['certificate'];m=json.loads((HERE/'margin_results.json').read_text());a=list(map(F,c['pair'][0]['x']));b=list(map(F,c['pair'][1]['x']));d=[y-x for x,y in zip(a,b)];D=sum(x*x for x in d);assert D>0;mu=-sum(x*y for x,y in zip(a,d))/D;variance=1/D;cut=F(m['crossing_parameter']);q=[x+mu*y for x,y in zip(a,d)];assert sum(x*y for x,y in zip(q,d))==0
    for l in (F(0),F(1,2),F(1)):
        assert sum((x+l*y)**2 for x,y in zip(a,d))==sum(x*x for x in q)+D*(l-mu)**2
    results=[]
    for precision in (60,90):
        with mp.workdps(precision):
            ctrl=controls();rows=[]
            for l,r in ((F(0),F(1)),(F(1,2),F(1))):
                z=conditional(mu,variance,l,r,cut);rows.append(dict(interval=[str(l),str(r)],**{k:mp.nstr(v,precision-5) for k,v in z.items()}))
            results.append(dict(dps=precision,controls=ctrl,rows=rows))
    with mp.workdps(100):
        diffs=[abs(mp.mpf(r[k])-mp.mpf(s[k])) for r,s in zip(results[0]['rows'],results[1]['rows']) for k in r if k!='interval'];assert max(diffs)<mp.mpf('1e-50');delta=mp.nstr(max(diffs),8)
    out=dict(mu=str(mu),variance=str(variance),direction_squared_norm=str(D),crossing=str(cut),exact_completion_of_square_controls=3,orthogonality_exact=True,evaluations=results,max_precision_difference=delta,cpu_seconds=time.process_time()-stamp,scope='Regular conditional Gaussian line law and two truncated slices; no global mass/accuracy or certified transcendental rounding bound.')
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(rows=results[-1]['rows'],mu=float(mu),variance=float(variance),cpu_seconds=out['cpu_seconds'],max_precision_difference=delta)))
if __name__=='__main__':main()
