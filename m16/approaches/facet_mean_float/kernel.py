"""Direct binary64 numerical evaluator of previously exact-prepared facets."""
import math
from fractions import Fraction
from collections import Counter

def evaluate(certificate):
    counts=Counter();terms=[];out=dict(status='unresolved',value=None,terms=terms,counts=counts)
    if certificate['status'] in ['zero_output','opposite_measure_zero']:
        return dict(status=certificate['status'],value=0.0,terms=[],counts={})
    if certificate['status']!='prepared':raise ValueError('unknown exact-prepared status')
    def f(pair):
        counts['exact_rational_to_binary64']+=1
        z=float(Fraction(int(pair[0]),int(pair[1])))
        counts['finite_checks']+=1
        if not math.isfinite(z):raise ValueError('nonfinite converted coefficient')
        return z
    def root(x):counts['sqrt']+=1;return math.sqrt(x)
    try:
        counts['multiplications']+=1;twopi=2.0*math.pi;normalizer=root(twopi);total=0.0
        for facet in certificate['facets']:
            k=facet['other_count']
            if k==0:p=1.0
            elif k==1:p=.5
            elif k==2:
                rad=f(facet['angle_radicand']);b=f(facet['projected_dot']);counts['comparisons']+=1
                if rad<0:raise ValueError('negative converted radicand')
                counts['negations']+=1;counts['atan2']+=1;angle=math.atan2(root(rad),-b);counts['divisions']+=1;p=angle/twopi
            else:raise ValueError('more than two projected constraints')
            num=f(facet['v_dot_normal']);norm=f(facet['norm2']);counts['comparisons']+=1
            if norm<=0:raise ValueError('nonpositive converted norm')
            counts['divisions']+=2;counts['multiplications']+=1;term=(num/root(norm))*p/normalizer
            counts['additions']+=1;total=total+term;terms.append(dict(probability=p,contribution=term))
        out['value']=total;counts['finite_checks']+=1;counts['comparisons']+=1
        out['status']='ok' if math.isfinite(total) and total>=0 else 'nonfinite_or_negative'
    except (ValueError,ZeroDivisionError,OverflowError) as e:out.update(status='arithmetic_failure',exception=type(e).__name__+': '+str(e))
    out['counts']=dict(counts);return out
