"""Heuristic signed-conditioning trigger and charged MP80 evaluation."""
from pathlib import Path
import importlib.util,math
from fractions import Fraction as F
from decimal import Decimal
SOURCE=Path('/workspace/work/recover-small_h2_34/m16/approaches/third_step_means/primitive.py')
spec=importlib.util.spec_from_file_location('pinned_facet_mp',SOURCE);mp_kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(mp_kernel)

def select(base):
    if base['status'] in ['zero_output','opposite_measure_zero']:return dict(fallback=False,reason='exact_zero_status',sumabs=0.0)
    v=base['value']
    if base['status']!='ok' or v is None or not math.isfinite(v) or v<=0:return dict(fallback=True,reason='failed_or_nonpositive_prepared',sumabs=None)
    s=0.0
    for t in base['terms']:s=s+abs(t['contribution'])
    indicator=(64.0*2.0**-53)*s;threshold=1e-10*abs(v)
    return dict(fallback=indicator>threshold,reason='signed_conditioning',sumabs=s,indicator=indicator,threshold=threshold,heuristic=True)

def rescue(certificate,base):
    decision=select(base);out=dict(decision=decision,base=base,status=base['status'],value=base['value'],mp_evaluations=0,rational_decodes=0)
    if not decision['fallback']:return out
    decoded=dict(status=certificate['status'],facets=[])
    for facet in certificate.get('facets',[]):
        row=dict(other_count=facet['other_count'])
        keys=['norm2','v_dot_normal']+(['angle_radicand','projected_dot'] if facet['other_count']==2 else [])
        for k in keys:row[k]=F(*map(int,facet[k]));out['rational_decodes']+=1
        decoded['facets'].append(row)
    try:
        out['mp_evaluations']=1;answer=mp_kernel.evaluate(decoded,80);out['mp80']=answer;v=float(answer['value']);out['value']=v
        if not math.isfinite(v) or v<0:out['status']='mp80_nonfinite_or_negative'
        elif v==0 and Decimal(answer['value'])!=0:out['status']='mp80_float_underflow'
        else:out['status']='mp80_ok'
    except (ValueError,ZeroDivisionError,OverflowError) as e:out.update(status='mp80_arithmetic_failure',exception=type(e).__name__+': '+str(e),value=None)
    return out

def evaluate(certificate):
    from kernel import evaluate as direct
    return rescue(certificate,direct(certificate))
