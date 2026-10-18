"""Only the frozen32 unresolved cases; exact bound work is explicit review cost."""
import time
START=time.process_time()
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import numpy as np
import mpmath as mp
from exact_tracking import Tracker,Tracked,acute_certificate

HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h2_34/m16/approaches')
PINS={
 BASE/'class_region_t4/quadratic_exact_mass_calls.npz':'07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581',
 BASE/'analytic_inner_mass/aggregate.json':'88f1969fbcb20f95e21fafc4332eb5ab6f153d88ed8f210d5ddddf4b40309488',
 BASE/'analytic_inner_mass/aggregate_values.npz':'879d18f30ee4dfdd1608c2e03bcccdf1e17b2b0af51f81ba42cc988de7464e13',
 BASE/'analytic_inner_mass/primitive.py':'30919d71e286ccc707b5acac5fbb5fa9d9a75b15d03d94b7d10267e37e4b0542',
 BASE/'radial_t4_reference/radial.py':'067e21bddd4c895653e0f3242a9521c7f05650671fc3bd8d256b4d433169e76b',
 BASE/'radial_t4_reference/combined_values.npz':'a6e3ed06492b1b6d7292c850c9cdedd5200b8d4c5bc21ce347b2bc91d398671b',
 BASE/'positive_mass_budget/aggregate.json':'ff1fe727115344d07e0ca728541c7e3a0950bf46520359d8bbe66df7bec21dc8',
 BASE/'positive_mass_budget/primitive.py':'0397ce32b78968c603f9d90faee276a790851b92b5c9600d6ae1f89b86780107'}
ORDERS=(2,3,4,6,8,12,16,20,24,32)
TOLERANCE=F(1,10**7*6347)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def load():
    for p,h in PINS.items():assert sha(p)==h,str(p)
    radial=module('pinned_radial',BASE/'radial_t4_reference/radial.py');radial.F=Tracked
    kernel=module('pinned_inner',BASE/'analytic_inner_mass/primitive.py')
    old=module('pinned_positive',BASE/'positive_mass_budget/primitive.py')
    return radial,kernel,old

def mfrac(value):
    f=F(value);return mp.mpf(f.numerator)/f.denominator
def bound(cert,old_det,n,dps=80):
    with mp.workdps(dps):
        a,b=mfrac(cert['a']),mfrac(cert['b']);det=abs(mfrac(cert['det']));assert 0<a<=b
        alpha=(a+b)/2;delta=mp.sqrt(a*b);rho=(mp.sqrt(b)-mp.sqrt(a))/(mp.sqrt(b)+mp.sqrt(a));N=n-2
        tail=2*rho**(N+1)/delta**3*(alpha/(1-rho)+delta*((N+1)/(1-rho)+rho/(1-rho)**2))
        quadrature=det*tail/(6*mp.pi**2)
        determinant=abs(mfrac(F(abs(old_det)))-det)/(12*mp.pi**2*a*a)
        total=quadrature+determinant
        return dict(order=n,quadrature_bound=float(quadrature),determinant_factor_bound=float(determinant),
                    total_bound=float(total),total_decimal=mp.nstr(total,dps),qualifies=bool(total<=mfrac(TOLERANCE)))

def choose(cert,old_det):
    trials=[]
    for n in ORDERS:
        row=bound(cert,old_det,n);trials.append(row)
        if row['qualifies']:return n,trials
    return None,trials

def one(R,old_det,radial,kernel,rules):
    tracker=Tracker();Tracked.current=tracker;start=time.process_time();cert=acute_certificate(R,radial)
    certificate_cpu=time.process_time()-start;stages=[];order=None
    if cert['acute']:
        start=time.process_time();order,trials=choose(cert,old_det);selection_cpu=time.process_time()-start
        stages.append(dict(method='acute',certificate=cert,trials=trials,certificate_cpu=certificate_cpu,selection_cpu=selection_cpu))
    else:
        stages.append(dict(method='acute_rejected',certificate=cert,certificate_cpu=certificate_cpu))
    if order is None:
        start=time.process_time();cert=radial.bounds(R);certificate_cpu=time.process_time()-start
        start=time.process_time();order,trials=choose(cert,old_det);selection_cpu=time.process_time()-start
        stages.append(dict(method='exact_kkt',certificate=cert,trials=trials,certificate_cpu=certificate_cpu,selection_cpu=selection_cpu))
    out=dict(stages=stages,exact_operation_inventory=tracker.report(),old_determinant=old_det,order=order,mode='unresolved')
    if order is not None:
        start=time.process_time();result=kernel.integrate(R,order,old_det,rules);out['kernel_cpu']=time.process_time()-start
        out.update(result);out['mode']='returned'
        selected=stages[-1]['trials'][-1];independent_precision=bound(cert,old_det,order,100)
        with mp.workdps(100):
            combined=mp.mpf(selected['total_decimal'])+mp.mpf(result['series_error_bound'])
            out['conditional_bound']=math.nextafter(float(combined),math.inf)
            out['conditional_bound_decimal']=mp.nstr(combined,90)
            out['bound_80_to_100_difference']=selected['total_bound']-independent_precision['total_bound']
            if combined>mfrac(TOLERANCE):out.update(mode='unresolved',reason='series remainder exceeded fixed target')
    else:out['reason']='Exact radial alternative still exceeds fixed finite-order target'
    return out

def save(name,out,rules):
    out.update(cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               cold_rule_events=rules.events,cold_rule_counts=rules.ledger.dict(),source_hashes={str(p):sha(p) for p in (*PINS,HERE/'run.py',HERE/'exact_tracking.py')})
    (HERE/name).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')

def controls():
    radial,kernel,old=load();rules=kernel.Rules();eye=np.eye(4)
    nonacute=np.array([[1,0,0,0],[-2,1,0,0],[0,0,1,0],[0,0,0,1]],float)
    boundary=np.array([[1,0,0,0],[2,1,0,0],[2,0,1,0],[2,0,0,1]],float)
    bases=[('orthant',eye,F(1,4),1/16),('diagonal',np.diag([1.,2.,3.,4.]),F(144,205),1/16),
           ('nonacute',nonacute,F(1,12),math.atan2(1,-2)/(8*math.pi)),('boundary_minimum',boundary,F(1),None)]
    rows=[]
    for name,A,minimum,truth in bases:
        for power in (-5,0,5):
            for shift in range(4):
                R=np.roll(A,shift,axis=0)*2.**power
                # Independent known minimum checks on every exact certificate, separate validation inventory.
                validation=Tracker();Tracked.current=validation;cert=radial.bounds(R)
                assert F(cert['a'])==minimum*F(2)**(2*power)
                old_det=abs(float(old.determinant(R.tolist())))
                row=one(R,old_det,radial,kernel,rules)
                acute=row['stages'][0]['certificate'];assert F(acute['b'])>=F(cert['a'])
                if acute['acute']:assert F(acute['a'])<=F(cert['a'])
                row.update(name=name,power=power,cyclic_shift=shift,exact_minimum=cert['a'],validation_inventory=validation.report(),truth=truth)
                if truth is not None and row['mode']=='returned':
                    row['error']=row['value']-truth;row['passes']=abs(row['error'])<=row['conditional_bound']+5e-14
                    assert row['passes']
                rows.append(row)
    out=dict(complete=True,controls=rows,returned=sum(r['mode']=='returned' for r in rows),unresolved=sum(r['mode']=='unresolved' for r in rows))
    save('controls.json',out,rules);print(json.dumps({k:v for k,v in out.items() if k not in ('controls','source_hashes','cold_rule_events')}))

def trained(chunk):
    radial,kernel,old=load();rules=kernel.Rules()
    previous=json.loads((BASE/'analytic_inner_mass/aggregate.json').read_text());indices=previous['unresolved_indices'];assert len(indices)==32
    original=json.loads((BASE/'positive_mass_budget/aggregate.json').read_text());rays=np.load(BASE/'class_region_t4/quadratic_exact_mass_calls.npz')['rays']
    reference=np.load(BASE/'radial_t4_reference/combined_values.npz');assert 0<=chunk<4
    out=dict(complete=False,chunk=chunk,indices=indices[8*chunk:8*(chunk+1)],rows=[])
    for index in out['indices']:
        if time.process_time()-START>18:out['stop_reason']='18CPU soft cap';break
        base=original['rows'][index];assert base['index']==index and base['mode']=='unresolved'
        row=one(rays[index],base['det'],radial,kernel,rules)
        row.update(index=index,previous_a=base['a'],previous_b=base['b'],previous_last_trial=base['trials'][-1],
                   reference=float(reference['values'][index]),reference_bound=float(reference['bounds'][index]))
        if row['mode']=='returned':
            row['error']=row['value']-row['reference'];row['comparison_pass']=abs(row['error'])<=row['conditional_bound']+row['reference_bound']+5e-14
        out['rows'].append(row);save(f'chunk_{chunk}.json',out,rules)
    else:out['complete']=True
    save(f'chunk_{chunk}.json',out,rules)
    print(json.dumps(dict(chunk=chunk,cases=len(out['rows']),returned=sum(r['mode']=='returned' for r in out['rows']),
                         failed_comparisons=[r['index'] for r in out['rows'] if r.get('comparison_pass') is False],cpu=out['cpu_seconds_including_imports'])))

if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(20,22))
    if sys.argv[1]=='controls':controls()
    else:trained(int(sys.argv[2]))
